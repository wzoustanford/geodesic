"""
OpenVLA model wrappers for Geodesic.

Four nn.Module subclasses that wrap the published HuggingFace OpenVLA-7B
checkpoint (`openvla/openvla-7b`) into Geodesic-native components:

  - OpenVLAVisionModel    fused DINOv2 + SigLIP @ 224^2
  - OpenVLAProjector      vision_dim -> llm_dim MLP
  - OpenVLALanguageModel  Llama-2-7B causal LM
  - OpenVLAPolicyModel    orchestrates multimodal forward + predict_action

Set HF_TOKEN in your environment before first use; the openvla/openvla-7b
weights are gated and require accepting the LICENSE on HuggingFace.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor

IGNORE_INDEX = -100
DEFAULT_OPENVLA_MODEL_ID = "openvla/openvla-7b"


_HF_REGISTERED = False


def _register_openvla_with_hf() -> None:
    """Idempotently register OpenVLA classes with HF Auto* (needed once per process)."""
    global _HF_REGISTERED
    if _HF_REGISTERED:
        return
    from transformers import AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    _HF_REGISTERED = True


class OpenVLAVisionModel(nn.Module):
    """Wraps PrismaticVisionBackbone (fused DINOv2 + SigLIP @ 224^2)."""

    def __init__(self, hf_vision_backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = hf_vision_backbone
        self.embed_dim: int = hf_vision_backbone.embed_dim

    @property
    def num_patches(self) -> int:
        return self.backbone.featurizer.patch_embed.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: [B, 6, 224, 224] -> patch_features: [B, P, vision_dim]."""
        return self.backbone(pixel_values)


class OpenVLAProjector(nn.Module):
    """Wraps PrismaticProjector (vision_dim -> llm_dim 3-layer MLP w/ GELU)."""

    def __init__(self, hf_projector: nn.Module) -> None:
        super().__init__()
        self.projector = hf_projector
        self.vision_dim: int = hf_projector.vision_dim
        self.llm_dim: int = hf_projector.llm_dim

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """patch_features: [B, P, vision_dim] -> [B, P, llm_dim]."""
        return self.projector(patch_features)


class OpenVLALanguageModel(nn.Module):
    """Wraps the underlying causal LM (Llama-2-7B for openvla/openvla-7b)."""

    def __init__(self, hf_language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = hf_language_model

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def forward(self, **kwargs):
        """Pass-through to the HF causal LM; expects inputs_embeds + attention_mask + labels."""
        return self.language_model(**kwargs)


class OpenVLAPolicyModel(nn.Module):
    """Orchestrates vision -> projector -> concat -> LM, plus predict_action."""

    def __init__(
        self,
        vision: OpenVLAVisionModel,
        projector: OpenVLAProjector,
        language: OpenVLALanguageModel,
        processor: Any,
        norm_stats: Dict[str, Any],
        n_action_bins: int = 256,
        hf_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vision, self.projector, self.language = vision, projector, language
        self.processor = processor
        self.norm_stats = norm_stats
        # Single shared discrete<->continuous translator used by both this module
        # (predict_action's decode) and the agent (CE-loss mask + L1 diagnostic).
        # ActionTokenizer also provides `__call__` (continuous->7 token strings)
        # for the data-pipeline round.
        from prismatic.vla.action_tokenizer import ActionTokenizer
        self.action_tokenizer = ActionTokenizer(processor.tokenizer, bins=n_action_bins)
        # Held via object.__setattr__ so nn.Module doesn't register `hf_model` as
        # an extra submodule (its params are aliases of vision/projector/language
        # and we'd otherwise double-count them in self.parameters()).
        object.__setattr__(self, "_hf_model", hf_model)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_OPENVLA_MODEL_ID,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
    ) -> "OpenVLAPolicyModel":
        _register_openvla_with_hf()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if device is not None:
            hf = hf.to(device)
        vision = OpenVLAVisionModel(hf.vision_backbone)
        projector = OpenVLAProjector(hf.projector)
        language = OpenVLALanguageModel(hf.language_model)
        return cls(
            vision,
            projector,
            language,
            processor=processor,
            norm_stats=hf.norm_stats,
            n_action_bins=hf.config.n_action_bins,
            hf_model=hf,
        )

    # ---- training forward --------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **lm_kwargs,
    ):
        patch_features = self.vision(pixel_values)
        projected_patches = self.projector(patch_features)
        P = projected_patches.shape[1]

        text_embeds = self.language.get_input_embeddings()(input_ids)
        
        multimodal_embeds = torch.cat(
            [text_embeds[:, :1], projected_patches, text_embeds[:, 1:]], dim=1
        )
        ## [ todo ] double check the extra dimension in the openvla code
        """
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )
        """

        patch_mask = torch.full(
            (projected_patches.shape[0], P),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        multimodal_labels = None
        if labels is not None:
            patch_labels = torch.full(
                (projected_patches.shape[0], P),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[:, :1], patch_labels, labels[:, 1:]], dim=1
            )

        return self.language(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_mask,
            labels=multimodal_labels,
            **lm_kwargs,
        )

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        prompt: str,
        image,
        *,
        unnorm_key: Optional[str] = None,
        do_sample: bool = False,
    ) -> np.ndarray:
        """Single-sample inference: text prompt + PIL image -> unnormalized action."""
        if self._hf_model is None:
            raise RuntimeError(
                "predict_action requires an HF model instance; construct via "
                "OpenVLAPolicyModel.from_pretrained(...)."
            )
        device = next(self.parameters()).device
        inputs = self.processor(prompt, image).to(device, torch.bfloat16)
        input_ids = inputs["input_ids"]
        # OpenVLA quirk: ensure trailing space-token id 29871 after "Out:".
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                [input_ids, torch.tensor([[29871]], device=input_ids.device)], dim=1
            )
            inputs["input_ids"] = input_ids

        action_dim = self._get_action_dim(unnorm_key)
        generated = self._hf_model.generate(
            **inputs, max_new_tokens=action_dim, do_sample=do_sample
        )
        action_token_ids = generated[0, -action_dim:].cpu().numpy()

        normalized = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids)

        stats = self._get_action_stats(unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        return np.where(
            mask, 0.5 * (normalized + 1) * (high - low) + low, normalized
        )

    # ---- norm-stats helpers (mirror OpenVLAForActionPrediction) ------------
    @staticmethod
    def _check_unnorm_key(
        norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]
    ) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Model was trained on >1 dataset; pass unnorm_key from: {list(norm_stats)}"
            )
            return next(iter(norm_stats))
        assert unnorm_key in norm_stats, (
            f"unnorm_key={unnorm_key!r} not in {list(norm_stats)}"
        )
        return unnorm_key

    def _get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        k = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[k]["action"]["q01"])

    def _get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        k = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[k]["action"]
