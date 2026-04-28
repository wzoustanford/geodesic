"""
OpenVLA-7B imitation-learning agent for Geodesic.

Wraps OpenVLAPolicyModel in the Agent ABC so it slots into the Orchestrator
(which drives .update(batch)) exactly like SACAgent / JAXSACAgent. The
optimizer and backward pass live inside .update(); the outer trainer never
touches gradients.

Training objective: next-token CE on the 7 action tokens + </s>, implemented
via labels == -100 elsewhere so HF's LlamaForCausalLM computes the right loss.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW

from agents import Agent
from vla_models import (
    DEFAULT_OPENVLA_MODEL_ID,
    IGNORE_INDEX,
    OpenVLAPolicyModel,
)


class OpenVLAAgent(Agent):
    """Imitation-learning agent around OpenVLAPolicyModel; update() does next-token CE."""

    def __init__(
        self,
        model_id: str = DEFAULT_OPENVLA_MODEL_ID,
        unnorm_key: Optional[str] = None,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.0,
        lr: float = 5e-4,
        grad_accumulation_steps: int = 1,
        grad_clip: Optional[float] = 1.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        save_dir: str = "./checkpoints/",
        experiment_prefix: str = "openvla_",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.unnorm_key = unnorm_key
        self.grad_accumulation_steps = grad_accumulation_steps
        self.grad_clip = grad_clip
        self.save_dir, self.experiment_prefix = save_dir, experiment_prefix
        os.makedirs(self.save_dir, exist_ok=True)

        # Build policy (loads HF weights under the hood; requires HF_TOKEN).
        self.policy = OpenVLAPolicyModel.from_pretrained(
            model_id, torch_dtype=torch_dtype, device=device,
        )

        # PEFT replaces every nn.Linear matching `target_modules` with a
        # lora.Linear that holds the frozen original + low-rank A/B adapters,
        # and flips requires_grad=False on all non-adapter params. Replacement
        # is in place at parent modules — since policy.{vision,projector,language}
        # share submodule objects with self.policy._hf_model, the swap is visible
        # everywhere. We keep the wrapper for save_pretrained / multi-adapter mgmt.
        self._peft_wrapper = None
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, 16),
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self._peft_wrapper = get_peft_model(self.policy._hf_model, lora_config)
            self._peft_wrapper.print_trainable_parameters()

        # Optimizer over params that survived LoRA's requires_grad=False freeze.
        # If use_lora=False, every param is trainable -> full finetune.
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=lr)
        self.optimizer.zero_grad()  # match reference's pre-loop zero_grad

        # action_token_begin_idx now lives on policy.action_tokenizer (Block 8).
        self.training_step = 0

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def select_actions(self, obs: Dict[str, Any]) -> np.ndarray:
        """obs: {'image': PIL.Image, 'instruction': str}. Returns [action_dim] np.ndarray."""
        prompt = (
            f"In: What action should the robot take to {obs['instruction']}?\nOut: "
        )
        return self.policy.predict_action(
            prompt,
            obs["image"],
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )

    # ---- training update ---------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Next-token CE on action-token positions (everything else masked by -100).

        batch (from PaddedCollatorForActionPrediction, next round):
          pixel_values  : [B, 6, 224, 224]  bf16
          input_ids     : [B, L]             int64
          attention_mask: [B, L]             bool
          labels        : [B, L]             int64, -100 outside action positions
        """
        dev = self.device
        # Policy weights are already bf16 (loaded via from_pretrained(torch_dtype=
        # torch.bfloat16)), so this autocast is largely a no-op for parameter ops.
        # We keep it for parity with the reference recipe and to cast any stray
        # fp32 inputs (e.g., a caller passing fp32 pixel_values) on entry.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.policy(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                labels=batch["labels"].to(dev),
            )
            loss = out.loss

        normalized_loss = loss / self.grad_accumulation_steps
        normalized_loss.backward()

        is_gradient_step = (self.training_step + 1) % self.grad_accumulation_steps == 0
        if is_gradient_step:
            if self.grad_clip is not None:
                # The published OpenVLA recipe does not clip gradients, but the
                # research playbook recommends grad_clip=1.0 and Geodesic's SAC
                # path uses clipping uniformly. Cheap insurance against rare
                # large-gradient steps from the bf16 LM during early training.
                trainable = [p for p in self.policy.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.training_step += 1

        # Diagnostics (match finetune.py:270-286 exactly).
        with torch.no_grad():
            P = self.policy.vision.num_patches
            action_logits = out.logits[:, P:-1]              # strip patches + last pos
            action_preds = action_logits.argmax(dim=-1)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > self.policy.action_tokenizer.action_token_begin_idx
            n = mask.sum().clamp_min(1)
            action_token_acc = ((action_preds == action_gt) & mask).sum().float() / n

            # L1 on decoded continuous actions (single shared tokenizer).
            decode = self.policy.action_tokenizer.decode_token_ids_to_actions
            pred_c = decode(action_preds[mask].detach().cpu().numpy())
            gt_c = decode(action_gt[mask].detach().cpu().numpy())
            l1 = float(np.mean(np.abs(pred_c - gt_c))) if pred_c.size else 0.0

        return {
            "loss": float(loss.item()),
            "action_token_acc": float(action_token_acc.item()),
            "action_l1": l1,
        }

    # ---- checkpoints -------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        """Save only requires_grad=True parameters (LoRA adapters in the default
        recipe). The base 14 GB bf16 model is recovered at load time by
        re-running OpenVLAPolicyModel.from_pretrained, which re-pulls the
        published checkpoint from HuggingFace. This keeps checkpoints small
        (~100 MB at lora_rank=32) and makes resumed runs deterministic w.r.t.
        the published weights — at the cost of needing HF reachability at load.

        TODO(later): add a `full_checkpoint=False` kwarg that, when True, also
        persists the frozen base weights so checkpoints survive HF outages /
        weight removal. Defer until use_lora=False becomes a real workflow.
        """
        path = path or self._default_path("latest")
        trainable = {
            name: p.detach().cpu()
            for name, p in self.policy.named_parameters()
            if p.requires_grad
        }
        torch.save(
            {
                "trainable_state_dict": trainable,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "unnorm_key": self.unnorm_key,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load checkpoint produced by .save(). Assumes the agent has already
        been constructed (so the base HF weights are loaded and any LoRA
        adapters have been allocated); only `requires_grad=True` tensors are
        overwritten. If a saved tensor's name is missing from the current
        model (e.g., LoRA disabled at load time but enabled at save time),
        we raise rather than silently mismatch.
        """
        path = path or self._default_path("latest")
        ckpt = torch.load(path, map_location=self.device)
        params = dict(self.policy.named_parameters())
        for name, tensor in ckpt["trainable_state_dict"].items():
            if name not in params:
                raise KeyError(
                    f"Checkpoint contains parameter {name!r} not present in current "
                    f"policy; LoRA config likely changed since save."
                )
            params[name].data.copy_(tensor.to(self.device))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step = ckpt["training_step"]
        self.unnorm_key = ckpt.get("unnorm_key", self.unnorm_key)
        print(f"Model loaded from {path}")

    def train(self):
        raise NotImplementedError(
            "OpenVLAAgent.train() not implemented; drive via Orchestrator.update()."
        )

    def _default_path(self, postfix: str) -> str:
        return os.path.join(self.save_dir, f"{self.experiment_prefix}{postfix}.pt")
