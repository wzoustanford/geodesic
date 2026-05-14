"""
Synthetic data fixtures for VLA smoke tests.

VLASchemaFixture emits per-sample dicts that match the schema
RLDSBatchTransform will eventually produce, so we can exercise the VLA
model's forward/backward without TFDS, OXE shards, or HuggingFace weights.

Per-sample schema (consumed by PaddedCollatorForActionPrediction next round):
  pixel_values : Tensor[vision_channels, H, W]    float32
  input_ids    : LongTensor[L]                     int64
  labels       : LongTensor[L]                     int64; IGNORE_INDEX outside
                                                  the trailing (action_dim+1)
                                                  action+EOS positions
  dataset_name : bytes                             (b"fixture")

Defaults reflect OpenVLA-7B: Llama-2 vocab=32000, n_action_bins=256,
action_dim=7, dual DINOv2+SigLIP vision tower at 224^2 channel-stacked
into 6 channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

IGNORE_INDEX = -100

# OpenVLA-7B defaults (mirrors mds/open_vla_model_data_investigation.md:618,898).
DEFAULT_VOCAB_SIZE = 32000        # Llama-2 base tokenizer vocab size
DEFAULT_N_ACTION_BINS = 256       # ActionTokenizer bin count
DEFAULT_BOS_ID = 1
DEFAULT_EOS_ID = 2


class VLASchemaFixture(Dataset):
    """Synthetic map-style Dataset matching the RLDSBatchTransform per-sample schema."""

    def __init__(
        self,
        num_samples: int = 64,
        action_dim: int = 7,
        prompt_len_range: Tuple[int, int] = (24, 36),
        vision_channels: int = 6,
        vision_size: int = 224,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_action_bins: int = DEFAULT_N_ACTION_BINS,
        bos_token_id: int = DEFAULT_BOS_ID,
        eos_token_id: int = DEFAULT_EOS_ID,
        dataset_name: bytes = b"fixture",
        seed: int = 0,
    ) -> None:
        if prompt_len_range[0] < action_dim + 2:
            raise ValueError(
                f"prompt_len_range[0]={prompt_len_range[0]} must be >= "
                f"action_dim+2={action_dim+2} (need >=1 prompt token + "
                f"{action_dim} action tokens + EOS)."
            )
        if n_action_bins + 1 >= vocab_size:
            raise ValueError(
                f"vocab_size={vocab_size} must exceed n_action_bins+1={n_action_bins+1}."
            )
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.prompt_len_range = prompt_len_range
        self.vision_channels = vision_channels
        self.vision_size = vision_size
        self.vocab_size = vocab_size
        self.n_action_bins = n_action_bins
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dataset_name = dataset_name

        # First id used for action tokens. Action ids occupy
        # [action_token_begin_idx, vocab_size - 1] (the last n_action_bins ids).
        # NOTE: this differs by 1 from prismatic's ActionTokenizer.action_token_begin_idx
        # (../openvla/prismatic/vla/action_tokenizer.py:325 defines it as
        # vocab_size - n_bins - 1, paired with a strict `id > begin_idx` mask). The
        # set of valid action ids is identical; the name here describes the first
        # action id rather than the mask threshold.
        self.action_token_begin_idx = vocab_size - n_action_bins
        self._seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self._seed + idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)
        rng = self._rng(idx)

        L = int(rng.integers(self.prompt_len_range[0], self.prompt_len_range[1] + 1))
        n_action = self.action_dim
        n_prompt = L - n_action - 1   # action tokens + EOS occupy last (action_dim+1)

        # Prompt tokens: any non-action id; position 0 reserved for BOS.
        prompt_body = rng.integers(
            low=3,                                  # skip BOS(1) / EOS(2) / unk(0)
            high=self.action_token_begin_idx,       # exclusive -> max id = begin_idx - 1
            size=n_prompt - 1,
        )
        prompt_ids = np.concatenate([np.array([self.bos_token_id]), prompt_body])

        # Action tokens: ids in [action_token_begin_idx, vocab_size - 1].
        action_ids = rng.integers(
            low=self.action_token_begin_idx,
            high=self.vocab_size,                   # exclusive -> max id = vocab - 1
            size=n_action,
        )

        input_ids_np = np.concatenate([prompt_ids, action_ids, [self.eos_token_id]])
        assert input_ids_np.shape == (L,), (input_ids_np.shape, L)

        input_ids = torch.from_numpy(input_ids_np.astype(np.int64))
        labels = input_ids.clone()
        labels[: -(n_action + 1)] = IGNORE_INDEX

        # Post-Normalize values can be negative; randn() is a fair stand-in for
        # real DINOv2+SigLIP normalized output.
        pixel_values = torch.from_numpy(
            rng.standard_normal(
                (self.vision_channels, self.vision_size, self.vision_size),
                dtype=np.float32,
            )
        )

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.dataset_name,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    """Pad variable-length VLA samples into a batch dict.

    Mirrors `../openvla/prismatic/util/data_utils.py:94-142`. Right-pads
    `input_ids` with `pad_token_id`, `labels` with IGNORE_INDEX. Derives
    `attention_mask` from `input_ids != pad_token_id`. Stacks `pixel_values`.
    """
    model_max_length: int = 2048               # Llama-2 default
    pad_token_id: int = 32000                  # OpenVLA processor's added pad token
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids    = [inst["input_ids"]    for inst in instances]
        labels       = [inst["labels"]       for inst in instances]
        pixel_values = [inst["pixel_values"] for inst in instances]
        dataset_names = (
            [inst["dataset_name"] for inst in instances]
            if "dataset_name" in instances[0] else None
        )

        assert self.padding_side == "right", f"Invalid padding_side `{self.padding_side}`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels    = pad_sequence(labels,    batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.model_max_length]
        labels    = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        assert all(pv is not None for pv in pixel_values), "VLA training requires multimodal inputs."
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        out = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            out["dataset_names"] = dataset_names
        return out


@dataclass
class RLDSBatchTransform:
    """Per-sample converter: raw RLDS dict -> {pixel_values, input_ids, labels, dataset_name}.

    Mirrors `../openvla/prismatic/vla/datasets/datasets.py:30-67`. Applies the
    action tokenizer, prompt builder, and image transform; injects the -100
    label mask so HF causal-LM loss only counts the trailing (action_dim + 1)
    positions (action_dim action ids + EOS).
    """
    action_tokenizer:   Any    # prismatic.vla.action_tokenizer.ActionTokenizer
    base_tokenizer:     Any    # transformers.PreTrainedTokenizerBase
    image_transform:    Any    # callable: PIL.Image -> Tensor[C, H, W]
    prompt_builder_fn:  Any    # type[PromptBuilder]; called as fn("openvla")
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image
        dataset_name = rlds_batch["dataset_name"]
        action = rlds_batch["action"][0]
        img  = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {lang}?")
        prompt_builder.add_turn("gpt", self.action_tokenizer(action))

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels    = torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # IMPORTANT: HF causal-LM.forward(labels=...) shifts internally — do NOT
        # shift here. Mask everything before the trailing action_dim + 1 positions.
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
        )


class RLDSDataset(IterableDataset):
    """IterableDataset wrapping prismatic's RLDS/TFDS interleaved pipeline.

    Mirrors `../openvla/prismatic/vla/datasets/datasets.py:70-154`. `data_mix`
    may be an OXE_NAMED_MIXTURES key (e.g. "libero_spatial_no_noops",
    "oxe_magic_soup") or a single-dataset name (treated as a 1.0-weight mix).

    Requires the `prismatic` package importable (currently from `../openvla/`
    on PYTHONPATH; see vla_models.py:31-47 for the HF Auto-class registration).
    Imports are lazy so `import vla_datasets` works without prismatic.
    """

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        from prismatic.vla.datasets.rlds.oxe import (
            OXE_NAMED_MIXTURES,
            get_oxe_dataset_kwargs_and_weights,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        self.data_root_dir   = data_root_dir
        self.data_mix        = data_mix
        self.batch_transform = batch_transform

        mixture_spec = (
            OXE_NAMED_MIXTURES[data_mix]
            if data_mix in OXE_NAMED_MIXTURES
            else [(data_mix, 1.0)]              # single-dataset mix at full weight
        )

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                    # single-step; bump for action chunking later
                future_action_window_size=0,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        if image_aug:
            rlds_config["frame_transform_kwargs"]["image_augment_kwargs"] = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )

        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError(
            "IterableDataset does not implement map-style __getitem__; see __iter__."
        )
