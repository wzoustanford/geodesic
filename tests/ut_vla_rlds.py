"""
End-to-end test of the real LIBERO RLDS pipeline:
  RLDSDataset (real shards) -> RLDSBatchTransform -> PaddedCollator
    -> VLAOrchestrator -> OpenVLAAgent.update -> OpenVLAPolicyModel

The key boundary this verifies: the real RLDS pipeline emits per-sample dicts
matching the schema VLASchemaFixture promises (same keys, same dtypes, same
-100 mask invariant). If this passes, the fixture-based ut_vla.py is a
legitimate proxy for "real RLDS -> orchestrator -> agent."

Requires:
  - CUDA (training is impractical on CPU)
  - HF_TOKEN with access to openvla/openvla-7b (gated)
  - prismatic importable (../openvla on PYTHONPATH)
  - dlimp + tensorflow + tensorflow_datasets
  - LIBERO RLDS shards on disk at configs.vla_libero_spatial_data_config.DATA_ROOT_DIR
    (override default path via VLA_DATA_ROOT env var)

Skips cleanly (exit 0) if any prerequisite is missing, so the file is safe to
import and parse anywhere.
"""
import os

# Silence TF Grappler PredictCost warnings — the cost-estimator heuristic chokes
# on unknown-shape CropAndResize inputs during image_aug, but the ops still
# execute correctly. Must be set BEFORE tensorflow is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import importlib.util
import math
import sys

import torch
from torch.utils.data import DataLoader

import configs.vla_libero_spatial_data_config as data_cfg
from vla_datasets import (
    IGNORE_INDEX,
    PaddedCollatorForActionPrediction,
    RLDSBatchTransform,
    RLDSDataset,
)


def _gate() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; OpenVLA-7B is impractical on CPU.")
        sys.exit(0)
    if not os.environ.get("HF_TOKEN"):
        print("[SKIP] HF_TOKEN not set; openvla/openvla-7b is gated.")
        sys.exit(0)
    for mod in ("prismatic", "dlimp", "tensorflow", "tensorflow_datasets"):
        if importlib.util.find_spec(mod) is None:
            print(f"[SKIP] {mod} not importable (see remote setup md Step 4).")
            sys.exit(0)
    shard_info = data_cfg.DATA_ROOT_DIR / data_cfg.DATA_MIX / "1.0.0" / "dataset_info.json"
    if not shard_info.exists():
        print(f"[SKIP] LIBERO TFDS shards not found at {shard_info}")
        print("       Clone via remote setup md Step 9, or set VLA_DATA_ROOT to override.")
        sys.exit(0)


def ut1_vla_rlds_train():
    _gate()

    # Lazy imports — keep file safe to parse on machines without prismatic / GPU.
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from vla_agent import OpenVLAAgent
    from vla_orchestrator import VLAOrchestrator

    print("Loading OpenVLA-7B (one-time, ~14 GB bf16)...", flush=True)
    agent = OpenVLAAgent(
        use_lora=True, lora_rank=8,
        grad_accumulation_steps=1, grad_clip=1.0,
    )
    agent.save = lambda *a, **kw: None     # disable checkpoint writes during smoke

    # Build the per-sample RLDS converter using the loaded model's pieces.
    batch_transform = RLDSBatchTransform(
        action_tokenizer=agent.policy.action_tokenizer,
        base_tokenizer=agent.policy.processor.tokenizer,
        image_transform=agent.policy.processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
    )

    print(
        f"Constructing RLDSDataset({data_cfg.DATA_MIX}) from {data_cfg.DATA_ROOT_DIR}...",
        flush=True,
    )
    rlds_dataset = RLDSDataset(
        data_root_dir=data_cfg.DATA_ROOT_DIR,
        data_mix=data_cfg.DATA_MIX,
        batch_transform=batch_transform,
        resize_resolution=data_cfg.RESIZE_RESOLUTION,
        # Override config's 100k buffer for fast smoke turnaround; production
        # training keeps SHUFFLE_BUFFER_SIZE per the config.
        shuffle_buffer_size=100,
        train=data_cfg.TRAIN,
        image_aug=data_cfg.IMAGE_AUG,
    )

    # --- sub-check 1: per-sample schema matches the fixture's contract ---
    # This is THE boundary test. If this passes, fixture-based ut_vla.py is a
    # legitimate proxy for the real pipeline.
    A = 7  # LIBERO is 7-DoF EEF (../openvla/prismatic/vla/datasets/rlds/oxe/configs.py:650)
    sample = next(iter(rlds_dataset))
    expected_keys = {"pixel_values", "input_ids", "labels", "dataset_name"}
    assert set(sample.keys()) >= expected_keys, set(sample.keys())
    assert sample["pixel_values"].shape == (6, 224, 224), sample["pixel_values"].shape
    assert sample["pixel_values"].dtype == torch.float32, sample["pixel_values"].dtype
    assert sample["input_ids"].dim() == 1 and sample["input_ids"].dtype == torch.int64
    assert sample["labels"].shape == sample["input_ids"].shape
    # -100 mask invariant: last (A+1) positions are real, everything before is masked.
    assert (sample["labels"][: -(A + 1)] == IGNORE_INDEX).all()
    assert (sample["labels"][-(A + 1) :] != IGNORE_INDEX).all()
    assert isinstance(sample["dataset_name"], bytes)
    print(
        f"  per-sample schema OK: L={sample['input_ids'].shape[0]}, "
        f"dataset_name={sample['dataset_name']!r}"
    )

    # --- sub-check 2: DataLoader + collator emit a model-shaped batch ---
    # Smoke-test batch size: smaller than data_cfg.TRAIN_BATCH_SIZE (16) to fit
    # a 40GB A100. openvla's batch=16 default (finetune.py:87) assumes 8x A100
    # 80GB via DDP (finetune.py:204), so effective per-GPU batch is 2. On a
    # single 40GB A100 with the full 7B forward+backward, batch=16 OOMs.
    # Production training should use data_cfg.TRAIN_BATCH_SIZE with
    # grad_accumulation_steps to recover the effective batch size.
    SMOKE_BATCH_SIZE = 2

    collator = PaddedCollatorForActionPrediction()
    loader = DataLoader(
        rlds_dataset,
        batch_size=SMOKE_BATCH_SIZE,
        num_workers=data_cfg.NUM_WORKERS,   # MUST be 0 per finetune.py:237
        collate_fn=collator,
    )
    batch = next(iter(loader))
    B = SMOKE_BATCH_SIZE
    assert batch["pixel_values"].shape == (B, 6, 224, 224), batch["pixel_values"].shape
    assert batch["input_ids"].dim() == 2 and batch["input_ids"].shape[0] == B
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["labels"].shape == batch["input_ids"].shape
    print(
        f"  collated batch shape: pixel_values={tuple(batch['pixel_values'].shape)}, "
        f"input_ids={tuple(batch['input_ids'].shape)}"
    )

    # --- sub-check 3: training updates run end-to-end on real LIBERO samples ---
    # We do NOT assert loss drops here. The loader yields different batches per
    # step (real data, not the fixed-batch memorization setup in ut_vla.py), so
    # loss noise dominates over 2 steps. The fixture UT already covers the
    # "loss drops on memorizable batch" invariant; this UT covers "real-data
    # wiring works end-to-end with no crashes and finite metrics."
    orch = VLAOrchestrator(
        agent,
        loader,
        max_steps=2,
        log_every=1,
        save_every=10**9,
    )
    final = orch.start()
    assert "loss" in final and math.isfinite(final["loss"]), final
    print(f"[ut1_vla_rlds_train] OK  (real LIBERO end-to-end)")


if __name__ == "__main__":
    ut1_vla_rlds_train()
