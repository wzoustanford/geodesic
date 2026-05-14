"""
Data config for VLA supervised fine-tuning on LIBERO-Spatial.

Consumed by training scripts that construct `RLDSDataset` + `RLDSBatchTransform`
+ `PaddedCollatorForActionPrediction` + `VLAOrchestrator`. The values here are
the *only* place LIBERO paths and mix names should appear — callers read from
this module rather than hardcoding paths.

LIBERO sub-mix:
  - "libero_spatial_no_noops"  (this file)
  - "libero_object_no_noops"   -> configs/vla_libero_object_data_config.py
  - "libero_goal_no_noops"     -> configs/vla_libero_goal_data_config.py
  - "libero_10_no_noops"       -> configs/vla_libero_10_data_config.py

Data layout assumed on disk (cloned via Step 9 in
mds/vla_v1_with_openvla_dep_remote_setup.md):
    <DATA_ROOT_DIR>/libero_spatial_no_noops/1.0.0/{dataset_info.json,*.tfrecord-*}

All defaults mirror `../openvla/vla-scripts/finetune.py` (fine-tuning, not
from-scratch pretraining). UNNORM_KEY is intentionally NOT in this file —
it depends on which checkpoint the caller loads (base OXE-trained vs.
LIBERO-fine-tuned) and belongs to inference/eval config, not data config.
"""
import os
from pathlib import Path

# === Path resolution ===
# Override with VLA_DATA_ROOT env var if your LIBERO clone lives elsewhere
# (e.g. /scratch on a cluster box). Absolute path so RLDSDataset construction
# works regardless of CWD at runtime.
DATA_ROOT_DIR: Path = Path(
    os.environ.get("VLA_DATA_ROOT", str(Path.home() / "code" / "modified_libero_rlds"))
)
DATA_MIX: str = "libero_spatial_no_noops"

# === RLDSDataset construction kwargs ===
# (224, 224) matches OpenVLA-7B's "dinosiglip-vit-so-224px" backbone
# (configuration_prismatic.py:22). Finetune.py derives this from
# `vla.config.image_sizes` at runtime; we hardcode since this config is per-model.
RESIZE_RESOLUTION: tuple = (224, 224)
SHUFFLE_BUFFER_SIZE: int = 100_000              # finetune.py:93 (reduce if OOM)
TRAIN: bool = True
IMAGE_AUG: bool = True                          # finetune.py:92 (fine-tuning default)

# === DataLoader / collator params ===
TRAIN_BATCH_SIZE: int = 16                      # finetune.py:87
# CRITICAL: must be 0 when using RLDSDataset. TFDS pipelines run their own
# multi-threading (traj_read_threads, num_parallel_calls inside RLDSDataset);
# torch DataLoader workers on top fork the TFDS iterator, causing duplicate
# data and resource contention. See finetune.py:237 + its inline comment.
NUM_WORKERS: int = 0
