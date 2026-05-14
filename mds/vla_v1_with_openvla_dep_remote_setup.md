# VLA v1 — remote GPU machine setup, from a clean pull

This is the end-to-end checklist for setting up `geodesic`'s VLA training stack
on a fresh GPU machine. It assumes you've already followed `pyproject.toml`
pinning on the laptop side (i.e. `transformers`, `peft`, `Pillow`, `tensorflow`,
`tensorflow_datasets`, `einops`, `timm`, `sentencepiece`, `dlimp` are committed
to `pyproject.toml` + `uv.lock`).

`openvla` itself is **not** declared as a `pyproject.toml` dependency — its
narrow version pins (`torch==2.2.0`, `transformers==4.40.1`, `peft==0.11.1`,
Python ≤ 3.10) are incompatible with `geodesic` (`torch>=2.10.0`, Python 3.12).
We treat `openvla` as a sibling repo on `PYTHONPATH` instead.

## Steps

```bash
# 1. Get both repos as siblings.
mkdir -p ~/code && cd ~/code
git clone <your-geodesic-remote-url> geodesic
git clone https://github.com/openvla/openvla.git openvla

# 2. Install uv if not present.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart shell

# 3. Sync geodesic's venv (uses geodesic/pyproject.toml + uv.lock from your commit).
cd ~/code/geodesic
uv sync

# 4. Install RLDS-only deps that are NOT in pyproject.toml.
#    Why excluded from pyproject.toml:
#      - dlimp pins tensorflow==2.15.0, which has no Python 3.12 wheels →
#        unresolvable in a 3.12 venv unless installed with --no-deps.
#      - tensorflow / tensorflow_datasets are large + only needed for the
#        RLDSDataset code path (fixture + model paths don't need them), so we
#        keep the default uv sync lean and install them here on the GPU box.
#
#    Skip this step if you only want to run fixture UTs (ut_vla_dataset_fixture.py)
#    or the model-only e2e UT (ut_vla.py against OpenVLA-7B). Run it before
#    constructing RLDSDataset.
uv pip install 'tensorflow>=2.15' 'tensorflow_datasets>=4.9'
uv pip install --no-deps 'dlimp @ git+https://github.com/moojink/dlimp_openvla'
uv pip install --no-deps 'tensorflow_graphics'

# 5. (Optional) Install flash-attn after editable sync, per openvla's README — only
#    needed if you want fast attention at training time.
#    uv pip install 'flash-attn==2.5.5' --no-build-isolation

# 6. Set PYTHONPATH so `import prismatic` finds ../openvla/prismatic.
#    Run this from inside ~/code/geodesic (or wherever you cloned geodesic);
#    `$(cd ../openvla && pwd)` resolves the sibling path to an absolute one
#    at echo-time, so .bashrc ends up with a hardcoded absolute path that
#    works regardless of future shell CWD.
echo "export PYTHONPATH=\$PYTHONPATH:$(cd ../openvla && pwd)" >> ~/.bashrc
source ~/.bashrc

# 7. (Optional) Make HF_TOKEN persist if you'll run ut_vla.py on this box.
echo 'export HF_TOKEN=hf_...' >> ~/.bashrc
source ~/.bashrc

## set these environment variables 
export HF_TOKEN="hf_..."

## the caches will store python packages, the HF VLA model, and the LIBERO data, so put them where you have HD storage; for LIBERO data, point to the cloned location below in step 9 
export UV_CACHE_DIR=/scratch/zouwil/.cache/uv/
export HF_HOME="/scratch/zouwil/.cache/huggingface/"
export VLA_DATA_ROOT='/scratch/zouwil/code/modified_libero_rlds/'

# 8. Smoke-test.
cd ~/code/geodesic
PYTHONPATH=$PYTHONPATH .venv/bin/python tests/ut_vla_dataset_fixture.py   # 4 UTs, all OK
PYTHONPATH=$PYTHONPATH .venv/bin/python tests/ut_vla.py                    # runs e2e if CUDA+HF_TOKEN present

# 9. Download LIBERO RLDS shards (~10 GB) — only if you'll construct RLDSDataset
#    for real training. Skip for fixture-only UTs or the model-only e2e UT.
#    Pre-built TFDS shards from the openvla team, drops cleanly under one parent.
cd ~/code
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
#   -> data_root_dir = ~/code/modified_libero_rlds
#   -> data_mix      = "libero_spatial_no_noops" | "libero_object_no_noops" |
#                      "libero_goal_no_noops"    | "libero_10_no_noops"
#   Resulting layout:
#     ~/code/modified_libero_rlds/
#     ├── libero_spatial_no_noops/1.0.0/{dataset_info.json,*.tfrecord-*}
#     ├── libero_object_no_noops/1.0.0/...
#     ├── libero_goal_no_noops/1.0.0/...
#     └── libero_10_no_noops/1.0.0/...
#
#   openvla gotchas if you hit them at training time (from openvla README
#   "VLA Troubleshooting"):
#     - "Could not load dataset_info.json" -> uv pip install 'tensorflow_datasets==4.9.3'
#     - "'DLataset' has no attribute 'traj_map'" ->
#         uv pip install --no-deps --force-reinstall \
#           'dlimp @ git+https://github.com/moojink/dlimp_openvla'
#
#   Full OXE Magic Soup (hundreds of GB) is a separate workflow via
#   https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh
#   — only needed for from-scratch pretraining or multi-dataset mixes.
```

For LIBERO / OXE data shards specifically, that's a separate `gsutil` / `wget`
step against the openvla data buckets — not part of the repo pull. Address that
when you actually want to wire up `RLDSDataset` for a training run.

## What this gets you

After these steps:

- Any future clean pull on a GPU box reduces to
  `git clone geodesic && git clone openvla && uv sync && export PYTHONPATH=...$HOME/code/openvla`.
- No version pin gymnastics. `uv` picks versions consistent with
  `torch>=2.10` / Python 3.12.
- `import prismatic` works via `PYTHONPATH`.
- `from prismatic.vla.action_tokenizer import ActionTokenizer` works.
- TFDS-backed `RLDSDataset` works once data shards are downloaded.
- `OpenVLAPolicyModel.from_pretrained` works (HF Hub serves weights + remote
  code via `trust_remote_code=True`).
