"""
End-to-end forward/backward smoke test for the VLA training cycle.

Exercises:
  VLASchemaFixture -> DataLoader (inline collate) -> VLAOrchestrator
      -> OpenVLAAgent.update + .validate
      -> OpenVLAPolicyModel.forward + backward + optimizer step

Requires:
  - CUDA-capable device (OpenVLA-7B is impractical on CPU)
  - HF_TOKEN env var with access to openvla/openvla-7b (gated repo)

Skips cleanly (exit 0) when either prerequisite is missing, so the file is
safe to import and parse anywhere.
"""
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from vla_datasets import PaddedCollatorForActionPrediction, VLASchemaFixture


def _gate() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; OpenVLA-7B is impractical on CPU.")
        sys.exit(0)
    if not os.environ.get("HF_TOKEN"):
        print("[SKIP] HF_TOKEN not set; openvla/openvla-7b is gated.")
        sys.exit(0)


def ut1_vla_forward_backward():
    _gate()

    # Import here so non-GPU machines can still parse/run the file (and SKIP).
    from vla_agent import OpenVLAAgent
    from vla_orchestrator import VLAOrchestrator

    print("Loading OpenVLA-7B (one-time, ~14 GB bf16)...", flush=True)
    agent = OpenVLAAgent(use_lora=True, lora_rank=8, grad_accumulation_steps=1, grad_clip=1.0)
    # Disable checkpoint writes during the smoke test.
    agent.save = lambda *a, **kw: None

    f = VLASchemaFixture(num_samples=8, prompt_len_range=(24, 28), seed=0)
    collator = PaddedCollatorForActionPrediction()   # OpenVLA-7B defaults
    loader = DataLoader(f, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    # --- sub-check 1: agent.update() — forward + backward + metrics ---
    metrics = agent.update(batch)
    expected = {"loss", "action_token_acc", "action_l1"}
    assert set(metrics) == expected, set(metrics)
    assert math.isfinite(metrics["loss"]), metrics["loss"]
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in agent.policy.parameters()
        if p.requires_grad
    )
    assert has_grad, "no trainable param received a non-zero gradient"
    print(f"  update():   loss={metrics['loss']:.4f}  "
          f"action_token_acc={metrics['action_token_acc']:.4f}  "
          f"action_l1={metrics['action_l1']:.4f}")

    # --- sub-check 2: agent.validate() — forward-only, val_-prefixed metrics ---
    vmetrics = agent.validate(batch)
    vexpected = {"val_loss", "val_action_token_acc", "val_action_l1"}
    assert set(vmetrics) == vexpected, set(vmetrics)
    assert math.isfinite(vmetrics["val_loss"]), vmetrics["val_loss"]
    print(f"  validate(): val_loss={vmetrics['val_loss']:.4f}  "
          f"val_action_token_acc={vmetrics['val_action_token_acc']:.4f}  "
          f"val_action_l1={vmetrics['val_action_l1']:.4f}")

    # --- sub-check 3: VLAOrchestrator.start() — 2 train steps + 1 val step ---
    orch = VLAOrchestrator(
        agent,
        loader,
        max_steps=2,
        log_every=1,
        save_every=10**9,
        val_loader=loader,
        val_every=2,
        val_steps=1,
    )
    final = orch.start()
    assert "loss" in final and math.isfinite(final["loss"]), final
    print(f"[ut1_vla_forward_backward] OK")


if __name__ == "__main__":
    ut1_vla_forward_backward()
