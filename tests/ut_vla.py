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

import numpy as np
import torch
from PIL import Image
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
    # unnorm_key picks which dataset's q01/q99 stats to use when denormalizing
    # predicted actions at inference time (sub-check 4 below). Used in
    # agent.select_actions -> policy.predict_action.
    #
    # Caveat: "bridge_orig" works for openvla/openvla-7b (the base). If you point
    # the agent at openvla/openvla-7b-finetuned-libero-spatial later, swap to
    # "libero_spatial_no_noops". The base-vs-finetuned distinction is precisely
    # what makes UNNORM_KEY an inference-side knob (the data config can't know
    # which checkpoint the caller picks).
    agent = OpenVLAAgent(
        use_lora=True, lora_rank=8,
        grad_accumulation_steps=1, grad_clip=1.0,
        unnorm_key="bridge_orig",
    )
    # Disable checkpoint writes during the smoke test.
    agent.save = lambda *a, **kw: None

    f = VLASchemaFixture(num_samples=8, prompt_len_range=(24, 28), seed=0)
    collator = PaddedCollatorForActionPrediction()   # OpenVLA-7B defaults
    loader = DataLoader(f, batch_size=2, collate_fn=collator)
    batch = next(iter(loader))

    # Snapshot a few LoRA params (CPU copies) so we can verify update() actually
    # moves them after the loop below. Complements the loss-decrease check:
    #   - param movement       => optimizer.step actually fired
    #   - loss decreasing      => gradients point in the right direction
    # Either one passing alone is insufficient (e.g. params could move under
    # an accidental sign-flipped autograd hookup while loss goes the wrong way).
    sampled_before = {}
    for n, p in agent.policy.named_parameters():
        if p.requires_grad:
            sampled_before[n] = p.detach().clone().to("cpu")
            if len(sampled_before) >= 5:
                break

    # --- sub-check 1: agent.update() — forward + backward + optimizer.step all
    #     functional. Run N updates on the same batch and assert (a) loss drops
    #     and (b) sampled trainable params moved. With 27M LoRA params and a
    #     2-sample batch the model should memorize trivially.
    N = 3
    losses = []
    for _ in range(N):
        metrics = agent.update(batch)
        losses.append(metrics["loss"])
        assert set(metrics) == {"loss", "action_token_acc", "action_l1"}, set(metrics)
        assert math.isfinite(metrics["loss"]), metrics["loss"]
    assert losses[-1] < losses[0], (
        f"loss should decrease over {N} updates on a fixed batch; got {losses}"
    )
    moved = any(
        not torch.equal(p.detach().to("cpu"), sampled_before[n])
        for n, p in agent.policy.named_parameters()
        if n in sampled_before
    )
    assert moved, "no sampled trainable param changed after update — optimizer.step didn't fire"
    print(f"  update() x{N}: losses={['{:.4f}'.format(l) for l in losses]}  "
          f"(loss {losses[0]:.4f} -> {losses[-1]:.4f}, params moved)")

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

    # --- sub-check 4: agent.select_actions() — inference / denorm path ---
    # Exercises the OTHER code path: policy.predict_action -> hf.generate ->
    # action-token decode -> denormalize via norm_stats[unnorm_key]. This is the
    # only sub-check that touches `unnorm_key`; the training-time sub-checks
    # above never read it.
    fake_obs = {
        "image": Image.fromarray(
            np.asarray(np.random.rand(224, 224, 3) * 255, dtype=np.uint8)
        ),
        "instruction": "pick up the spoon",
    }
    action = agent.select_actions(fake_obs)
    assert isinstance(action, np.ndarray), type(action)
    assert action.shape == (7,), action.shape          # OpenVLA-7B is 7-DoF EEF
    assert np.all(np.isfinite(action)), action
    print(f"  select_actions(): action={action}")

    print(f"[ut1_vla_forward_backward] OK")


if __name__ == "__main__":
    ut1_vla_forward_backward()
