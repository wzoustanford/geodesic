"""
Smoke tests for vla_datasets.VLASchemaFixture.

Verifies the per-sample contract that downstream VLA training (collator,
model.forward, agent.update) depends on:
  ut1: schema keys, tensor shapes, dtypes, sequence length in declared range
  ut2: -100 label mask sits at the right positions (prefix masked, trailing
       action_dim+1 positions un-masked + equal to input_ids)
  ut3: BOS/EOS placement, action ids in [begin_idx, vocab-1], prompt ids in
       [3, begin_idx-1]
"""
import torch

from vla_datasets import IGNORE_INDEX, PaddedCollatorForActionPrediction, VLASchemaFixture


def ut1_fixture_schema():
    f = VLASchemaFixture(num_samples=16, seed=0)
    assert len(f) == 16, len(f)

    expected_keys = {"pixel_values", "input_ids", "labels", "dataset_name"}
    L_lo, L_hi = f.prompt_len_range
    for idx in range(len(f)):
        s = f[idx]
        assert set(s.keys()) == expected_keys, (idx, set(s.keys()))

        pv = s["pixel_values"]
        assert isinstance(pv, torch.Tensor)
        assert pv.shape == (f.vision_channels, f.vision_size, f.vision_size), pv.shape
        assert pv.dtype == torch.float32, pv.dtype

        ii, lb = s["input_ids"], s["labels"]
        assert isinstance(ii, torch.Tensor) and isinstance(lb, torch.Tensor)
        assert ii.dim() == 1 and ii.shape == lb.shape, (ii.shape, lb.shape)
        assert ii.dtype == torch.int64 and lb.dtype == torch.int64
        assert L_lo <= ii.shape[0] <= L_hi, (ii.shape, L_lo, L_hi)

        assert isinstance(s["dataset_name"], bytes)
        assert s["dataset_name"] == f.dataset_name
    print(f"[ut1_fixture_schema] OK  ({len(f)} samples)")


def ut2_fixture_label_mask():
    f = VLASchemaFixture(num_samples=32, seed=1)
    A = f.action_dim
    for idx in range(len(f)):
        s = f[idx]
        ii, lb = s["input_ids"], s["labels"]

        prefix, kept = lb[: -(A + 1)], lb[-(A + 1) :]
        assert (prefix == IGNORE_INDEX).all(), (idx, prefix.tolist())
        assert (kept != IGNORE_INDEX).all(), (idx, kept.tolist())
        # kept positions must equal the underlying input_ids (no shift, no extra masking)
        assert torch.equal(kept, ii[-(A + 1) :]), (idx, kept.tolist(), ii[-(A + 1) :].tolist())
    print(f"[ut2_fixture_label_mask] OK  ({len(f)} samples)")


def ut3_fixture_action_token_range():
    f = VLASchemaFixture(num_samples=32, seed=2)
    A = f.action_dim
    begin = f.action_token_begin_idx
    vocab = f.vocab_size
    for idx in range(len(f)):
        ii = f[idx]["input_ids"]

        assert int(ii[0]) == f.bos_token_id, (idx, int(ii[0]))
        assert int(ii[-1]) == f.eos_token_id, (idx, int(ii[-1]))

        action_slice = ii[-(A + 1) : -1]
        assert action_slice.shape == (A,), action_slice.shape
        assert (action_slice >= begin).all() and (action_slice <= vocab - 1).all(), (
            idx,
            action_slice.min().item(),
            action_slice.max().item(),
            begin,
            vocab - 1,
        )

        prompt_body = ii[1 : -(A + 1)]
        if prompt_body.numel():
            assert (prompt_body >= 3).all() and (prompt_body <= begin - 1).all(), (
                idx,
                prompt_body.min().item(),
                prompt_body.max().item(),
                begin - 1,
            )
    print(f"[ut3_fixture_action_token_range] OK  ({len(f)} samples)")


def ut4_collator():
    """PaddedCollatorForActionPrediction over fixture samples.

    Verifies batch shapes, pad invariants (input_ids==pad_token_id, labels==-100,
    attention_mask==False at padded positions), and dataset_names passthrough.
    """
    B = 4
    f = VLASchemaFixture(num_samples=B, seed=3)
    collator = PaddedCollatorForActionPrediction()  # OpenVLA-7B defaults
    out = collator([f[i] for i in range(B)])

    expected_keys = {"pixel_values", "input_ids", "attention_mask", "labels", "dataset_names"}
    assert set(out.keys()) == expected_keys, set(out.keys())

    L_max = max(f[i]["input_ids"].shape[0] for i in range(B))
    assert out["input_ids"].shape == (B, L_max), out["input_ids"].shape
    assert out["labels"].shape == (B, L_max), out["labels"].shape
    assert out["attention_mask"].shape == (B, L_max), out["attention_mask"].shape
    assert out["attention_mask"].dtype == torch.bool, out["attention_mask"].dtype
    assert out["pixel_values"].shape == (B, f.vision_channels, f.vision_size, f.vision_size), out["pixel_values"].shape

    # Pad invariants: where attention_mask is False -> input_ids == pad, labels == -100.
    pad_positions = ~out["attention_mask"]
    if pad_positions.any():
        assert (out["input_ids"][pad_positions] == collator.pad_token_id).all()
        assert (out["labels"][pad_positions] == IGNORE_INDEX).all()
    # Non-pad positions still satisfy labels==-100 OR labels==input_ids (prompt mask vs action tokens)
    real = out["attention_mask"]
    assert ((out["labels"][real] == IGNORE_INDEX) | (out["labels"][real] == out["input_ids"][real])).all()

    assert out["dataset_names"] == [f.dataset_name] * B, out["dataset_names"]
    print(f"[ut4_collator] OK  (B={B}, L_max={L_max})")


if __name__ == "__main__":
    ut1_fixture_schema()
    ut2_fixture_label_mask()
    ut3_fixture_action_token_range()
    ut4_collator()
