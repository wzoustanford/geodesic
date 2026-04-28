# OpenVLA — model, training, and data pipeline investigation

This document maps the published OpenVLA reference implementation
(`openvla/openvla` on GitHub) to the `VLAAgent` design specified in §A.3 of the
deep-research playbook (`mds/openvla_pi07_geodesic.pdf`). It is intended as a
working reference while porting OpenVLA into Geodesic.

The walkthrough is organized in three parts:

1. **Model** — the HF `OpenVLAForActionPrediction` class and the multimodal
   `forward()` it inherits.
2. **Training** — the canonical `vla-scripts/finetune.py` loop (model load,
   LoRA, AdamW, CE loss, action-token accuracy diagnostic).
3. **Data processing** — `ActionTokenizer`, `RLDSBatchTransform`, `RLDSDataset`,
   and `PaddedCollatorForActionPrediction`, with a step-by-step trace of one
   sample through the pipeline.

All code references are to a local checkout at `../openvla/` relative to the
Geodesic repo root.

---

## Part 1 — Model

Three pieces map to A.3.

### 1.1 `OpenVLAForActionPrediction`

The HF model class. This is what
`AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")` instantiates
inside `VLAAgent.__init__`, and its `predict_action()` is the heart of
`VLAAgent.act()`.

`prismatic/extern/hf/modeling_prismatic.py:492-562`

```python
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
```

### 1.2 The multimodal `forward()`

This is what `update_actor()` calls — and it's where the `IGNORE_INDEX = -100`
mask gets injected for every vision patch position so CE only counts action
tokens.

`prismatic/extern/hf/modeling_prismatic.py:361-415` (multimodal branch)

```python
# === Handle Multimodal Forward ===
elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
    assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

    # Visual Feature Extraction
    patch_features = self.vision_backbone(pixel_values)

    # Projection Logic =>> Update Attention Mask
    projected_patch_embeddings = self.projector(patch_features)
    projected_patch_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    # Get Input Embeddings (from Language Model Embeddings)
    input_embeddings = self.get_input_embeddings()(input_ids)

    # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
    )
    multimodal_attention_mask = None
    if attention_mask is not None:
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
        )

    # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
    multimodal_labels = None
    if labels is not None:
        projected_patch_labels = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=IGNORE_INDEX,
            dtype=labels.dtype,
            device=labels.device,
        )
        multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

    # Dispatch to Language Model
    language_model_output = self.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=multimodal_labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
```

---

## Part 2 — Training (the canonical `update_actor` body)

The training loop in `vla-scripts/finetune.py` is the procedural blueprint the
report compresses into `VLAAgent.__init__` and `VLAAgent.update_actor`.

### 2.1 Model + tokenizer setup

`vla-scripts/finetune.py:151-192`

```python
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
```

### 2.2 The actual `update_actor` body

`vla-scripts/finetune.py:250-317`

```python
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            ...

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
```

### 2.3 Mapping back to the report's `VLAAgent`

| Report's `VLAAgent` element | OpenVLA source |
| --- | --- |
| `__init__`: load processor + model + LoRA wrap | `finetune.py:151–182` |
| `act()`: greedy generate + decode + unnormalize | `OpenVLAForActionPrediction.predict_action` |
| `update_actor()`: forward → `out.loss` → `.backward()` | `finetune.py:253–267` |
| `action_token_acc` diagnostic | `finetune.py:270–277` |
| `IGNORE_INDEX = -100` patch-position mask | `modeling_prismatic.py:393–401` (auto-injected by `forward()`) |
| Optimizer | `AdamW(trainable_params, lr=...)` at `finetune.py:189` |

One important correction the source makes vs the report: OpenVLA's reference
recipe uses **DDP, not FSDP** (`finetune.py:185`); FSDP is used in the
from-scratch `train.py` pretraining script, not in finetune. For LoRA + bf16 +
7B, DDP is what the published recipe runs.

---

## Part 3 — Data pipeline

Four pieces, in the order they fire per batch.

### 3.1 `ActionTokenizer`

Continuous 7-DoF action → 7 discrete token ids that hijack the tail of the
Llama-2 vocab (ids `vocab_size - n_bins` … `vocab_size - 1`).

`prismatic/vla/action_tokenizer.py` (full file, 73 lines)

```python
"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """Returns continuous actions for discrete action token IDs."""
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
```

Key contract: `action_token_begin_idx` is the threshold the finetune loop uses
to build the CE-loss mask
(`action_gt > action_tokenizer.action_token_begin_idx`).

### 3.2 `RLDSBatchTransform`

Single-sample converter. Takes one raw RLDS dict →
`{pixel_values, input_ids, labels}` ready for the model, including the `-100`
prompt mask. This *is* the report's `build_vla_example` in A.5.

`prismatic/vla/datasets/datasets.py:30-67`

```python
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
```

The single line `labels[: -(len(action) + 1)] = IGNORE_INDEX` is the `-100` mask
from report §A.4 — everything before the trailing `action_dim + 1` tokens (7
action ids + `</s>`) is masked out of the CE loss.

### 3.3 `RLDSDataset`

Thin `IterableDataset` that wraps the TFDS/RLDS pipeline
(`make_interleaved_dataset`) and yields `RLDSBatchTransform`-processed samples.
This is what A.5's `VLADataset` wraps in a Ray actor.

`prismatic/vla/datasets/datasets.py:70-154`

```python
class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

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
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
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
            )}),

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
```

Notable: `window_size=1, future_action_window_size=0` is the OpenVLA single-step
config — these are the two knobs you crank to get the 50-step action chunks π0
uses (Phase 3 of the upgrade ladder).

### 3.4 `PaddedCollatorForActionPrediction`

Batches a list of per-sample dicts into a right-padded tensor batch.
`input_ids` pads with `pad_token_id`; `labels` pads with `-100` (so padding
positions are also excluded from CE). `attention_mask` is derived from non-pad
positions.

`prismatic/util/data_utils.py:94-142`

```python
@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
```

### 3.5 Per-sample data flow — birds-eye view

```
RLDS TFDS shard
   │  (dict: observation.image_primary, action, task.language_instruction)
   ▼
RLDSDataset.__iter__            ← yields one rlds_batch at a time
   │
   ▼
RLDSBatchTransform.__call__     ← per sample:
   │    • Image.fromarray(obs["image_primary"][0])  → PIL
   │    • ActionTokenizer(action)                    → 7 token strings
   │    • PromptBuilder("openvla") adds human+gpt turns
   │    • base_tokenizer(prompt)                     → input_ids (list[int])
   │    • image_transform(img)                       → pixel_values (Tensor)
   │    • labels = input_ids.clone(); labels[:-(A+1)] = -100
   ▼  {pixel_values, input_ids, labels, dataset_name}
PaddedCollatorForActionPrediction.__call__   ← per batch:
   │    • pad_sequence(input_ids, pad_value=pad_token_id)
   │    • pad_sequence(labels,   pad_value=-100)
   │    • attention_mask = input_ids.ne(pad_token_id)
   │    • torch.stack(pixel_values)
   ▼  batch dict → vla(**batch).loss
```

That closes the loop with the training body in `finetune.py:255–261`:

```python
output = vla(
    input_ids=batch["input_ids"].to(device_id),
    attention_mask=batch["attention_mask"].to(device_id),
    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
    labels=batch["labels"],
)
loss = output.loss
```

The three `-100` injection points (prompt positions in `RLDSBatchTransform`,
padding positions in the collator, vision-patch positions in
`modeling_prismatic.forward`) are what make the CE mean only count the 7 action
tokens + `</s>` — nothing else.

---

## Part 4 — Step-by-step trace through one sample

Walking through one concrete example, OpenVLA-7B (`vocab_size=32064`,
`pad_to_multiple_of=64` → effective vocab 32000, `n_bins=256`, `action_dim=7`,
dual vision tower at 224²).

### 4.1 RLDSBatchTransform — per-sample

**Input.** One `rlds_batch` dict (all numpy, fresh out of TFDS), shapes for the
OpenVLA defaults (`window_size=1`, `future_action_window_size=0`):

```python
rlds_batch = {
    "dataset_name": b"bridge_orig",
    "action":             np.ndarray, shape [1, 7],   # window of 1, 7-DoF
    "observation": {
        "image_primary":  np.ndarray, shape [1, H, W, 3], uint8,
    },
    "task": {
        "language_instruction": b"PUT THE SPOON IN THE BOWL",
    },
}
```

#### Step 1 — unpack the window
```python
dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
img  = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
lang = rlds_batch["task"]["language_instruction"].decode().lower()
```
- `action` becomes `np.ndarray` shape `[7]` (the `[0]` peels off the window dim).
- `img` is a PIL.Image, raw camera resolution (the RLDS pipeline already resized to `resize_resolution=(224, 224)` upstream — see `RLDSDataset.frame_transform_kwargs`).
- `lang = "put the spoon in the bowl"`.

#### Step 2 — discretize + tokenize the action

`self.action_tokenizer(action)`:
```python
action = np.clip(action, -1, 1)            # action ∈ [-1, 1]^7
discretized = np.digitize(action, self.bins)   # ints in [1, 256], shape [7]
ids         = self.tokenizer.vocab_size - discretized   # e.g. [31866, 31874, …, 31744]
return self.tokenizer.decode(list(ids))    # → str of 7 chars/tokens
```

Concrete: action `0.02` → `np.digitize` returns ~`134` (bin 134 of 256 in
`[-1, 1]`) → token id `32000 - 134 = 31866`. Decode that single id back to its
string form (e.g. `"⟪31866⟫"` — a Unicode glyph that nobody else uses, which is
the whole point of grabbing the tail of the BPE vocab). `__call__` returns a
7-character string of these glyphs.

Why `vocab_size - bin`? The Llama-2 tokenizer's last 256 ids are essentially
never used by natural English, so they're free real estate for action codes.
`action_token_begin_idx = vocab_size - (n_bins + 1) = 31743` is the threshold
the loss mask later uses to identify "this position is an action token."

#### Step 3 — build the prompt with `PurePromptBuilder`

`prompt_builder = PurePromptBuilder("openvla")` initializes:
```python
self.bos, self.eos = "<s>", "</s>"
self.wrap_human = lambda msg: f"In: {msg}\nOut: "
self.wrap_gpt   = lambda msg: f"{msg}{self.eos}"
self.prompt, self.turn_count = "", 0
```

Two `add_turn` calls:
```python
add_turn("human", "What action should the robot take to put the spoon in the bowl?")
# turn_count=0, even → wrap_human
# self.prompt = "In: What action should the robot take to put the spoon in the bowl?\nOut: "

add_turn("gpt", "⟪31866⟫⟪31874⟫⟪31862⟫⟪31873⟫⟪31873⟫⟪31858⟫⟪31744⟫")
# turn_count=1, odd → wrap_gpt
# self.prompt += "⟪…7 action glyphs…⟫</s>"
```

`prompt_builder.get_prompt()` strips the leading `<s>` (because the tokenizer
auto-adds BOS) and returns:
```
In: What action should the robot take to put the spoon in the bowl?\nOut: ⟪…7 glyphs…⟫</s>
```

#### Step 4 — base_tokenizer
```python
input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
```

This produces a Python `list[int]` like (sketched, not exact):
```
[ 1,    512,   29901, 1724,  3158, …,  29871,  31866, 31874, 31862, 31873, 31873, 31858, 31744,  2 ]
  ↑BOS  "In"  ":"   "What"  "action"   "▁"     ↑ 7 action token ids                          ↑EOS
                                       ↑ token 29871 is the lone " " after "Out:"
                                         (predict_action injects this if missing — see modeling_prismatic:512)
```
Total length `L` ≈ 30–35 tokens (English varies, action portion is exactly 8: 7
actions + EOS).

#### Step 5 — labels start as a copy
```python
labels = list(input_ids)            # same Python list
input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   # both [L] int64
```

#### Step 6 — image transform via `PrismaticImageProcessor.apply_transform`

For OpenVLA-7B the processor is configured with **two** input sizes (one per
vision backbone — DINOv2 and SigLIP, both at 224²):
```python
self.input_sizes = [(3, 224, 224), (3, 224, 224)]
self.means       = [(0.485, 0.456, 0.406), (0.5, 0.5, 0.5)]   # ImageNet vs SigLIP
self.stds        = [(0.229, 0.224, 0.225), (0.5, 0.5, 0.5)]
```

`apply_transform(img)` runs the TIMM
`Compose([Resize, CenterCrop, ToTensor, Normalize])` **once per backbone**, then
channel-stacks:
```python
imgs_t = []
for idx in range(len(self.input_sizes)):       # idx ∈ {0, 1}
    img_idx   = TVF.resize(img, **self.tvf_resize_params[idx])      # e.g. → 256
    img_idx   = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])  # → 224×224
    img_idx_t = TVF.to_tensor(img_idx)                              # [3, 224, 224] float32 ∈ [0,1]
    img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
    imgs_t.append(img_idx_t)
img_t = torch.vstack(imgs_t)        # [6, 224, 224]  ← 3+3 channels, NOT a batch dim
```

`pixel_values` is the `[6, 224, 224]` tensor the model later splits back into
3+3 channels inside `vision_backbone.forward()`.

#### Step 7 — apply the `-100` mask

```python
labels[: -(len(action) + 1)] = IGNORE_INDEX     # mask everything except last 8 positions
if not self.predict_stop_token:
    labels[-1] = IGNORE_INDEX                   # default True → keep </s> in CE
```

Visualized (length `L=32` for the example; `A=7`, so we keep the last 8):
```
positions      0    1    2    …    L-9   L-8   L-7   L-6   L-5   L-4   L-3   L-2   L-1
input_ids      1    512  29901      29871 31866 31874 31862 31873 31873 31858 31744 2
labels         -100 -100 -100  …    -100  31866 31874 31862 31873 31873 31858 31744 2
                                          ╰────────── kept (8 positions) ──────────╯
```

#### Step 8 — return
```python
return dict(pixel_values=pixel_values,    # [6, 224, 224]  bf16 later
            input_ids=input_ids,          # [L]            int64
            labels=labels,                # [L]            int64, -100 masked
            dataset_name=dataset_name)    # bytes
```

Critical invariant established here: **the seven action positions and `</s>`
are the only places where `labels != -100`.** Every downstream loss reduction
depends on this.

### 4.2 PaddedCollatorForActionPrediction — per batch

**Input.** A `Sequence[Dict]` of length `B` (the dataloader's collected
mini-batch). Each dict is what `RLDSBatchTransform` returned. The samples have
**different `L`** because prompts differ in token count — this is exactly what
the collator handles.

#### Step 1 — gather lists
```python
input_ids, labels = tuple([instance[key] for instance in instances]
                          for key in ("input_ids", "labels"))
pixel_values     = [instance["pixel_values"] for instance in instances]
dataset_names    = [instance["dataset_name"] for instance in instances]  # if present
```
After this: `input_ids` is `list[Tensor[L_i]]` of length `B`, `labels` likewise,
`pixel_values` is `list[Tensor[6,224,224]]`.

#### Step 2 — assert right padding
```python
assert self.padding_side == "right"
```
Right-padding matters because the action tokens are at the **tail** of each
sample. If you padded on the left, every sample would still have its action
tokens at fixed offsets from the right edge — fine — but right-pad keeps the
original positions stable from index 0, which is what the rest of the pipeline
assumes (and what the diagnostic action-logit slice in `finetune.py:270`
assumes — it takes `logits[:, n_patches:-1]` from the left).

#### Step 3 — pad input_ids
```python
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
# shape: [B, L_max]
```
- `L_max = max(L_i for i in B)`.
- Pad positions get `pad_token_id` (Llama-2 OpenVLA: `32000`, the special pad token added at construction time).
- Right-padded: `[real_tokens..., 32000, 32000, ..., 32000]`.

#### Step 4 — pad labels with `-100`
```python
labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
# shape: [B, L_max]
```
**This is the second `-100` injection point.** The first masked the prompt
positions inside each sample; this masks the right-pad positions added when
shorter samples were extended to `L_max`. After this step:
- For each row, positions `0..L_i-9` are `-100` (prompt mask from RLDSBatchTransform).
- Positions `L_i-8..L_i-1` carry real action ids + `</s>`.
- Positions `L_i..L_max-1` are `-100` (collator pad mask).

So per row, exactly 8 (or 7 if `predict_stop_token=False`) positions contribute
to the CE mean — regardless of where they sit in the row.

#### Step 5 — truncate
```python
input_ids = input_ids[:, : self.model_max_length]
labels    = labels[:, : self.model_max_length]
```
For Llama-2-7B, `model_max_length=2048`. Action prompts are ~30 tokens, so this
almost never bites — but it's a safety net for malformed instructions.

#### Step 6 — attention_mask
```python
attention_mask = input_ids.ne(self.pad_token_id)
# shape: [B, L_max], dtype: bool
```
True at non-pad positions, False at pad positions. This mask is for the
**text-only** sequence; the vision-patch positions get added later inside
`PrismaticForConditionalGeneration.forward()`, which extends the mask by 256
`True`s (see `modeling_prismatic.py:371-377` in piece 2 above).

#### Step 7 — stack pixel_values
```python
assert all(pv is not None for pv in pixel_values)        # no unimodal data in VLA training
if isinstance(pixel_values[0], torch.Tensor):
    pixel_values = torch.stack(pixel_values)             # [B, 6, 224, 224]
elif isinstance(pixel_values[0], dict):
    pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
```
The `dict` branch handles vision backbones that return separate tensors per
view (uncommon for OpenVLA-7B; common for fused multi-view setups).

#### Step 8 — return
```python
return dict(pixel_values=pixel_values,        # [B, 6, 224, 224]
            input_ids=input_ids,              # [B, L_max]
            attention_mask=attention_mask,    # [B, L_max] bool
            labels=labels,                    # [B, L_max] int64, -100 masked
            dataset_names=dataset_names)      # list[bytes], optional
```

### 4.3 What the model sees, end-to-end

When `vla(**batch)` is called in `finetune.py:255-261`, this batch dict hits
`PrismaticForConditionalGeneration.forward()` (piece 2 in Part 1). That forward
does the **third** `-100` injection: 256 patch-position labels for the projected
SigLIP/DINOv2 tokens are inserted between BOS and the rest of the sequence:

```
final sequence (per row, conceptually):
  [ BOS ] [ 256 vision patch embs ] [ "In: …" prompt ] [ "Out: " glyph ] [ 7 action ids ] [ </s> ] [ pad pad … ]
labels:
  [ BOS ] [ 256 × -100            ] [        -100   ] [   -100         ] [ real ids     ] [   2  ] [ -100 -100 ]
```

Final loss after `cross_entropy(..., ignore_index=-100)` and mean reduction:
```
loss = (1/N) Σ_{positions where labels != -100} CE(logits, labels)
     = (1/N) Σ_{8 positions × B rows} ...   ← N = 8B (or 7B if predict_stop_token=False)
```

So the per-batch denominator is **constant at `8B`** regardless of `L_max` —
the three `-100` injection sites cooperate to make sure padding length doesn't
dilute the loss signal. That's why the report leans so hard on the
`IGNORE_INDEX = -100` convention being the critical correctness step.

### 4.4 Two gotchas worth flagging

1. **The diagnostic `action_token_acc` mask in `finetune.py:273` excludes `</s>`.**
   It uses `action_gt > action_tokenizer.action_token_begin_idx` (= `31743`).
   Action token ids are in `[31744, 31999]` and pass; `</s>` is `2` and fails.
   So the **CE loss** counts 8 positions per row but the **logged accuracy** is
   over 7. Easy to mis-read these as the same denominator.

2. **`pad_token_id` on OpenVLA is `32000`.** Llama-2 doesn't ship a pad token;
   the OpenVLA processor adds one at vocab index 32000, then
   `pad_to_multiple_of=64` rounds the embedding table out to 32064. The "256
   action tokens at the tail" actually live at `[31744, 31999]` — strictly
   *below* the pad token. Worth knowing if you ever build a custom collator:
   don't reuse Llama-2's default behaviour blindly.
