# HRM-Text-RLVR

RLVR (Reinforcement Learning with Verifiable Rewards) training for
[**DFM Mimir**](https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5)
([tech report](https://arxiv.org/abs/2608.13517)), an HRM-Text model with a
Gemma-style tokenizer and chat template, using GRPO ([TRL](https://github.com/huggingface/trl)'s
`GRPOTrainer`). Kicked off with GSM8K; the reward/dataset pieces are meant to be swapped out for
other verifiable tasks.

## Why this isn't just `trl.GRPOTrainer`

Mimir (`hrm_text` architecture, `HrmTextForCausalLM`, native in `transformers>=5.13`) is a
**PrefixLM**: the prompt is meant to be attended to bidirectionally
(`token_type_ids == 1`), and only the completion causally (`== 0`). Skipping this is not a minor
ablation — a 24-question ARC-Challenge probe measured greedy accuracy **0.71 with the
bidirectional prefix vs. 0.12 (below chance) fully causal**. Stock `GRPOTrainer` never sets
`token_type_ids` for a text-only model, so `train_grpo.py` subclasses it
(`PrefixLMGRPOTrainer`) with two small overrides that set it correctly for both rollout
generation and every scoring pass (policy / old-policy / reference model). See the module
docstring in `train_grpo.py` for the full explanation, and `--check-prefix-lm` below to verify it
empirically on your setup.

## Files

| File | What it is |
|---|---|
| `train_grpo.py` | The training script — the only thing you run. |
| `requirements.txt` | Dependencies; see [Setup](#setup). |
| `NOTES.md` | Scratch notes on Mimir (tokenizer, chat template, model key). |

## Setup

```bash
uv pip install -r requirements.txt          # or: pip install -r requirements.txt
```

Do **not** install vLLM — there is no `hrm_text` vLLM backend, so the script always runs with
`use_vllm=False`.

⚠️ If you end up on an older `trl` (0.16.x) some other way, importing `GRPOTrainer` against
`transformers>=5.13` crashes with `RuntimeError: Failed to import trl.trainer.grpo_trainer ...
No module named 'vllm'` — a version-compat bug where `is_vllm_available()` becomes
unconditionally truthy. `trl>=1.13` (pinned in `requirements.txt`) fixes it, and also gets you
`token_type_ids` support in generation for free.

## Usage

```bash
# Sanity check (~2 min on a laptop): asserts single-BOS prompts, correct token_type_ids,
# correct chat-template rendering, then runs 2 optimizer steps. No W&B, tiny everything.
python train_grpo.py --smoke-test

# Confirm the PrefixLM plumbing actually matters on your model/checkpoint: generates GSM8K
# answers with bidirectional vs. fully-causal prompt attention and compares accuracy.
python train_grpo.py --check-prefix-lm

# Full run (danish-foundation-models/DFM-Mimir-v1.5, GSM8K, single GPU, full fine-tune).
python train_grpo.py

# LoRA instead of full fine-tune.
python train_grpo.py --lora

# Answer directly instead of reasoning first (shorter completions, faster, weaker signal).
python train_grpo.py --no-reasoning

# Disable W&B / resume an interrupted run.
python train_grpo.py --report-to none
python train_grpo.py --resume-from-checkpoint auto

# Lower the rollout sampling temperature (default: 1.0).
python train_grpo.py --temperature 0.7
```

Run `python train_grpo.py --help` for the full flag list (batch size, generations per prompt,
learning rate, checkpoint frequency, W&B project/entity, etc.).

### Reward

Two reward functions, weighted `[1.0, 0.2]`:

- **`correctness_reward`** — 1.0 if the completion's final `Answer: <number>` line matches the
  GSM8K gold answer numerically, else 0.0. Anchors on the *last* such line, so it's robust to a
  preceding reasoning trace that happens to contain other numbers.
- **`format_reward`** — 1.0 if a parseable `Answer: <number>` line is present at all.

### Per-module learning-rate scaling

On by default (`--no-lr-module-scaling` disables it). Mimir's H- and L-modules reuse their
weights across cycles, so one parameter update changes the model's output through every
application of that module within a forward pass — with AdamW, a module applied (and
back-propagated through) more often would otherwise move the model function more per optimizer
step than one applied less often. `train_grpo.py` divides the base LR by the number of
gradient-carrying applications each module receives, auto-computed from the loaded checkpoint's
`H_cycles`/`L_cycles`/`L_bp_cycles` (`compute_module_lr_scales`; printed at startup as
`[lr-scaling] ...`, and everything but H/L trains at the base LR). For the default
`DFM-Mimir-v1.5` (`H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3]`), that's `H: lr/2`, `L: lr/6`; the
original `DFM-Mimir` (`L_bp_cycles=[0, 3]`) gives `H: lr/2`, `L: lr/3`. `--h-lr-scale`/
`--l-lr-scale` override the auto-computed divisors, for ablation or hand-tuning.

`--learning-rate` (the base `lr` above) defaults to `1e-5` — DFM-Mimir-v1.5's
end-of-pretraining LR *before* dividers, so with scaling on by default, GRPO fine-tuning
picks up each module at roughly the LR it was already being trained at (`H: 5e-6`, `L: ~1.7e-6`
for v1.5), rather than an arbitrary RL-fine-tuning default.

### Reasoning mode

On by default. Mimir's chat template has an `enable_thinking` toggle, but that alone does not
make the model reason — what actually elicits a chain of thought is *also* prefilling the
assistant turn with the thought-channel opener `<|channel>thought\n` so generation continues
inside the channel. `--no-reasoning` disables both and shortens `max_completion_length`
accordingly (256 vs. 1024 tokens by default).

### Tracking and checkpoints

- `--report-to` defaults to `wandb` for real runs (`none` under `--smoke-test` unless overridden).
  Reward metrics (`rewards/correctness_reward/mean`, etc.) and, via `--log-completions` (default
  on), a sample completions table are logged automatically — no extra plumbing needed beyond
  `wandb login`.
- `--save-steps` (default 100) + `--save-total-limit` (default 3) give periodic, disk-bounded
  checkpointing to `<output-dir>/checkpoint-<step>` — full fine-tune checkpoints are large
  (~GBs), so the limit matters. `--resume-from-checkpoint auto` resumes the latest one.
