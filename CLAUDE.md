# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RLVR (GRPO) training for `danish-foundation-models/DFM-Mimir-v1.5` (default checkpoint), an HRM-Text model
(`hrm_text` architecture, `HrmTextForCausalLM`, native in `transformers>=5.13`, not
`trust_remote_code`). `train_grpo.py` is the whole deliverable; it has no test suite, linter, or
build step — it's a single-file research script. See `README.md` for setup and usage; this file
only covers what isn't there.

## Commands

```bash
python train_grpo.py --smoke-test                 # ~2 min sanity check, no GPU required, no W&B
python train_grpo.py --check-prefix-lm             # verifies the PrefixLM fix actually matters
python train_grpo.py --help                        # full flag list
python -m py_compile train_grpo.py                 # only "build" step there is
```

There are no automated tests. Correctness for the PrefixLM plumbing is checked by
`run_self_check()` (invoked automatically under `--smoke-test`), which asserts on a real batch
built via `PrefixLMGRPOTrainer._tokenize_prompts` — read that function before changing prompt
rendering, since it's the thing the assertions pin down. `run_lr_scaling_check()` (same gate)
builds the real optimizer and asserts the H-/L-module param groups got the expected scaled LR.

## Architecture

**The one thing this codebase exists to get right:** Mimir is trained as a PrefixLM — the
rendered prompt must be attended to bidirectionally (`token_type_ids == 1`), only the sampled
completion causally (`== 0`). `transformers/models/hrm_text/modeling_hrm_text.py` turns
`token_type_ids` into an attention-mask overlay only on the first (prefill) forward pass; stock
`trl.GRPOTrainer` never populates `token_type_ids` for a text-only model (it's wired up for VLM
image/text segment ids only). `PrefixLMGRPOTrainer` in `train_grpo.py` fixes this with exactly
two overrides:

- `_tokenize_prompts` — sets `token_type_ids` for the rollout `generate()` call, and appends the
  reasoning thought-prefill tokens (kept causal, `== 0`) after the rendered prompt.
- `_get_per_token_logps_and_entropies` — rebuilds `token_type_ids` from `attention_mask` for
  *every* scoring pass (policy, old-policy for importance sampling, and the reference model when
  `beta > 0`), since this is the single method all three funnel through.

Both overrides must agree on the convention (prompt = 1, everything else = 0) or rollout and
training distributions diverge silently — there's no runtime error, just a model that trains
wrong the way the docstring's ARC probe (0.71 → 0.12) describes. When touching either override,
re-run `--smoke-test` and read its three self-check assertions (single BOS, `token_type_ids`
shape/values, chat-template rendering) before trusting the diff.

**Per-module LR scaling.** `compute_module_lr_scales` derives the H-/L-module LR divisors from
the loaded checkpoint's `H_cycles`/`L_cycles`/`L_bp_cycles` config, replicating
`HrmTextModel`'s own gradient-truncation logic exactly (H always gets gradient on every
application; L is truncated per `L_bp_cycles` — see the function's docstring for the derivation
and the source lines it tracks). `PrefixLMGRPOTrainer.create_optimizer` applies the result as
per-parameter-group `"lr"` overrides, matched by **substring** (`.H_module.`/`.L_module.`, not
`.startswith()`) — under `--lora`, PEFT renames params to
`base_model.model.model.L_module.layers.N....lora_A.default.weight`, and only the substring
still matches; a prefix check would silently train everything at one LR. It raises instead of
silently no-op-ing if H or L ends up empty during a full fine-tune (an empty "other" bucket
under `--lora` is expected and fine). If `modeling_hrm_text.py`'s gradient-truncation logic ever
changes, `compute_module_lr_scales` needs to be updated to match it — it isn't derived
automatically from the model code.

**Reward functions** (`correctness_reward`, `format_reward`) anchor on the *last* `Answer:
<number>` line in the completion — this is what makes them robust to a reasoning trace that
contains other numbers. If you add a new task beyond GSM8K, follow the same "last matching line"
pattern rather than searching the whole completion.

**Environment trap:** the installed `trl` must be ≥1.13 against `transformers>=5.13`, or
`from trl import GRPOTrainer` crashes at import time (a version-compat bug, not a missing
dependency — see README's Setup section). If you're debugging an import error here, check the
`trl` version before anything else.
