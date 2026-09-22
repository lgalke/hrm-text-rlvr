"""RLVR (GRPO) training for DFM Mimir (HRM-Text) on tasks with verifiable rewards.

Kicks off with GSM8K: a policy is rolled out on grade-school math questions and
rewarded 1.0 for a numerically-correct final answer (plus a small bonus for
emitting the answer in a parseable format), using TRL's ``GRPOTrainer``.

**Why this isn't just ``trl.GRPOTrainer`` out of the box.** DFM-Mimir
(``hrm_text``, see ``danish-foundation-models/DFM-Mimir`` / ``config.json``'s
``"prefix_lm": true``) is trained with the *prompt* attended to bidirectionally
and only the *completion* attended to causally -- ``HrmTextModel.forward`` turns
this on via ``token_type_ids`` (1 = bidirectional block, 0/absent = causal):

    is_first_iteration = past_key_values is None or not past_key_values.is_initialized
    if token_type_ids is not None and is_first_iteration:
        if self.config.prefix_lm:
            mask_kwargs["block_sequence_ids"] = torch.where(token_type_ids == 1, 0, -1)

(``transformers/models/hrm_text/modeling_hrm_text.py``). Skipping this is not a
minor ablation: a 24-question ARC-Challenge probe measured greedy accuracy
0.71 with the bidirectional prefix vs 0.12 (below chance) fully causal -- see
the sibling ``eval.py`` in ``hrm-interp``. Stock ``GRPOTrainer`` never sets
``token_type_ids`` for a text-only model (it's wired up for VLM image/text
segment ids only), so plain GRPO would silently roll out *and* score Mimir in
the wrong regime. ``PrefixLMGRPOTrainer`` below fixes exactly this, in two
small overrides, and nothing else about GRPO changes.

**Reasoning mode.** Mimir's chat template has an ``enable_thinking`` toggle,
but per ``eval.py``'s notes that alone does not make the model reason -- what
works is *also* prefilling the assistant turn with the thought-channel opener
``<|channel>thought\\n`` so generation continues inside the channel. That
prefill is model-side text, not part of the user's prompt, so it stays causal
(``token_type_id == 0``) even though it lives in the "prompt" half of the
rollout (before any tokens the policy itself samples).

**Tracking and checkpoints.** ``report_to`` defaults to Weights & Biases (run
``wandb login`` once beforehand); ``--report-to none`` disables it. TRL logs
its reward/completion metrics through the same ``Trainer.log()`` call
regardless of backend, so nothing else has to change to get
``rewards/correctness_reward/mean``, ``rewards/format_reward/mean``, etc. in
the dashboard -- ``--log-completions`` (on by default) additionally sends a
sample of (prompt, completion) pairs every ``--logging-steps`` steps, logged
as a W&B table, for eyeballing *what* is being rewarded. Checkpoints save
every ``--save-steps`` steps to ``<output-dir>/checkpoint-<step>``
(``--save-total-limit`` caps how many are kept); ``--resume-from-checkpoint``
resumes an interrupted run.

**Periodic validation eval.** Every ``--eval-steps`` steps (default on;
``--no-eval`` disables it), the trainer rolls out on a held-out slice of the
GSM8K *test* split (``--eval-split``/``--eval-samples``) and scores it with
the same reward functions, logged as ``eval_rewards/correctness_reward/mean``
etc. -- a genuine train/test split, since GSM8K's ``train`` and ``test`` are
disjoint. This is what actually answers "is the policy generalizing" as
opposed to just fitting the training rollouts' reward.

Usage
-----
    python train_grpo.py --smoke-test                  # ~2 min on a Mac, sanity only, no wandb
    python train_grpo.py --check-prefix-lm              # bidirectional vs causal prompt, quick GSM8K probe
    python train_grpo.py                                # full run, danish-foundation-models/DFM-Mimir, GSM8K
    python train_grpo.py --lora                          # LoRA instead of full fine-tune
    python train_grpo.py --no-reasoning                  # direct answers, short completions
    python train_grpo.py --report-to none                # disable W&B
    python train_grpo.py --resume-from-checkpoint auto    # resume the latest checkpoint in --output-dir
    python train_grpo.py --no-eval                        # disable periodic validation eval
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

DEFAULT_MODEL = "danish-foundation-models/DFM-Mimir"

# The thought-channel opener that actually elicits reasoning (see module docstring
# and eval.py's _THINK_PREFILL). A no-op string when --no-reasoning is passed.
THINK_PREFILL = "<|channel>thought\n"

INSTRUCTION = (
    "Solve the problem. Finish your response with a final line in exactly this "
    "form, with no other text after it:\nAnswer: <number>"
)


# --------------------------------------------------------------------------- #
# GSM8K -> GRPO conversational dataset.
# --------------------------------------------------------------------------- #
def _gold_answer(solution: str) -> str:
    """GSM8K's ``answer`` column is a worked solution ending in ``#### 72``."""
    raw = solution.rsplit("####", 1)[-1].strip()
    return raw.replace(",", "").replace("$", "")


def build_gsm8k_dataset(split: str = "train", max_samples: int | None = None) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_prompt(example: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}],
            "answer": _gold_answer(example["answer"]),
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def filter_by_prompt_length(ds: Dataset, tokenizer, max_tokens: int) -> Dataset:
    """Drop examples whose rendered prompt exceeds ``max_tokens``.

    TRL >=1.x dropped ``GRPOConfig.max_prompt_length`` (it no longer truncates
    prompts for you), so we filter instead of silently feeding an
    over-length prompt into generation.
    """

    def ok(example: dict) -> bool:
        ids = tokenizer.apply_chat_template(
            example["prompt"], tokenize=True, add_generation_prompt=True
        )
        return len(ids) <= max_tokens

    before = len(ds)
    ds = ds.filter(ok)
    dropped = before - len(ds)
    if dropped:
        print(f"[dataset] dropped {dropped}/{before} examples over {max_tokens} prompt tokens")
    return ds


# --------------------------------------------------------------------------- #
# Verifiable reward: numeric correctness + answer-line format.
# --------------------------------------------------------------------------- #
# Anchors on the LAST "Answer: X" line, exactly like eval.py's _split_reasoning:
# robust to a preceding thought-channel trace, which may itself contain numbers
# or the word "answer".
_ANSWER_LINE = re.compile(r"(?im)^[ \t]*answer[ \t]*:[ \t]*(-?\$?[\d,]*\.?\d+)")
_ANY_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_prediction(text: str) -> float | None:
    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        raw = matches[-1].group(1)
    else:
        # No "Answer:" line at all -- fall back to the last number in the
        # completion so a near-miss still gets partial signal from format_reward
        # even though correctness_reward will very likely score it 0.
        found = _ANY_NUMBER.findall(text)
        if not found:
            return None
        raw = found[-1]
    raw = raw.replace(",", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _completion_text(completion) -> str:
    # Conversational completions are [{"role": "assistant", "content": "..."}].
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = _extract_prediction(_completion_text(completion))
        try:
            gold_val = float(gold)
        except (TypeError, ValueError):
            gold_val = None
        correct = pred is not None and gold_val is not None and abs(pred - gold_val) < 1e-4
        rewards.append(1.0 if correct else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        rewards.append(1.0 if _ANSWER_LINE.search(text) else 0.0)
    return rewards


# --------------------------------------------------------------------------- #
# PrefixLM-aware GRPO trainer.
# --------------------------------------------------------------------------- #
class PrefixLMGRPOTrainer(GRPOTrainer):
    """``GRPOTrainer`` that rolls out and scores HRM-Text's PrefixLM correctly.

    Two overrides, both keyed off the same convention used by ``eval.py`` and
    the interp scripts: ``token_type_ids == 1`` over the rendered prompt,
    ``== 0`` over everything else (left-padding, the reasoning prefill, and the
    sampled completion). Every scoring pass (policy, old-policy for importance
    sampling, and the reference model when ``beta > 0``) funnels through
    ``_get_per_token_logps_and_entropies``, so overriding it here is sufficient
    to keep rollout and training distributions identical -- there's no separate
    code path to patch.

    ``prefill_ids`` is the tokenized reasoning prefill (``THINK_PREFILL``, or
    ``[]`` when reasoning is disabled).
    """

    def __init__(self, *args, prefill_ids: list[int] | None = None, **kwargs):
        # Stored before super().__init__() (which never calls _tokenize_prompts
        # itself -- that only happens at train/eval time) so both overrides can
        # rely on it as soon as rollouts start.
        self._prefill_ids = list(prefill_ids or [])
        self._prefill_len = len(self._prefill_ids)
        super().__init__(*args, **kwargs)

    def _tokenize_prompts(self, prompts: list):
        prompt_ids, images, multimodal_fields = super()._tokenize_prompts(prompts)
        n = self._prefill_len
        if n:
            prompt_ids = [list(ids) + self._prefill_ids for ids in prompt_ids]
        # PrefixLM convention: the user-rendered prompt is the bidirectional
        # block; the model-side thought prefill stays causal (matches
        # eval.py's HRMModelAPI.complete(), which zeroes token_type_ids over
        # the same prefill for the same reason).
        multimodal_fields = dict(multimodal_fields)
        multimodal_fields["token_type_ids"] = [
            [1] * (len(ids) - n) + [0] * n for ids in prompt_ids
        ]
        return prompt_ids, images, multimodal_fields

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, *args, token_type_ids=None, **kwargs
    ):
        if token_type_ids is None:
            # Stock TRL only ever populates token_type_ids for VLM batches
            # (image/text segment ids); for a text-only model it's always None
            # here, so every scoring call -- policy, old-policy, reference --
            # needs it rebuilt from the attention mask.
            token_type_ids = attention_mask.clone()  # 1 on every real (non-pad) token
            token_type_ids[:, -logits_to_keep:] = 0  # completion: causal
            if self._prefill_len:
                start = -logits_to_keep - self._prefill_len
                token_type_ids[:, start:-logits_to_keep] = 0  # prefill: causal
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, *args,
            token_type_ids=token_type_ids, **kwargs,
        )


# --------------------------------------------------------------------------- #
# Shared prompt-rendering helper (used by the self-check and --check-prefix-lm;
# mirrors what PrefixLMGRPOTrainer does internally, but standalone for a plain
# model.generate() call).
# --------------------------------------------------------------------------- #
def render_and_encode(tokenizer, question: str, *, reasoning: bool, device: str):
    messages = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=reasoning
    )
    prefill = THINK_PREFILL if reasoning else ""
    # add_special_tokens=False: the chat template already renders <bos> as text
    # (see module docstring / prompt_templates.md); re-tokenizing with the
    # default would double it and misalign every position.
    enc = tokenizer(prompt + prefill, add_special_tokens=False, return_tensors="pt").to(device)
    token_type_ids = torch.ones_like(enc["input_ids"])
    if prefill:
        n_prefill = len(tokenizer(prefill, add_special_tokens=False)["input_ids"])
        token_type_ids[:, -n_prefill:] = 0
    enc["token_type_ids"] = token_type_ids
    return enc


# --------------------------------------------------------------------------- #
# --check-prefix-lm: quick echo of eval.py's bidirectional-vs-causal ARC probe,
# on GSM8K, so a broken token_type_ids wire-up is caught before a real run.
# --------------------------------------------------------------------------- #
def check_prefix_lm(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attn_implementation
    ).eval()
    device = _resolve_device(args.device)
    model = model.to(device)

    ds = build_gsm8k_dataset(split="test", max_samples=args.check_samples)
    print(f"[check-prefix-lm] {len(ds)} GSM8K test examples, reasoning={args.reasoning}")

    for label, bidirectional in (("bidirectional (as trained)", True), ("causal (ablation)", False)):
        correct = 0
        for example in ds:
            question = example["prompt"][0]["content"].split("\n\n" + INSTRUCTION)[0]
            enc = render_and_encode(tokenizer, question, reasoning=args.reasoning, device=device)
            if not bidirectional:
                enc["token_type_ids"] = torch.zeros_like(enc["token_type_ids"])
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.check_max_new_tokens, do_sample=False
                )
            text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = _extract_prediction(text)
            gold = float(example["answer"])
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1
        acc = correct / len(ds)
        print(f"[check-prefix-lm] {label}: {correct}/{len(ds)} = {acc:.3f}")


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# --smoke-test: build one real batch and assert the plumbing is correct before
# spending any compute on optimizer steps.
# --------------------------------------------------------------------------- #
def run_self_check(trainer: PrefixLMGRPOTrainer, tokenizer, reasoning: bool) -> None:
    prompts = [
        {"prompt": [{"role": "user", "content": f"What is {i} + {i}?\n\n{INSTRUCTION}"}], "answer": str(2 * i)}
        for i in range(1, 4)
    ]
    prompt_ids, _, multimodal_fields = trainer._tokenize_prompts([p["prompt"] for p in prompts])
    token_type_ids = multimodal_fields["token_type_ids"]

    bos_id = tokenizer.bos_token_id
    for ids, tt in zip(prompt_ids, token_type_ids):
        assert ids.count(bos_id) == 1, f"expected exactly one BOS, got {ids.count(bos_id)}: {ids[:10]}"
        assert len(tt) == len(ids), "token_type_ids must be one entry per prompt token"
        n_prefill = trainer._prefill_len
        if n_prefill:
            assert tt[-n_prefill:] == [0] * n_prefill, "reasoning prefill must be marked causal (0)"
        assert tt[: len(tt) - n_prefill] == [1] * (len(tt) - n_prefill), (
            "rendered prompt must be marked bidirectional (1)"
        )
        text = tokenizer.decode(ids)
        assert "<|turn>model" in text, f"prompt doesn't end in the model turn: {text!r}"
        if reasoning:
            assert text.endswith(THINK_PREFILL), f"reasoning prefill missing from rendered prompt: {text!r}"

    print(f"[self-check] {len(prompts)}/{len(prompts)} prompts: single BOS, token_type_ids OK, template OK")


# --------------------------------------------------------------------------- #
# CLI / main.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local dir (default: %(default)s).")
    parser.add_argument("--output-dir", default="./outputs/grpo-gsm8k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training examples (debugging).")
    parser.add_argument("--max-prompt-tokens", type=int, default=512, help="Filter out longer prompts.")

    reasoning = parser.add_mutually_exclusive_group()
    reasoning.add_argument(
        "--reasoning", dest="reasoning", action="store_true", default=True,
        help="Thought-channel prefill before answering (default).",
    )
    reasoning.add_argument(
        "--no-reasoning", dest="reasoning", action="store_false",
        help="Answer directly, no thought-channel prefill.",
    )

    parser.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)

    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 skips the reference model.")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-completion-length", type=int, default=None,
                         help="Default: 1024 with reasoning, 256 without.")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100,
                         help="Save a checkpoint every N optimizer steps (default: %(default)s).")
    parser.add_argument("--save-total-limit", type=int, default=3,
                         help="Keep at most N checkpoints in --output-dir (default: %(default)s); older ones are "
                              "deleted. Full fine-tune checkpoints are large (~GBs), so this matters.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                         help="'auto' resumes the latest checkpoint in --output-dir; or pass an explicit "
                              "checkpoint path.")
    parser.add_argument("--seed", type=int, default=42)

    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval", dest="do_periodic_eval", action="store_true", default=True,
        help="Periodic validation eval on a held-out GSM8K split, every --eval-steps (default).",
    )
    eval_group.add_argument(
        "--no-eval", dest="do_periodic_eval", action="store_false",
        help="Disable periodic validation eval.",
    )
    parser.add_argument("--eval-split", default="test", help="GSM8K split for validation (default: %(default)s).")
    parser.add_argument("--eval-samples", type=int, default=200,
                         help="Held-out examples per eval pass (default: %(default)s). Each one gets a full "
                              "rollout, so this trades eval cost for a tighter estimate.")
    parser.add_argument("--eval-steps", type=int, default=100,
                         help="Run validation eval every N optimizer steps (default: %(default)s).")
    parser.add_argument("--eval-num-generations", type=int, default=None,
                         help="Completions per eval prompt; default: same as --num-generations.")

    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa",
                         help="'sdpa' (default) or 'flex_attention'. FlashAttention raises with prefix_lm=True.")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--report-to", default=None,
                         help="'wandb' (default for real runs; none by default for --smoke-test), 'none', "
                              "'tensorboard', or a comma-separated combination.")
    parser.add_argument("--wandb-project", default="dfm-mimir-grpo")
    parser.add_argument("--wandb-entity", default=None, help="W&B team/user; default is your wandb-configured default.")
    parser.add_argument("--wandb-run-name", default=None, help="Default: auto-generated from model/reasoning/lora.")
    parser.add_argument("--wandb-log-model", choices=["false", "checkpoint", "end"], default="false",
                         help="Also upload model checkpoints to W&B artifacts. 'checkpoint' uploads every "
                              "--save-steps, 'end' uploads once at the end. Off by default: these checkpoints "
                              "are large.")
    log_completions = parser.add_mutually_exclusive_group()
    log_completions.add_argument(
        "--log-completions", dest="log_completions", action="store_true", default=True,
        help="Log a sample of (prompt, completion) pairs every --logging-steps to the tracking backend (default).",
    )
    log_completions.add_argument(
        "--no-log-completions", dest="log_completions", action="store_false",
        help="Don't log sample completions (only scalar metrics).",
    )

    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny run (~2 min) that self-checks the PrefixLM plumbing, for a local sanity pass.")
    parser.add_argument("--check-prefix-lm", action="store_true",
                         help="Compare bidirectional vs. causal prompt attention on a small GSM8K sample, then exit.")
    parser.add_argument("--check-samples", type=int, default=16)
    parser.add_argument("--check-max-new-tokens", type=int, default=512)

    args = parser.parse_args()

    if args.max_completion_length is None:
        args.max_completion_length = 1024 if args.reasoning else 256

    if args.smoke_test:
        args.max_samples = args.max_samples or 8
        args.num_generations = 2
        args.per_device_train_batch_size = 2
        args.gradient_accumulation_steps = 1
        args.max_completion_length = min(args.max_completion_length, 64)
        args.max_steps = 2
        args.logging_steps = 1
        args.beta = 0.0
        args.device = "auto"
        # Shrink rather than disable periodic eval, so --smoke-test also exercises
        # that code path: eval_steps=1 makes it actually fire within 2 train steps.
        args.eval_samples = min(args.eval_samples, 4)
        args.eval_steps = 1

    # --report-to left unset: default to W&B for a real run, but don't spam a
    # wandb project (or require being logged in) for a pure sanity check.
    if args.report_to is None:
        args.report_to = "none" if args.smoke_test else "wandb"

    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.check_prefix_lm:
        check_prefix_lm(args)
        return

    device = _resolve_device(args.device)
    print(f"[setup] model={args.model} device={device} dtype={args.dtype} reasoning={args.reasoning}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", truncation_side="left")

    prefill_ids = (
        tokenizer(THINK_PREFILL, add_special_tokens=False)["input_ids"] if args.reasoning else []
    )

    model_init_kwargs = dict(dtype=args.dtype, attn_implementation=args.attn_implementation)

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # gate_proj appears in both HrmTextAttention (an output gate) and
            # HrmTextMLP -- targeting both is intended.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    train_dataset = build_gsm8k_dataset(split=args.dataset_split, max_samples=args.max_samples)
    train_dataset = filter_by_prompt_length(train_dataset, tokenizer, args.max_prompt_tokens)
    print(f"[dataset] {len(train_dataset)} GSM8K ({args.dataset_split}) examples")

    eval_dataset = None
    eval_num_generations = args.eval_num_generations or args.num_generations
    if args.do_periodic_eval:
        eval_dataset = build_gsm8k_dataset(split=args.eval_split, max_samples=args.eval_samples)
        eval_dataset = filter_by_prompt_length(eval_dataset, tokenizer, args.max_prompt_tokens)
        print(f"[dataset] {len(eval_dataset)} GSM8K ({args.eval_split}) held-out eval examples, "
              f"every {args.eval_steps} steps")

    report_to = [] if args.report_to.lower() == "none" else [s.strip() for s in args.report_to.split(",")]

    run_name = args.wandb_run_name
    if run_name is None:
        tag = "lora" if args.lora else "full"
        reasoning_tag = "reasoning" if args.reasoning else "direct"
        run_name = f"grpo-gsm8k-{args.model.split('/')[-1]}-{tag}-{reasoning_tag}"

    if "wandb" in report_to:
        # WandbCallback.setup() reads these env vars at wandb.init() time; must
        # be set before trainer.train() (the first Trainer.log() call) fires.
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_LOG_MODEL", args.wandb_log_model)
        print(f"[wandb] project={args.wandb_project} run={run_name} log_model={args.wandb_log_model}")

    config = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs=model_init_kwargs,
        chat_template_kwargs={"enable_thinking": args.reasoning},
        beta=args.beta,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if args.do_periodic_eval else "no",
        eval_steps=args.eval_steps if args.do_periodic_eval else None,
        # One prompt group per device batch, so the (batch % num_generations == 0)
        # constraint holds trivially regardless of --eval-num-generations.
        per_device_eval_batch_size=eval_num_generations,
        num_generations_eval=args.eval_num_generations,
        seed=args.seed,
        bf16=(args.dtype == "bfloat16" and device == "cuda"),
        gradient_checkpointing=(device == "cuda"),
        use_vllm=False,  # no hrm_text vLLM backend
        reward_weights=[1.0, 0.2],  # correctness, format
        report_to=report_to,
        run_name=run_name,
        log_completions=args.log_completions,
        num_completions_to_print=10,
        remove_unused_columns=False,
    )

    trainer = PrefixLMGRPOTrainer(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        prefill_ids=prefill_ids,
    )

    if args.smoke_test:
        run_self_check(trainer, tokenizer, args.reasoning)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
    print(f"[checkpoints] every {args.save_steps} steps -> {args.output_dir}/checkpoint-<step> "
          f"(keeping last {args.save_total_limit})")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
