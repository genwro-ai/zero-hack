"""GRPO (Group Relative Policy Optimization) finetuning for next-step models.

This implements RLVR-style finetuning of a *pretrained* autoregressive step
model for the sequence-completion objective. The reward is a validity *gate*
(``validate_sequence(prefix + completion)`` reporting no process-rule violations)
times a continuous quality score measuring fidelity to the prompt's true
continuation — block/token/exact accuracy, length sanity, and block diversity (the
same Task-2 metrics we report). Validity alone saturates once the base model is
mostly rule-compliant, giving GRPO no within-group advantage; the fidelity terms
restore that signal and prevent the "shortest valid tail" reward hack. See
:class:`RewardConfig`.

The policy is the autoregressive step model. Two checkpoint formats are
supported and auto-detected:

* **LSTM** (``save_lstm_checkpoint``): keys ``token_to_id`` / ``id_to_token`` /
  ``lstm_config``. Loaded via :func:`load_lstm_checkpoint`; the trainable module
  is ``inference.model``.
* **Transformer decoder** (``decoder_training._save_best``): key ``architecture``
  plus nested ``vocabulary`` and ``model_config``. Rebuilt as a
  :class:`TransformerModel`.

Both are wrapped behind :class:`StepPolicy`, a small common interface reusing the
existing encoding / sampling-mask logic so we never reinvent the wheel.

GRPO update (per prompt = a prefix):

#. Sample ``G`` completions (a *group*), getting per-step log-probs.
#. Compute rewards ``r_i`` (see :class:`RewardConfig`).
#. Group-relative advantages ``A_i = (r_i - mean(r)) / (std(r) + eps)``.
#. Loss ``= -mean_i(A_i * mean_t logprob_t(i)) + kl_coef * KL_k3(policy||reference)``
   where the reference is a frozen copy of the initial policy. Log-probs are
   length-averaged (not summed) to avoid the GRPO long-rollout bias, and the KL
   uses Schulman's positive k3 estimator.

Sampling does **not** apply :class:`ViolationMask` by default (the policy must be
able to make mistakes for the reward to be informative); pass
``mask_sampling=True`` to enable it.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, SequenceRecord, Vocabulary
from zero_hack.eval.blocks import block_runs
from zero_hack.eval.completion import _block_accuracy, _token_accuracy
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.lstm.inference import load_lstm_checkpoint, save_lstm_checkpoint
from zero_hack.models.lstm.model import LSTMConfig, LSTMModel
from zero_hack.models.transformer.model import TransformerConfig, TransformerModel
from zero_hack.models.violation_mask import ViolationMask

SEQUENCE_TERMINATOR = "SHIP LOT"


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class StepPolicy:
    """Common interface over the LSTM / Transformer autoregressive step models.

    The wrapped ``nn.Module`` (:attr:`model`) maps
    ``forward(input_ids[B, T], attention_mask[B, T]) -> logits[B, V]`` for the
    *last* position. This class adds prefix encoding, masked sampling, and the
    log-prob recomputation needed for the policy-gradient term.
    """

    def __init__(
        self,
        model: nn.Module,
        vocabulary: Vocabulary,
        *,
        kind: str,
        max_context: int = 192,
        device: torch.device | str = "cpu",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind  # "lstm" | "transformer"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.vocabulary = vocabulary
        self.max_context = max_context
        self.meta = meta or {}

        # Tokens that must never be *sampled* (special + family markers): the
        # policy only ever emits real process steps, exactly like the existing
        # inference wrappers.
        blocked = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
        self._blocked_ids = sorted(
            vocabulary.token_to_id[tok] for tok in blocked if tok in vocabulary.token_to_id
        )
        self._violation_mask = ViolationMask(vocabulary.id_to_token)
        self._terminator_id = vocabulary.token_to_id.get(SEQUENCE_TERMINATOR)

    # -- encoding -----------------------------------------------------------

    def _family_token(self, family: str) -> str:
        return FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])

    def encode_prefix(
        self, family: str, prefix: list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids[1, T], attention_mask[1, T])`` for ``<BOS> fam prefix``."""
        tokens = ["<BOS>", self._family_token(family), *prefix][-self.max_context :]
        ids = self.vocabulary.encode(tokens)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
        return input_ids, attention_mask

    def next_logits(
        self,
        family: str,
        prefix: list[str] | tuple[str, ...],
        *,
        mask_sampling: bool = False,
    ) -> torch.Tensor:
        """Next-step logits ``[V]`` with special/family tokens masked out.

        When ``mask_sampling`` is set, also apply :class:`ViolationMask` so the
        policy can only sample rule-compliant continuations.
        """
        input_ids, attention_mask = self.encode_prefix(family, prefix)
        logits = self.model(input_ids, attention_mask)[0]
        if self._blocked_ids:
            logits = logits.index_fill(
                0, torch.tensor(self._blocked_ids, device=logits.device), float("-inf")
            )
        if mask_sampling:
            logits = self._violation_mask(logits, list(prefix))
        return logits

    # -- sampling -----------------------------------------------------------

    def sample_completion(
        self,
        family: str,
        prefix: list[str] | tuple[str, ...],
        *,
        max_steps: int = 400,
        temperature: float = 1.0,
        mask_sampling: bool = False,
    ) -> tuple[list[str], torch.Tensor, bool]:
        """Roll out a completion under the current params.

        Returns ``(steps, per_step_logprobs[T], terminated)`` where ``steps`` are
        the produced steps (excluding the prefix), ``per_step_logprobs`` carries
        gradients, and ``terminated`` is True when ``SHIP LOT`` was emitted.
        """
        steps: list[str] = []
        logprobs: list[torch.Tensor] = []
        seq: list[str] = list(prefix)
        terminated = False
        temperature = max(1e-6, temperature)

        while len(steps) < max_steps:
            logits = self.next_logits(family, seq, mask_sampling=mask_sampling)
            if bool(torch.isfinite(logits).any().logical_not()):
                # Every candidate masked out (-inf): nothing legal to sample.
                break
            log_probs = F.log_softmax(logits / temperature, dim=-1)
            probs = log_probs.exp()
            sampled_id = int(torch.multinomial(probs, num_samples=1).item())
            logprobs.append(log_probs[sampled_id])
            token = self.vocabulary.id_to_token[sampled_id]
            steps.append(token)
            seq.append(token)
            if token == SEQUENCE_TERMINATOR:
                terminated = True
                break

        stacked = torch.stack(logprobs) if logprobs else torch.zeros(0, device=self.device)
        return steps, stacked, terminated

    def completion_logprobs(
        self,
        family: str,
        prefix: list[str] | tuple[str, ...],
        completion: list[str] | tuple[str, ...],
        *,
        mask_sampling: bool = False,
    ) -> torch.Tensor:
        """Per-step log-probs ``[T]`` of ``completion`` under the current params.

        Used to recompute the policy-gradient term (and the reference log-probs
        for the KL estimate) on a fixed completion. Carries gradients.
        """
        logprobs: list[torch.Tensor] = []
        seq: list[str] = list(prefix)
        for token in completion:
            logits = self.next_logits(family, seq, mask_sampling=mask_sampling)
            log_probs = F.log_softmax(logits, dim=-1)
            token_id = self.vocabulary.token_to_id.get(token, self.vocabulary.unk_id)
            logprobs.append(log_probs[token_id])
            seq.append(token)
        if logprobs:
            return torch.stack(logprobs)
        return torch.zeros(0, device=self.device)


# ---------------------------------------------------------------------------
# Checkpoint load / save (auto-detect format)
# ---------------------------------------------------------------------------


def detect_checkpoint_kind(payload: dict[str, Any]) -> str:
    """Return ``"lstm"``, ``"transformer"`` or ``"gpt"`` for a loaded checkpoint dict."""
    if "lstm_config" in payload and "token_to_id" in payload:
        return "lstm"
    if "architecture" in payload and "vocabulary" in payload:
        return "transformer"
    # GPTNextStepModel checkpoints (scripts/run_holdout_experiments._save_gpt_checkpoint):
    # model_config + model_state + vocabulary, but no `architecture` tag. Checked
    # after the transformer branch so the architecture-tagged format wins.
    if "model_config" in payload and "model_state" in payload and "vocabulary" in payload:
        return "gpt"
    raise ValueError(
        "Unrecognized checkpoint format: expected LSTM keys (lstm_config, "
        "token_to_id), transformer-decoder keys (architecture, vocabulary), or "
        "GPT keys (model_config, model_state, vocabulary). "
        f"Got keys: {sorted(payload)}"
    )


def load_policy(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> StepPolicy:
    """Load a checkpoint into a :class:`StepPolicy`, auto-detecting the format."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = detect_checkpoint_kind(payload)

    if kind == "lstm":
        inference = load_lstm_checkpoint(checkpoint, device=device)
        return StepPolicy(
            inference.model,
            inference.vocabulary,
            kind="lstm",
            max_context=inference.max_context,
            device=device,
            meta=inference.meta,
        )

    vocab = payload["vocabulary"]
    vocabulary = Vocabulary(
        token_to_id=dict(vocab["token_to_id"]),
        id_to_token=tuple(vocab["id_to_token"]),
    )

    if kind == "gpt":
        config = GPTConfig(**payload["model_config"])
        model = GPTNextStepModel(
            vocab_size=len(vocabulary.id_to_token),
            config=config,
            pad_id=vocabulary.pad_id,
        )
        model.load_state_dict(payload["model_state"])
        return StepPolicy(
            model,
            vocabulary,
            kind="gpt",
            max_context=config.max_context,
            device=device,
            meta={"source_payload_keys": sorted(payload)},
        )

    architecture = payload["architecture"]
    if architecture != "transformer":
        raise ValueError(
            f"Unsupported decoder architecture {architecture!r}; only 'transformer' "
            "checkpoints are supported as a GRPO policy."
        )
    config = TransformerConfig(**payload["model_config"])
    model = TransformerModel(
        vocab_size=len(vocabulary.id_to_token),
        config=config,
        pad_id=vocabulary.pad_id,
    )
    model.load_state_dict(payload["model_state"])
    return StepPolicy(
        model,
        vocabulary,
        kind="transformer",
        max_context=config.max_context,
        device=device,
        meta={"source_payload_keys": sorted(payload)},
    )


def save_policy(
    policy: StepPolicy,
    out: str | Path,
    *,
    source_checkpoint: str | Path,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save the (finetuned) policy back in the *same* format as ``source_checkpoint``.

    This keeps existing eval scripts and ``ViolationMask`` working unchanged.
    """
    out = Path(out)
    source_payload = torch.load(Path(source_checkpoint), map_location="cpu", weights_only=False)
    kind = detect_checkpoint_kind(source_payload)

    if kind == "lstm":
        meta = dict(source_payload.get("meta", {}))
        meta.update(extra_meta or {})
        return save_lstm_checkpoint(
            out,
            policy.model,  # type: ignore[arg-type]
            policy.vocabulary,
            max_context=policy.max_context,
            meta=meta,
        )

    # Transformer / GPT: preserve the source payload schema exactly (model_config,
    # vocabulary, train_families/train_size, ...), swapping only model_state for the
    # finetuned weights, so existing eval/inference loaders keep working unchanged.
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(source_payload)
    payload["model_state"] = policy.model.state_dict()
    grpo_meta = dict(payload.get("grpo", {}))
    grpo_meta.update(extra_meta or {})
    payload["grpo"] = grpo_meta
    torch.save(payload, out)
    return out


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Gated composite reward: a validity gate times a continuous quality score.

    Process validity is a hard requirement, so it enters *multiplicatively* — an
    invalid rollout collapses to (near) zero regardless of how well it matches the
    target. Among the (usually many) valid rollouts of a prompt, the continuous
    fidelity terms spread the scores out, which is what gives GRPO a non-degenerate
    within-group advantage (``A_i = (r_i - mean) / std``): a binary validity reward
    saturates and produces no gradient once the base model is already mostly valid.

    Quality is measured against the prompt's *true continuation* (the gold suffix of
    the training record the prefix was cut from), reusing the exact Task-2 metrics
    from :mod:`zero_hack.eval.completion`, so GRPO optimizes the metric we report:

    * ``w_block``  — block-shape accuracy (LCS over collapsed block runs).
    * ``w_token``  — positional token accuracy.
    * ``w_exact``  — bonus for an exact reproduction of the gold suffix.
    * ``w_length`` — length sanity ``exp(-|len-gold|/len_gold)``; kills the
      "ship immediately" hack (a minimal valid tail is short but scores ~0 here).
    * ``w_diversity`` — fraction of distinct block runs; penalizes repetition loops.

    ``graded_validity`` softens the gate to ``max(0, 1 - n_violations/len)`` so a
    rollout with one violation still ranks above one with three (more informative
    advantages than a hard 0/1). Termination is rewarded and unterminated rollouts
    (hit ``max_steps`` without ``SHIP LOT``) are penalized.
    """

    w_block: float = 0.6
    w_token: float = 0.4
    w_exact: float = 0.5
    w_length: float = 0.3
    w_diversity: float = 0.2
    termination_bonus: float = 0.1
    truncation_penalty: float = 0.3
    graded_validity: bool = True


@dataclass
class RewardResult:
    reward: float
    valid: bool
    first_violation_rule: str | None
    terminated: bool
    block_accuracy: float = 0.0
    token_accuracy: float = 0.0
    exact_match: bool = False


def compute_reward(
    prefix: list[str] | tuple[str, ...],
    completion: list[str] | tuple[str, ...],
    gold: list[str] | tuple[str, ...],
    *,
    terminated: bool,
    config: RewardConfig,
) -> RewardResult:
    """Gated composite reward for a sampled completion against its gold suffix.

    ``reward = validity_gate * quality + termination/truncation`` where ``quality``
    is the weighted sum of block accuracy, token accuracy, an exact-match bonus, a
    length-sanity factor, and a block-diversity factor (see :class:`RewardConfig`).
    """
    full = [*prefix, *completion]
    violations = validate_sequence(full)
    valid = not violations
    rule = None if valid else first_violated_rule(full)

    if valid:
        gate = 1.0
    elif config.graded_validity:
        gate = max(0.0, 1.0 - len(violations) / max(1, len(full)))
    else:
        gate = 0.0

    comp = list(completion)
    gold_list = list(gold)
    block_acc = _block_accuracy(comp, gold_list)
    token_acc = _token_accuracy(comp, gold_list)
    exact = comp == gold_list
    length_match = math.exp(-abs(len(comp) - len(gold_list)) / max(1, len(gold_list)))
    runs = block_runs(comp)
    diversity = (len(set(runs)) / len(runs)) if runs else 0.0

    quality = (
        config.w_block * block_acc
        + config.w_token * token_acc
        + config.w_exact * (1.0 if exact else 0.0)
        + config.w_length * length_match
        + config.w_diversity * diversity
    )
    reward = gate * quality
    reward += config.termination_bonus if terminated else -config.truncation_penalty

    return RewardResult(
        reward=reward,
        valid=valid,
        first_violation_rule=rule,
        terminated=terminated,
        block_accuracy=block_acc,
        token_accuracy=token_acc,
        exact_match=exact,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompts(
    records: list[SequenceRecord],
    *,
    fractions: tuple[float, ...] = (0.6, 0.8),
    min_prefix: int = 2,
) -> list[tuple[str, list[str], list[str]]]:
    """Cut each record at the given fractions; prefix = head, gold = tail.

    Returns a list of ``(family, prefix_steps, gold_suffix)`` prompts. The gold
    suffix is the record's true continuation, used by :func:`compute_reward` to
    score fidelity. A record only yields a prompt at a fraction when the head is at
    least ``min_prefix`` steps and the tail is non-empty (so there is something to
    complete and to score against).
    """
    prompts: list[tuple[str, list[str], list[str]]] = []
    for record in records:
        steps = list(record.steps)
        n = len(steps)
        if n < min_prefix + 1:
            continue
        for frac in fractions:
            cut = max(min_prefix, int(round(frac * n)))
            cut = min(cut, n - 1)
            if cut < min_prefix:
                continue
            prompts.append((record.family, steps[:cut], steps[cut:]))
    return prompts


# ---------------------------------------------------------------------------
# GRPO trainer
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    group_size: int = 8
    steps: int = 200
    prompts_per_step: int = 8
    lr: float = 1e-5
    kl_coef: float = 0.02
    temperature: float = 1.0
    max_steps: int = 400
    grad_clip: float = 1.0
    mask_sampling: bool = False
    log_every: int = 10


class GRPOTrainer:
    """Group Relative Policy Optimization over verifiable completion rewards."""

    def __init__(
        self,
        policy: StepPolicy,
        prompts: list[tuple[str, list[str], list[str]]],
        *,
        config: GRPOConfig,
        reward_config: RewardConfig,
        seed: int = 0,
    ) -> None:
        if not prompts:
            raise ValueError("GRPOTrainer needs at least one prompt")
        self.policy = policy
        self.prompts = prompts
        self.config = config
        self.reward_config = reward_config
        self.rng = random.Random(seed)

        # Frozen reference copy of the *initial* policy for the KL term.
        self.reference = self._freeze_reference(policy)

        self.optimizer = torch.optim.AdamW(self.policy.model.parameters(), lr=config.lr)
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _freeze_reference(policy: StepPolicy) -> StepPolicy:
        ref_model = copy.deepcopy(policy.model)
        for param in ref_model.parameters():
            param.requires_grad_(False)
        ref_model.eval()
        ref = StepPolicy(
            ref_model,
            policy.vocabulary,
            kind=policy.kind,
            max_context=policy.max_context,
            device=policy.device,
            meta=policy.meta,
        )
        return ref

    def _group(self, family: str, prefix: list[str], gold: list[str]) -> dict[str, Any]:
        """Sample a group of completions for one prompt and accumulate its loss."""
        cfg = self.config
        self.policy.model.train()

        completions: list[list[str]] = []
        sample_logprobs: list[torch.Tensor] = []
        rewards: list[float] = []
        results: list[RewardResult] = []

        for _ in range(cfg.group_size):
            steps, logprobs, terminated = self.policy.sample_completion(
                family,
                prefix,
                max_steps=cfg.max_steps,
                temperature=cfg.temperature,
                mask_sampling=cfg.mask_sampling,
            )
            result = compute_reward(
                prefix, steps, gold, terminated=terminated, config=self.reward_config
            )
            completions.append(steps)
            sample_logprobs.append(logprobs)
            rewards.append(result.reward)
            results.append(result)

        reward_t = torch.tensor(rewards, dtype=torch.float32, device=self.policy.device)
        advantages = (reward_t - reward_t.mean()) / (reward_t.std(unbiased=False) + 1e-6)

        pg_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        for i, completion in enumerate(completions):
            logprobs = sample_logprobs[i]
            if logprobs.numel() == 0:
                continue
            # Policy-gradient term: advantage-weighted *mean* log-prob. Averaging
            # over length (rather than summing) removes the GRPO length bias where
            # longer rollouts receive proportionally larger-magnitude updates
            # (cf. Dr. GRPO). The sampled log-probs already carry gradients.
            pg_terms.append(advantages[i] * logprobs.mean())

            if cfg.kl_coef > 0:
                with torch.no_grad():
                    ref_logprobs = self.reference.completion_logprobs(
                        family, prefix, completion, mask_sampling=cfg.mask_sampling
                    )
                # Schulman's k3 unbiased, always-positive KL(policy||reference)
                # estimator per token: E_policy[exp(logq-logp) - (logq-logp) - 1].
                # Lower variance than the naive (logp - logq) k1 estimator, and it
                # never goes negative, so the regularizer can't accidentally reward
                # drift. Averaged over length to match the PG term's scale.
                log_ratio = ref_logprobs - logprobs  # logq - logp
                kl_terms.append((log_ratio.exp() - log_ratio - 1.0).mean())

        if not pg_terms:
            return {
                "loss": None,
                "rewards": rewards,
                "advantages": advantages.detach(),
                "results": results,
            }

        pg_loss = -torch.stack(pg_terms).mean()
        kl_loss = (
            torch.stack(kl_terms).mean()
            if (kl_terms and cfg.kl_coef > 0)
            else torch.zeros((), device=self.policy.device)
        )
        loss = pg_loss + cfg.kl_coef * kl_loss
        return {
            "loss": loss,
            "kl": float(kl_loss.detach().item()),
            "rewards": rewards,
            "advantages": advantages.detach(),
            "results": results,
        }

    def step(self) -> dict[str, Any]:
        """One GRPO update over ``prompts_per_step`` groups."""
        cfg = self.config
        prompts = [self.rng.choice(self.prompts) for _ in range(cfg.prompts_per_step)]

        self.optimizer.zero_grad(set_to_none=True)
        all_rewards: list[float] = []
        all_valid: list[bool] = []
        all_results: list[RewardResult] = []
        adv_mags: list[float] = []
        kls: list[float] = []
        loss_sum = 0.0
        n_back = 0

        # Backward each group as soon as it is built, then drop its graph, instead
        # of retaining every group's autograd graph until a single end-of-step
        # backward. Gradients accumulate in ``.grad`` (the optimizer step happens
        # once, after the loop), so this is mathematically gradient accumulation:
        # peak memory is one group's rollout graphs, not prompts_per_step of them.
        # Each loss is pre-scaled by 1/prompts_per_step so the accumulated gradient
        # matches a mean over the groups in this step.
        for family, prefix, gold in prompts:
            out = self._group(family, prefix, gold)
            all_rewards.extend(out["rewards"])
            all_valid.extend(r.valid for r in out["results"])
            all_results.extend(out["results"])
            adv_mags.append(float(out["advantages"].abs().mean().item()))
            if out["loss"] is not None:
                (out["loss"] / cfg.prompts_per_step).backward()
                loss_sum += float(out["loss"].detach().item())
                kls.append(out.get("kl", 0.0))
                n_back += 1

        if n_back:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            loss_value = loss_sum / n_back
        else:
            loss_value = float("nan")

        n = max(1, len(all_rewards))
        metrics = {
            "loss": loss_value,
            "mean_reward": sum(all_rewards) / n,
            "valid_rate": sum(1 for v in all_valid if v) / n,
            "block_accuracy": sum(r.block_accuracy for r in all_results) / n,
            "token_accuracy": sum(r.token_accuracy for r in all_results) / n,
            "exact_rate": sum(1 for r in all_results if r.exact_match) / n,
            "mean_adv_mag": sum(adv_mags) / max(1, len(adv_mags)),
            "kl": sum(kls) / max(1, len(kls)) if kls else 0.0,
        }
        return metrics

    def train(self) -> list[dict[str, Any]]:
        cfg = self.config
        for step_idx in range(1, cfg.steps + 1):
            metrics = self.step()
            metrics["step"] = step_idx
            self.history.append(metrics)
            if cfg.log_every and (step_idx == 1 or step_idx % cfg.log_every == 0):
                print(
                    f"step={step_idx}/{cfg.steps} "
                    f"loss={metrics['loss']:.4f} "
                    f"mean_reward={metrics['mean_reward']:.4f} "
                    f"valid_rate={metrics['valid_rate']:.4f} "
                    f"block_acc={metrics['block_accuracy']:.4f} "
                    f"exact={metrics['exact_rate']:.4f} "
                    f"mean_adv_mag={metrics['mean_adv_mag']:.4f} "
                    f"kl={metrics['kl']:.4f}",
                    flush=True,
                )
        return self.history


def build_minimal_lstm_checkpoint(
    records: list[SequenceRecord],
    out: str | Path,
    *,
    max_context: int = 192,
) -> Path:
    """Build a tiny randomly-initialized LSTM checkpoint for smoke testing.

    Constructs a :class:`Vocabulary` from ``records`` and a small
    :class:`LSTMModel`, then persists via :func:`save_lstm_checkpoint`.
    """
    from zero_hack.data import build_vocabulary

    vocabulary = build_vocabulary(records)
    config = LSTMConfig(embedding_dim=32, hidden_dim=32, num_layers=1, dropout=0.0)
    model = LSTMModel(
        vocab_size=len(vocabulary.id_to_token),
        config=config,
        pad_id=vocabulary.pad_id,
    )
    return save_lstm_checkpoint(out, model, vocabulary, max_context=max_context)


__all__ = [
    "GRPOConfig",
    "GRPOTrainer",
    "RewardConfig",
    "RewardResult",
    "StepPolicy",
    "build_minimal_lstm_checkpoint",
    "build_prompts",
    "compute_reward",
    "detect_checkpoint_kind",
    "load_policy",
    "save_policy",
]
