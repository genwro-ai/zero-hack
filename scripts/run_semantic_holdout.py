#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import complete_sequence, predict_anomaly
from zero_hack.models.common import load_split_records, pick_device
from zero_hack.models.semantic_lm import (
    CausalLM,
    LearnedEmbedding,
    Predictor,
    SemanticEmbedding,
    fit,
)


def build_vocab(cache):
    names = cache["names"]
    vectors = cache["vectors"]
    tokens = list(SPECIAL_TOKENS) + list(FAMILY_TOKENS.values()) + names
    token_to_id = {token: i for i, token in enumerate(tokens)}
    vocab = Vocabulary(token_to_id=token_to_id, id_to_token=tuple(tokens))
    matrix = torch.zeros(len(tokens), vectors.size(1))
    for name, vector in zip(names, vectors, strict=True):
        matrix[token_to_id[name]] = vector
    return vocab, matrix


def predict_and_score(predictor, eval_dir, pred_dir, threshold):
    pred_dir.mkdir(parents=True, exist_ok=True)
    valid = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
    io.write_next_step_predictions(
        pred_dir / "nextstep.csv",
        [
            {
                "example_id": r["example_id"],
                "ranks": predictor.predict_topk(r["family"], r["partial_sequence"], 5),
            }
            for r in valid
        ],
    )
    io.write_completion_predictions(
        pred_dir / "completion.csv",
        [
            {
                "example_id": r["example_id"],
                "steps": complete_sequence(predictor, r["family"], r["partial_sequence"]),
            }
            for r in valid
        ],
    )
    anomaly = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
    io.write_anomaly_predictions(
        pred_dir / "anomaly.csv",
        [
            {
                "example_id": r["example_id"],
                **predict_anomaly(predictor, r["family"], r["sequence"], "likelihood", threshold),
            }
            for r in anomaly
        ],
    )
    files = {"next_step": "nextstep", "completion": "completion", "anomaly": "anomaly"}
    inputs = {
        "next_step": "eval_input_valid",
        "completion": "eval_input_valid",
        "anomaly": "eval_input_anomaly",
    }
    result = {}
    for task, stem in files.items():
        metrics = score_task(
            task,
            ground_truth=eval_dir / f"{stem}_truth.csv",
            predictions=pred_dir / f"{stem}.csv",
            eval_input=eval_dir / f"{inputs[task]}.csv",
        )
        result[task] = metrics.get("all", metrics)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir", default=str(PROJECT_ROOT / "data/generated/valid_s005k/splits")
    )
    parser.add_argument("--holdout", default="ic", choices=("mosfet", "igbt", "ic"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(None)
    dataset = Path(args.splits_dir).parent.name
    cache = torch.load(PROJECT_ROOT / "data/generated/step_name_sbert.pt", weights_only=False)
    vocab, vectors = build_vocab(cache)
    vectors = vectors.to(device)
    bundle = load_split_records(
        args.splits_dir, holdout_family=args.holdout, limit_per_family=args.limit
    )
    vocab_size = len(vocab.id_to_token)

    embeddings = {
        "baseline": lambda: LearnedEmbedding(vocab_size, args.d_model),
        "sbert": lambda: SemanticEmbedding(vectors, args.d_model),
    }
    summary = {}
    for name, make_embedding in embeddings.items():
        print(f"== {name} ==")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model = CausalLM(make_embedding(), args.d_model, 8, args.layers, 256, 0.1)
        model = fit(
            model, bundle.records["train"], vocab, device, args.epochs, args.batch_size, args.lr
        )
        predictor = Predictor(model, vocab, device, 256)
        threshold = tune_anomaly_threshold(
            predictor, bundle.records["valid"], n_valid=200, n_invalid=129, seed=args.seed
        ).threshold
        summary[name] = {}
        for view in ("id", "ood"):
            eval_dir = PROJECT_ROOT / "data/eval" / dataset / f"holdout_{args.holdout}" / view
            pred_dir = PROJECT_ROOT / "outputs/preds/semantic" / args.holdout / name / view
            summary[name][view] = predict_and_score(predictor, eval_dir, pred_dir, threshold)

    print(f"\n#### holdout={args.holdout} (train on the other two families) ####")
    for view in ("id", "ood"):
        print(f"== {view.upper()} ==")
        for name in embeddings:
            ns = summary[name][view]["next_step"]
            c = summary[name][view]["completion"]
            a = summary[name][view]["anomaly"]
            print(
                f"{name:9s}| T1 top1={ns['top1']:.3f} mrr={ns['mrr']:.3f}"
                f" | T2 block={c['block_accuracy']:.3f} token={c['token_accuracy']:.3f}"
                f" | T3 f1={a['f1']:.3f} auc={a['roc_auc']}"
            )


if __name__ == "__main__":
    main()
