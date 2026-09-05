"""Repeated evaluation.

Every number quoted in the README is produced here, so a reader can rerun it
instead of taking it on trust:

    python scripts/evaluate.py --repeats 10

Rows here are prefixes, not VINs. Splitting on VINs would put the same
nine-character prefix in both the training and the test half, where a
memorised answer scores as generalisation -- measured at 30.6% of test rows
and 98.3% accuracy on them, against 83.8% on genuinely unseen prefixes.

Why repeated stratified holdout and not k-fold. Stratified k-fold at k=5
requires five examples of every class; this corpus has model classes with
two. Dropping those classes to enable k-fold would quietly improve every
metric by removing the hardest cases, which is the opposite of the point.
Repeated stratified holdout keeps them: at a 60/20/20 split a class needs
only two members to be splittable, and repeating over seeds recovers the
variance estimate that a single split cannot give.

The standard deviations below are the reason the repeats exist. On 270 test
rows a single split moves by a couple of points, so any single-run number is
worth roughly its mean plus or minus that spread.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vindecode.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from vindecode.model import _pipeline, build_corpus  # noqa: E402

THRESHOLDS = (0.60, 0.70, 0.80, 0.90)


def evaluate_field(frame, field: str, repeats: int) -> dict:
    usable = frame.dropna(subset=[field])
    counts = usable[field].value_counts()
    usable = usable[usable[field].isin(counts[counts >= 2].index)]

    runs = []
    for seed in range(repeats):
        x_train, x_rest, y_train, y_rest = train_test_split(
            usable["prefix"],
            usable[field],
            test_size=0.4,
            random_state=seed,
            stratify=usable[field],
        )
        x_cal, x_test, y_cal, y_test = train_test_split(
            x_rest, y_rest, test_size=0.5, random_state=seed
        )

        pipeline = _pipeline()
        pipeline.fit(x_train, y_train)
        temperature = fit_temperature(
            pipeline.decision_function(x_cal), y_cal.to_numpy(), pipeline.classes_
        )

        logits = pipeline.decision_function(x_test)
        truth = y_test.to_numpy()
        raw = apply_temperature(logits, 1.0)
        calibrated = apply_temperature(logits, temperature)
        confidence = calibrated.max(axis=1)
        correct = (pipeline.classes_[calibrated.argmax(axis=1)] == truth).astype(float)

        gate = {}
        for threshold in THRESHOLDS:
            asserted = confidence >= threshold
            gate[threshold] = {
                "coverage": float(asserted.mean()),
                "accuracy_asserted": float(correct[asserted].mean()) if asserted.any() else np.nan,
                "accuracy_escalated": (
                    float(correct[~asserted].mean()) if (~asserted).any() else np.nan
                ),
            }

        runs.append(
            {
                "accuracy": float(correct.mean()),
                "mean_confidence": float(confidence.mean()),
                "temperature": temperature,
                "ece_raw": expected_calibration_error(raw.max(axis=1), correct),
                "ece_calibrated": expected_calibration_error(confidence, correct),
                "gate": gate,
            }
        )

    def summarise(key):
        values = [run[key] for run in runs]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}

    return {
        "field": field,
        "repeats": repeats,
        "n_classes": int(usable[field].nunique()),
        "n_usable_prefixes": int(len(usable)),
        "accuracy": summarise("accuracy"),
        "mean_confidence": summarise("mean_confidence"),
        "temperature": summarise("temperature"),
        "ece_raw": summarise("ece_raw"),
        "ece_calibrated": summarise("ece_calibrated"),
        "gate": {
            str(threshold): {
                metric: {
                    "mean": float(np.nanmean([r["gate"][threshold][metric] for r in runs])),
                    "std": float(np.nanstd([r["gate"][threshold][metric] for r in runs], ddof=1)),
                }
                for metric in ("coverage", "accuracy_asserted", "accuracy_escalated")
            }
            for threshold in THRESHOLDS
        },
    }


def _pct(stat: dict) -> str:
    return f"{stat['mean']:.1%} ± {stat['std']:.1%}"


def report(result: dict) -> None:
    print(f"\n=== {result['field']} ===")
    print(
        f"{result['n_classes']} classes, {result['n_usable_prefixes']} usable prefixes, "
        f"{result['repeats']} repeats"
    )
    print(f"  accuracy            {_pct(result['accuracy'])}")
    print(
        f"  confidence (stated) {_pct(result['mean_confidence'])}"
        f"   <- compare with accuracy above"
    )
    print(
        f"  temperature         {result['temperature']['mean']:.3f} "
        f"± {result['temperature']['std']:.3f}"
    )
    print(
        f"  ECE raw -> cal      {result['ece_raw']['mean']:.4f} -> "
        f"{result['ece_calibrated']['mean']:.4f}"
    )
    print(f"\n  {'thr':>5} {'asserted':>16} {'acc | asserted':>18} {'acc | escalated':>18}")
    for threshold, stats in result["gate"].items():
        print(
            f"  {threshold:>5} {_pct(stats['coverage']):>16} "
            f"{_pct(stats['accuracy_asserted']):>18} "
            f"{_pct(stats['accuracy_escalated']):>18}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repeated stratified evaluation.")
    parser.add_argument("--data", default="data/ml-engineer-challenge-redacted-data.csv")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", default="artifacts/evaluation.json")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    corpus = build_corpus(args.data)
    print(
        f"{corpus.n_vins} VINs collapse to {len(corpus.frame)} prefixes "
        f"(the model's actual key). Withheld as ambiguous: "
        + ", ".join(f"{field} {n}" for field, n in corpus.withheld.items())
    )

    results = [evaluate_field(corpus.frame, field, args.repeats) for field in ("model", "body")]
    for result in results:
        report(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
