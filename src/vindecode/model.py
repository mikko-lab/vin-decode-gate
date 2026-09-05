"""Learned layer.

The model is asked only for what the deterministic layer cannot answer:
model family and body type. Both are encoded in the VDS (positions 4-9),
which each manufacturer defines privately -- there is no public rule to
apply, so this is where learning genuinely earns its place.

Features are character n-grams over the decodable prefix (positions 1-9).
The masked serial is excluded on purpose: it carries no information and
would only invite the model to memorise noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .calibration import (
    apply_temperature,
    as_logits,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
)
from .deterministic import VIN_ALPHABET, VIN_LENGTH
from .labels import normalize_model, resolve

PREFIX_LENGTH = 9  # WMI + VDS; everything after this is masked or serial


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)),
            ("classifier", LogisticRegression(max_iter=2000, C=10.0)),
        ]
    )


@dataclass
class TrainingReport:
    """What the training run actually produced, for the record."""

    field: str
    n_train: int
    n_calibration: int
    n_test: int
    n_classes: int
    accuracy: float
    temperature: float
    ece_raw: float
    ece_calibrated: float
    withheld_ambiguous: int
    reliability: list

    def as_dict(self) -> dict:
        return {
            "field": self.field,
            "n_train": self.n_train,
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
            "n_classes": self.n_classes,
            "accuracy": round(self.accuracy, 4),
            "temperature": round(self.temperature, 4),
            "ece_raw": round(self.ece_raw, 4),
            "ece_calibrated": round(self.ece_calibrated, 4),
            "withheld_ambiguous": self.withheld_ambiguous,
            "reliability": self.reliability,
        }


@dataclass
class Corpus:
    """The corpus reduced to the key the model actually sees.

    Conflicts must be resolved at the prefix, not the VIN. Positions 10-17
    are either masked or excluded from the features, so two VINs sharing a
    prefix are one example as far as the classifier is concerned. Resolving
    per VIN leaves those conflicts intact under different row identities --
    and, worse, lets the same prefix land in both the training and the test
    split, where it is then scored as generalisation instead of recall.
    """

    frame: pd.DataFrame
    ambiguous: dict[str, dict[str, tuple[str, ...]]]
    withheld: dict[str, int]
    n_vins: int


def _resolve_body(labels: list[str]) -> tuple[str | None, bool]:
    """Body has no parent-child structure, so any disagreement is ambiguity."""
    distinct = {b for b in labels if b}
    if not distinct:
        return None, False
    if len(distinct) == 1:
        return distinct.pop(), False
    return None, True


def build_corpus(csv_path: str | Path) -> Corpus:
    """Collapse the raw corpus to one row per prefix with clean labels.

    Ambiguity is recorded per field rather than per row: a prefix whose model
    is contested still carries a usable body label, and dropping the whole
    row would throw that away.
    """
    raw = pd.read_csv(csv_path, dtype=str)
    # Normalise exactly as the decoder does before grouping. Training grouped
    # raw strings while the API upper-cases its input, so a lower-case VIN in
    # the corpus produced a second group and an ambiguity key the gate could
    # never look up -- while TF-IDF, which lower-cases by default, happily
    # merged the two into one feature representation.
    raw["vin"] = raw["vin"].fillna("").str.strip().str.upper()
    raw = raw[raw["vin"].str.fullmatch(f"[{VIN_ALPHABET}]{{{VIN_LENGTH}}}", na=False)]
    raw["prefix"] = raw["vin"].str[:PREFIX_LENGTH]

    rows: list[dict] = []
    ambiguous: dict[str, dict[str, tuple[str, ...]]] = {"model": {}, "body": {}}

    for prefix, group in raw.groupby("prefix"):
        row = {"prefix": prefix, "model": None, "body": None}

        model_label, model_conflict = resolve(list(group["model"].dropna()))
        if model_conflict:
            ambiguous["model"][prefix] = tuple(
                sorted({normalize_model(m) for m in group["model"].dropna()} - {None})
            )
        row["model"] = model_label

        body_label, body_conflict = _resolve_body(list(group["body"].dropna()))
        if body_conflict:
            ambiguous["body"][prefix] = tuple(sorted(set(group["body"].dropna())))
        row["body"] = body_label

        rows.append(row)

    frame = pd.DataFrame(rows, columns=["prefix", "model", "body"])
    return Corpus(
        frame=frame,
        ambiguous=ambiguous,
        withheld={field: len(prefixes) for field, prefixes in ambiguous.items()},
        n_vins=int(raw["vin"].nunique()),
    )


def train_field(corpus: "Corpus", field: str, seed: int = 7):
    """Fit one classifier and calibrate it. Returns (artifact, report).

    The data is split three ways: fit on train, fit the temperature on a
    calibration split the classifier has never seen, and report on test.
    Calibrating on the training split would give a temperature near 1 and a
    flattering ECE, because the classifier is already confident where it has
    memorised.
    """
    usable = corpus.frame.dropna(subset=[field])
    # A class seen once cannot be split into train and test honestly.
    counts = usable[field].value_counts()
    usable = usable[usable[field].isin(counts[counts >= 2].index)]

    x_train, x_rest, y_train, y_rest = train_test_split(
        usable["prefix"], usable[field], test_size=0.4, random_state=seed, stratify=usable[field]
    )
    # No stratification on the second split: the rare classes cannot support it.
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
    correct = (pipeline.classes_[calibrated.argmax(axis=1)] == truth).astype(float)

    report = TrainingReport(
        field=field,
        n_train=len(x_train),
        n_calibration=len(x_cal),
        n_test=len(x_test),
        n_classes=int(usable[field].nunique()),
        accuracy=float(correct.mean()),
        temperature=temperature,
        ece_raw=expected_calibration_error(raw.max(axis=1), correct),
        ece_calibrated=expected_calibration_error(calibrated.max(axis=1), correct),
        withheld_ambiguous=corpus.withheld.get(field, 0),
        reliability=reliability_table(calibrated.max(axis=1), correct),
    )
    return (
        {
            "pipeline": pipeline,
            "temperature": temperature,
            "ambiguous_prefixes": corpus.ambiguous.get(field, {}),
        },
        report,
    )


def train(csv_path: str | Path, out_dir: str | Path) -> list[TrainingReport]:
    """Train every learned field and persist artifacts to `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(csv_path)

    reports = []
    for field in ("model", "body"):
        artifact, report = train_field(corpus, field)
        joblib.dump(artifact, out / f"{field}.joblib")
        reports.append(report)

    (out / "training_report.json").write_text(
        json.dumps(
            {
                "n_vins": corpus.n_vins,
                "n_prefixes": int(len(corpus.frame)),
                "fields": [r.as_dict() for r in reports],
            },
            indent=2,
        )
        + "\n"
    )
    return reports


class Predictor:
    """Loads persisted artifacts and returns a label with a calibrated probability."""

    def __init__(self, artifact_dir: str | Path):
        directory = Path(artifact_dir)
        self._artifacts = {
            field: joblib.load(directory / f"{field}.joblib")
            for field in ("model", "body")
            if (directory / f"{field}.joblib").exists()
        }

    @property
    def temperatures(self) -> dict[str, float]:
        return {field: art["temperature"] for field, art in self._artifacts.items()}

    def known_ambiguous(self, field: str, prefix: str) -> tuple[str, ...] | None:
        """Candidate labels when this prefix was contested in training.

        Withholding a contested prefix from training is not enough on its own:
        neighbouring prefixes will still produce a confident prediction for it.
        The knowledge has to survive into inference, so it ships in the
        artifact and the gate consults it.
        """
        artifact = self._artifacts.get(field)
        if artifact is None:
            return None
        return artifact.get("ambiguous_prefixes", {}).get(prefix)

    def predict(self, field: str, prefix: str) -> tuple[str | None, float]:
        artifact = self._artifacts.get(field)
        if artifact is None:
            return None, 0.0
        pipeline = artifact["pipeline"]
        probabilities = apply_temperature(
            as_logits(pipeline.decision_function([prefix])), artifact["temperature"]
        )[0]
        best = int(np.argmax(probabilities))
        return str(pipeline.classes_[best]), float(probabilities[best])
