"""Decision gate.

Every field leaves this service with a verdict attached, and the verdict is
part of the contract:

    RULE      -- resolved by the VIN standard. Auditable, no model involved.
    MODEL     -- predicted, and confident enough to publish.
    ESCALATE  -- the system has a guess but not the right to state it.
    UNKNOWN   -- nothing available. Said plainly.

A decoder that returns "A5" and one that returns "A5 or S5, we cannot tell"
are different products. Downstream consumers -- pricing, damage matching,
insurance risk -- need to know which one they just received, because a
silently wrong model is more expensive than a declared gap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum

from .deterministic import InvalidVin, decode_make, decode_year, parse
from .model import PREFIX_LENGTH, Predictor

# Below these calibrated probabilities an answer is reported but not asserted.
#
# One threshold per field, because the fields do not behave alike. Repeated
# evaluation (scripts/evaluate.py, 10 repeats) shows model reaching 94.4%
# precision at 0.80 while still asserting 78% of traffic; body at the same
# threshold asserts only 58%. Body's probability mass simply sits lower --
# 11 classes with overlapping evidence -- so a shared threshold silently
# escalates two fifths of body decodes for no precision gain worth having.
DEFAULT_THRESHOLDS: dict[str, float] = {"model": 0.80, "body": 0.70}
DEFAULT_THRESHOLD = 0.80  # fallback for a field with no entry above


class Verdict(str, Enum):
    RULE = "RULE"
    MODEL = "MODEL"
    ESCALATE = "ESCALATE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Field:
    value: str | int | None
    verdict: Verdict
    confidence: float | None = None
    reason: str | None = None
    candidates: tuple | None = None


@dataclass
class Decoded:
    vin: str
    make: Field
    model: Field
    year: Field
    body: Field

    def as_dict(self) -> dict:
        out = {"vin": self.vin}
        for name in ("make", "model", "year", "body"):
            field = asdict(getattr(self, name))
            field["verdict"] = Verdict(field["verdict"]).value
            if field["candidates"] is not None:
                field["candidates"] = list(field["candidates"])
            out[name] = field
        return out


class Decoder:
    def __init__(
        self,
        predictor: Predictor,
        thresholds: dict[str, float] | None = None,
        threshold: float | None = None,
        year_rule_applies: Callable[[str, object], bool] | None = None,
    ):
        """thresholds maps field -> cutoff; threshold sets one value for all fields.

        year_rule_applies is the applicability policy for the model-year rule.
        There is no default: without one the service offers a year reading and
        declines to assert it, which is the correct behaviour when nothing
        establishes that position 10 encodes a year on the VIN at hand.
        """
        self._predictor = predictor
        self._year_rule_applies = year_rule_applies
        if threshold is not None:
            self._thresholds = {field: threshold for field in ("model", "body")}
        else:
            self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def threshold_for(self, field: str) -> float:
        return self._thresholds.get(field, DEFAULT_THRESHOLD)

    def _learned(self, field: str, prefix: str) -> Field:
        # A prefix the corpus itself could not agree on is never asserted,
        # however confident the classifier is about its neighbours.
        contested = self._predictor.known_ambiguous(field, prefix)
        if contested:
            return Field(
                None,
                Verdict.ESCALATE,
                reason="prefix carries conflicting labels in the corpus",
                candidates=tuple(contested),
            )

        label, probability = self._predictor.predict(field, prefix)
        if label is None:
            return Field(None, Verdict.UNKNOWN, reason="no trained model for this field")
        cutoff = self.threshold_for(field)
        if probability >= cutoff:
            return Field(label, Verdict.MODEL, confidence=round(probability, 3))
        return Field(
            label,
            Verdict.ESCALATE,
            confidence=round(probability, 3),
            reason=f"below confidence threshold {cutoff} for {field}",
        )

    def decode(self, vin: str) -> Decoded:
        """Decode one VIN. Raises InvalidVin for structurally impossible input."""
        parts = parse(vin)  # deliberately propagates: a malformed VIN is a 4xx, not a guess
        normalised = vin.strip().upper()
        prefix = normalised[:PREFIX_LENGTH]

        make = decode_make(parts)
        make_field = (
            Field(make, Verdict.RULE, reason=f"WMI {parts.wmi}")
            if make
            else Field(None, Verdict.UNKNOWN, reason=f"WMI {parts.wmi} not in register")
        )

        reading = decode_year(parts, normalised, self._year_rule_applies)
        if reading.value is None:
            year_field = Field(None, Verdict.UNKNOWN, reason=reading.basis)
        elif reading.resolved:
            year_field = Field(reading.value, Verdict.RULE, reason=reading.basis)
        else:
            # Two readings survive the 30-year cycle. The newer one is the
            # better bet, but a bet is not a rule.
            year_field = Field(
                reading.value,
                Verdict.ESCALATE,
                reason=reading.basis,
                candidates=reading.candidates,
            )

        return Decoded(
            vin=normalised,
            make=make_field,
            model=self._learned("model", prefix),
            year=year_field,
            body=self._learned("body", prefix),
        )


__all__ = [
    "Decoder",
    "Decoded",
    "Field",
    "Verdict",
    "InvalidVin",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_THRESHOLD",
]
