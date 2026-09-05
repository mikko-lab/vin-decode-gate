"""Inference API.

Two endpoints: a single decode and a batch decode. Artifacts are loaded once
at import, not per request -- the workload is high-volume by assumption, and
deserialising a pipeline per call is the first thing that would fall over.
"""

from __future__ import annotations

import os
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field as PydanticField

from .deterministic import InvalidVin, north_american_passenger_car
from .gate import DEFAULT_THRESHOLDS, Decoder
from .model import Predictor

ARTIFACT_DIR = os.environ.get("VINDECODE_ARTIFACTS", "artifacts")
# VINDECODE_THRESHOLDS overrides per field, e.g. "model=0.85,body=0.65".
THRESHOLDS = dict(DEFAULT_THRESHOLDS)
if override := os.environ.get("VINDECODE_THRESHOLDS"):
    THRESHOLDS.update(
        {
            field.strip(): float(value)
            for field, _, value in (pair.partition("=") for pair in override.split(","))
            if field.strip()
        }
    )

app = FastAPI(
    title="VIN Decode Gate",
    version="1.0.0",
    description="VIN decoding where every field states how it was resolved.",
)
# The model-year rule is only asserted when an applicability policy is chosen
# explicitly. Unset, the service offers year readings and asserts none.
#
# An unrecognised name is a startup failure, not a silent fallback. Quietly
# treating a typo as "no policy" would leave the service running, reporting
# healthy, and escalating every year decode it was configured to assert --
# a config error that only shows up as a rising review queue weeks later.
_POLICIES = {"north-american-passenger-car": north_american_passenger_car}
YEAR_POLICY = os.environ.get("VINDECODE_YEAR_POLICY", "").strip().lower() or None
if YEAR_POLICY is not None and YEAR_POLICY not in _POLICIES:
    raise RuntimeError(
        f"VINDECODE_YEAR_POLICY={YEAR_POLICY!r} is not a known applicability policy. "
        f"Use one of {sorted(_POLICIES)}, or leave it unset to assert no years."
    )
_year_rule = _POLICIES.get(YEAR_POLICY) if YEAR_POLICY else None

_decoder = Decoder(
    Predictor(ARTIFACT_DIR), thresholds=THRESHOLDS, year_rule_applies=_year_rule
)


class BatchRequest(BaseModel):
    vins: list[str] = PydanticField(..., min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "thresholds": THRESHOLDS,
        "artifacts": ARTIFACT_DIR,
        # Reports the policy actually in force, never the raw setting.
        "year_rule_policy": YEAR_POLICY if _year_rule else None,
    }


@app.get("/decode/{vin}")
def decode(vin: str) -> dict:
    try:
        return _decoder.decode(vin).as_dict()
    except InvalidVin as exc:
        # Malformed input is a client error, never a low-confidence guess.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/decode")
def decode_batch(request: BatchRequest) -> dict:
    results, errors = [], []
    for vin in request.vins:
        try:
            results.append(_decoder.decode(vin).as_dict())
        except InvalidVin as exc:
            errors.append({"vin": vin, "error": str(exc)})

    # Verdict mix is the number worth alerting on: a rising ESCALATE share
    # means the input distribution has moved away from the training data.
    verdicts = Counter(
        item[field]["verdict"] for item in results for field in ("make", "model", "year", "body")
    )
    return {"results": results, "errors": errors, "verdict_counts": dict(verdicts)}
