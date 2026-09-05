"""Deterministic layer.

Everything the VIN standard (ISO 3779) guarantees is resolved here, by rule.
No model is consulted for anything a rule can answer. Rules that do not
apply return None -- they never guess.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass

# ISO 3779 excludes I, O and Q to avoid confusion with 1 and 0.
VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
VIN_LENGTH = 17

# World Manufacturer Identifier -> make. Sourced from the SAE/ISO WMI register,
# restricted to the manufacturers present in the training corpus.
WMI_MAKE: dict[str, str] = {
    "WAU": "Audi", "WA1": "Audi", "WUA": "Audi", "TRU": "Audi", "WAV": "Audi",
    "WBA": "BMW", "WBS": "BMW", "WBX": "BMW", "WBY": "BMW", "WB1": "BMW",
    "5UX": "BMW", "5YM": "BMW", "3MW": "BMW", "4US": "BMW", "X4X": "BMW",
}

# Position 10 model-year code. The alphabet repeats on a 30-year cycle, so
# every code has two readings: B is 1981 or 2011, 7 is 2007 or 2037.
_YEAR_LETTERS = "ABCDEFGHJKLMNPRSTVWXY"
_LETTER_EPOCHS = (1980, 2010)
_DIGIT_EPOCHS = (2000, 2030)

# Transliteration and weights for the ISO 7064 check digit at position 9.
# North American VINs must carry a valid one; most European VINs do not,
# which is what lets us tell the two conventions apart.
_TRANSLITERATION = {
    **{str(d): d for d in range(10)},
    **{c: v for v, c in enumerate("ABCDEFGHJKLMNPRSTUVWXYZ", start=1) if c not in "OQI"},
}
_TRANSLITERATION.update({"J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9})
_TRANSLITERATION.update({"S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9})
_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


@dataclass(frozen=True)
class VinParts:
    """Structural decomposition of a VIN."""

    wmi: str  # positions 1-3, manufacturer
    vds: str  # positions 4-9, vehicle descriptor (model, body, engine)
    vis: str  # positions 10-17, vehicle identifier (year, plant, serial)


class InvalidVin(ValueError):
    """Raised when the input cannot be a VIN at all."""


def parse(vin: str) -> VinParts:
    """Split a VIN into its three ISO 3779 sections.

    Raises InvalidVin on anything that is structurally not a VIN. Masked
    serial characters ('X') are tolerated: the redacted corpus uses them and
    the serial carries no decodable meaning anyway.
    """
    if vin is None:
        raise InvalidVin("VIN is missing")
    v = vin.strip().upper()
    if len(v) != VIN_LENGTH:
        raise InvalidVin(f"VIN must be {VIN_LENGTH} characters, got {len(v)}")
    illegal = set(v) - set(VIN_ALPHABET)
    if illegal:
        raise InvalidVin(f"VIN contains characters outside ISO 3779: {sorted(illegal)}")
    return VinParts(wmi=v[0:3], vds=v[3:9], vis=v[9:17])


def decode_make(parts: VinParts) -> str | None:
    """Make from the WMI register. None when the WMI is not in the register."""
    return WMI_MAKE.get(parts.wmi)


@dataclass(frozen=True)
class YearReading:
    """A model year, and whether the VIN alone proves it.

    `resolved` is False when the 30-year cycle leaves two plausible answers.
    `value` is then the more recent one -- the better bet, not a fact.
    """

    value: int | None
    candidates: tuple[int, ...]
    resolved: bool
    basis: str


def has_valid_check_digit(vin: str) -> bool:
    """True when position 9 satisfies the ISO 7064 check, as NA VINs must.

    A masked or absent serial makes this unanswerable, so a VIN carrying 'X'
    outside position 9 is rejected rather than assumed valid.
    """
    text = vin.strip().upper()
    if len(text) != VIN_LENGTH or any(c not in _TRANSLITERATION for c in text.replace("X", "0")):
        return False
    if "X" in text[9:]:  # redacted serial: the check digit cannot be verified
        return False
    try:
        total = sum(_TRANSLITERATION[c] * w for c, w in zip(text, _WEIGHTS))
    except KeyError:
        return False
    remainder = total % 11
    expected = "X" if remainder == 10 else str(remainder)
    return text[8] == expected


def _cycle_readings(code: str) -> dict[str, int]:
    """Both readings of a year code, keyed by cycle. Empty if not a year code.

    No plausibility filtering happens here. Filtering before the cycle is
    chosen is how an out-of-range modern year silently becomes a confident
    answer from the older cycle.
    """
    if code in "123456789":
        base, offset = _DIGIT_EPOCHS, int(code)
    elif code in _YEAR_LETTERS:
        base, offset = _LETTER_EPOCHS, _YEAR_LETTERS.index(code)
    else:
        return {}
    older, modern = sorted(base)
    return {"older": older + offset, "modern": modern + offset}


def north_american_passenger_car(vin: str, parts: VinParts) -> bool:
    """One applicability policy, offered rather than assumed.

    ISO 3779 assigns WMIs beginning 1-5 to North America, and vehicles sold
    there carry a check digit. Both are verifiable from the VIN. What is not
    verifiable is vehicle class and weight, which is what the NHTSA rule is
    actually scoped by -- so this policy encodes an assumption, and callers
    opt into it knowingly rather than receiving it as a default.
    """
    return parts.wmi[0] in "12345" and has_valid_check_digit(vin)


def decode_year(
    parts: VinParts,
    vin: str | None = None,
    applies: Callable[[str, VinParts], bool] | None = None,
    today: _dt.date | None = None,
) -> YearReading:
    """Model year from position 10, asserted only where the rule provably applies.

    Three separate questions, answered in order, because collapsing any two
    of them is how this function was wrong twice before:

    1. Is the character a year code at all? If not, UNKNOWN.
    2. Does the model-year rule apply to this VIN? Position 10 is a US/NHTSA
       requirement, not a universal one, and its scope is set by market,
       vehicle class and weight. A valid check digit shows arithmetic
       consistency and nothing else -- an arbitrary VIN satisfies it about
       one time in eleven. So applicability must be supplied by the caller
       through `applies`; there is no default, and without one no year is
       ever asserted.
    3. Which of the two 30-year cycles is meant? Position 7 answers this
       where the rule applies. The cycle is chosen first and the plausibility
       window checked afterwards: if the chosen cycle yields an impossible
       year the answer is a contradiction, never the other cycle.
    """
    latest = (today or _dt.date.today()).year + 1
    readings = _cycle_readings(parts.vis[0])
    if not readings:
        return YearReading(None, (), False, "position 10 is not a model-year code")

    plausible = tuple(
        sorted({y for y in readings.values() if 1980 <= y <= latest}, reverse=True)
    )

    policy_verdict = None if not (vin and applies) else applies(vin, parts)

    if policy_verdict:
        # NHTSA convention: position 7 alphabetic -> modern cycle, digit -> older.
        cycle = "modern" if parts.vds[3].isalpha() else "older"
        picked = readings[cycle]
        if not 1980 <= picked <= latest:
            return YearReading(
                None,
                plausible,
                False,
                f"the {cycle} cycle indicated by position 7 gives {picked}, "
                "which is not a possible model year; the VIN is inconsistent",
            )
        return YearReading(
            picked,
            (picked,),
            True,
            "VIN positions 10 and 7, under the caller's applicability policy",
        )

    if not plausible:
        return YearReading(None, (), False, "no plausible model year for this code")

    # These are different situations and the caller should be able to tell
    # them apart: one is an unconfigured service, the other is a policy that
    # looked at this VIN and declined it.
    if policy_verdict is None:
        reason = (
            "no applicability policy supplied, so nothing shows position 10 "
            "encodes a year on this VIN"
        )
    else:
        reason = "the applicability policy does not cover this VIN"
    if len(plausible) > 1:
        reason += "; both 30-year cycles remain possible"
    return YearReading(max(plausible), plausible, False, reason)
