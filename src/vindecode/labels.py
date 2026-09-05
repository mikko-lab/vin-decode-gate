"""Label hygiene.

The corpus labels the same VIN inconsistently. Three distinct problems hide
behind that, and they need three different answers:

1. Formatting noise -- "4" and "4 Series" are the same car written twice;
   "530D (EUR)" and "428I (USA)" carry a market suffix that is not part of
   the model. These are canonicalised.

2. Granularity conflicts -- "X5" and "X Series" are both correct, at
   different depths. The specific label wins; the coarse one is its parent.

3. Genuine ambiguity -- "Q5" and "SQ5", "A5" and "S5" are different cars
   sharing a VIN prefix. Nothing in the visible VIN separates them once the
   serial is masked. These are marked ambiguous and withheld from training,
   so the model never learns to guess a coin flip.

Collapsing all three into one "dedupe" step is how a decoder ends up
confidently wrong about an SQ5.
"""

from __future__ import annotations

import re

# Coarse family labels that act as parents of more specific ones.
COARSE = {"X Series", "Z Series", "i", "G", "R"}

_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*")
_BMW_NUMERIC = re.compile(r"^(\d)\d{2}[A-Za-z]*$")  # 335, 530D, 428I, 630dx


def normalize_model(raw: str | None) -> str | None:
    """Canonicalise a model label. None stays None."""
    if raw is None:
        return None
    text = _PARENTHETICAL.sub("", str(raw)).strip()
    if not text:
        return None

    # BMW three-digit designations denote a series: 530D -> 5 Series.
    numeric = _BMW_NUMERIC.match(text)
    if numeric:
        return f"{numeric.group(1)} Series"

    # A bare digit is a series written without the word.
    if re.fullmatch(r"\d", text):
        return f"{text} Series"

    # Audi performance lines are written inconsistently: "rs 7" -> "RS7".
    compact = text.replace(" ", "")
    if re.fullmatch(r"(?i)(rs|s|sq)\d", compact):
        return compact.upper()

    # BMW electric line keeps its lowercase i: i3, i8.
    if re.fullmatch(r"(?i)i\d", compact):
        return "i" + compact[-1]

    # Audi/BMW alphanumerics: A4, Q7, X3, TT, R8.
    if re.fullmatch(r"(?i)[a-z]{1,2}\d", compact):
        return compact.upper()

    return text


def is_parent_of(coarse: str, specific: str) -> bool:
    """True when `coarse` is the family label of `specific` (X Series -> X5)."""
    if coarse not in COARSE or coarse == specific:
        return False
    prefix = coarse.split()[0]
    return specific.startswith(prefix)


def resolve(labels: list[str]) -> tuple[str | None, bool]:
    """Reduce one VIN's competing model labels to a single answer.

    Returns (label, ambiguous). When ambiguous is True the label is None and
    the caller must not train on -- or answer with -- this row.
    """
    distinct = {normalize_model(x) for x in labels if x is not None}
    distinct.discard(None)
    if not distinct:
        return None, False
    if len(distinct) == 1:
        return distinct.pop(), False

    # Drop any coarse label that a more specific sibling already covers.
    specific = {
        candidate
        for candidate in distinct
        if not any(is_parent_of(candidate, other) for other in distinct)
    }
    if len(specific) == 1:
        return specific.pop(), False

    # Siblings, not parent and child. Real ambiguity.
    return None, True
