# VIN Decode Gate

A VIN decoder in which every field states how it was resolved — and a case
study in how often that turned out to be less than it first looked.

Each field comes back with a verdict. `RULE` when the VIN standard settles
it, `MODEL` when a classifier is confident enough to assert it, `ESCALATE`
when there is a reading but no right to state it, `UNKNOWN` when nothing is
available. A decoder that returns `A5` and one that returns `A5 or S5, we
cannot tell` are different products, and pricing, damage matching and
insurance risk all need to know which one they just received.

The interesting part is not the classifier. It is what happened when the
project's own claims were checked:

- **Splitting on VINs inflated accuracy by four points.** The feature is the
  nine-character prefix, and 1 453 VINs collapse to 1 121 prefixes. 30.6% of
  test rows had their prefix in training and scored 98.3%, against 83.8% on
  unseen ones. Honest figure: **82.9% ± 3.4%** on model, 76.8% ± 2.8% on body.
- **The deterministic year rule now asserts nothing.** Position 10 repeats on
  a 30-year cycle, and three successive versions of that rule each claimed
  more than the VIN proves. The reading offered is right 96.5% of the time
  and still does not qualify as a `RULE`.
- **Some ambiguity is structural.** `Q5` and `SQ5` share a prefix; with the
  serial masked, nothing separates them. More data does not fix it, so the
  gate refuses those prefixes outright.

`docs/REVIEW-LOG.md` records the five review rounds that produced most of
this, including what each one found. Everything quoted here is reproducible
with `python scripts/evaluate.py --repeats 10`.

## Quick start

```bash
pip install -e ".[dev]"                 # installs the package, not just its deps
./scripts/fetch_data.sh                 # corpus is not vendored; see LICENSE
vindecode-train --data data/ml-engineer-challenge-redacted-data.csv --out artifacts
uvicorn vindecode.api:app --reload      # http://localhost:8000/docs
pytest
```

```bash
curl -s localhost:8000/decode/WAUZZZ4H1DN0XXXXX | jq
```

```json
{
  "vin": "WAUZZZ4H1DN0XXXXX",
  "make":  { "value": "Audi",  "verdict": "RULE",  "reason": "WMI WAU" },
  "model": { "value": "A8",    "verdict": "MODEL", "confidence": 0.885 },
  "year":  { "value": 2013,    "verdict": "ESCALATE", "candidates": [2013, 1983],
             "reason": "position 10 is ambiguous across the 30-year cycle and no check digit is verifiable" },
  "body":  { "value": "sedan", "verdict": "MODEL", "confidence": 0.867 }
}
```

## Approach

I started by profiling the corpus rather than by choosing a model, and the
profile decided the architecture.

**3 000 rows, 1 453 distinct VINs, two makes.** `year` is missing on 89% of
rows and `body` on 55%. So the headline problem is not accuracy on a clean
task; it is that most of the target is absent and the rest disagrees with
itself.

**Most of the missing data is not missing.** VIN position 10 encodes the
model year under ISO 3779. Applied to the rows that do carry a year label,
the rule agrees 96.5% of the time. Make is likewise fixed by the WMI —
`WAU` is Audi, `WBA` is BMW — with no learning involved. Two of the four
requested fields are therefore rule territory, and training a model on them
would replace an auditable constant with a probabilistic approximation.

**The label conflicts are three different problems.** 105 VINs carry more
than one model label, and treating them uniformly is how a decoder ends up
confidently wrong:

| Kind | Example | Answer |
|---|---|---|
| Formatting noise | `4` vs `4 Series`, `530D (EUR)`, `rs 7` | canonicalise |
| Granularity | `X5` vs `X Series` | the specific label wins |
| Genuine ambiguity | `Q5` vs `SQ5`, `A5` vs `S5` | withhold, record, escalate at inference |

The third row is the important one. An SQ5 and a Q5 share a VIN prefix, and
once the serial is masked nothing visible separates them. Majority voting
would teach the model to answer a coin flip with a straight face. 25 model
prefixes and 74 body prefixes are withheld on these grounds, counted per
field in the training report, and refused at inference rather than merely
dropped from training.

**Conflicts are resolved on the prefix.** The classifier never sees
positions 10–17, so the prefix is the unit of identity throughout: label
resolution, ambiguity accounting and the train/test split all group by it.
Contested prefixes are recorded per field — a prefix whose model is disputed
usually still has a usable body label — and the list ships inside the model
artifact, so the gate can refuse to assert a prefix the corpus itself could
not agree on, no matter how confident the classifier is about its
neighbours.

**The model handles only what rules cannot.** Model family and body type sit
in the VDS (positions 4–9), which every manufacturer defines privately.
There is no public rule to apply, so this is where learning genuinely earns
its place. Character n-grams (2–5) over positions 1–9, TF-IDF, logistic
regression. The masked serial is excluded deliberately — it carries no
information and would only invite memorisation.

**The gate is the deliverable.** A threshold on predicted probability splits
assertions from escalations.

## Results

Repeated stratified holdout, 10 repeats, mean ± standard deviation. Rows are
**prefixes, not VINs** — see the note below. Reproduce with
`python scripts/evaluate.py --repeats 10`.

| Field | Classes | Usable prefixes | Accuracy | Stated confidence | Temperature | ECE raw → calibrated |
|---|---|---|---|---|---|---|
| model | 32 | 1 051 | 82.9% ± 3.4% | 81.5% ± 1.6% | 0.76 | 0.135 → 0.052 |
| body | 11 | 865 | 76.8% ± 2.8% | 77.8% ± 2.3% | 0.75 | 0.106 → 0.061 |

Compare accuracy against stated confidence: after calibration the model
claims 81.5% and delivers 82.9%. Before calibration the gap was thirteen
points — the classifier was *under*confident, so both temperatures sit below
1 and sharpen. Accuracy is untouched by calibration, necessarily: temperature
scaling is monotone in the logits and cannot move the argmax.

### Why these numbers are lower than the obvious way of computing them

The classifier's feature is the nine-character prefix. 1 453 VINs in this
corpus collapse to 1 121 distinct prefixes, so splitting on VINs puts the
same prefix in both halves. Measured: **30.6% of test rows had their prefix
in the training set and scored 98.3%, against 83.8% on genuinely unseen
prefixes.** That is recall reported as generalisation, and it inflated the
headline by roughly four points.

Everything here therefore groups by prefix before splitting. The same
mistake is why label conflicts are resolved on the prefix rather than the
VIN: two VINs sharing a prefix are one example, and resolving them
separately leaves the contradiction intact under two row identities.

### What the gate buys

| Field | Threshold | Traffic asserted | Accuracy when asserted | Accuracy when escalated |
|---|---|---|---|---|
| model | ungated | 100% | 82.9% | — |
| model | 0.70 | 77.1% | 92.7% | 49.6% |
| model | **0.80** | **68.3%** | **93.6%** | **59.7%** |
| model | 0.90 | 54.1% | 95.6% | 67.9% |
| body | ungated | 100% | 76.8% | — |
| body | **0.70** | **69.0%** | **90.7%** | **45.6%** |
| body | 0.80 | 58.9% | 93.1% | 53.4% |

Read the escalated column first. That bucket sits near a coin flip at every
threshold, which is the point: the gate isolates the cases where the model
does not know. Surrendering three tenths of model traffic to review moves the
asserted answers from 83% to 94%.

The two fields need different cutoffs, and the repeats are what showed it.
Body's probability mass sits lower — 11 classes with overlapping evidence —
so defaults are per field: `model` 0.80, `body` 0.70. Override with
`VINDECODE_THRESHOLDS="model=0.85,body=0.65"`.

### The year rule asserts nothing unless you tell it the rule applies

VIN position 10 repeats on a 30-year cycle: `E` is 2014 or 1984. Three
separate questions stand between a character and a fact, and collapsing any
two of them produces a confident wrong answer — which this project did twice
before getting here:

1. **Is the character a year code at all?** If not, `UNKNOWN`.
2. **Does the model-year rule apply to this VIN?** Position 10 is a US/NHTSA
   requirement scoped by market, vehicle class and weight. A valid ISO 7064
   check digit shows arithmetic consistency and nothing more — an arbitrary
   VIN satisfies it roughly one time in eleven, and `ZZZ3N3C51EK200000` does.
   So applicability is a caller-supplied policy with **no default**: unset,
   no year is ever asserted.
3. **Which cycle?** Position 7 answers this where the rule applies. The cycle
   is chosen *first* and the plausibility window checked afterwards. Filtering
   candidates before choosing let an out-of-range modern year fall through to
   the older one: `5UXFC5CAXYK200000` returned 2000 as a `RULE`, when position 7
   actually indicates 2030 and the VIN is simply inconsistent. It now returns
   the contradiction.

One policy ships — `north_american_passenger_car`, requiring an ISO 3779
North American WMI plus a valid check digit — and it is opt-in via
`VINDECODE_YEAR_POLICY`, because it still assumes the vehicle class the VIN
cannot state. An unrecognised name there aborts startup rather than falling
back to "no policy": a typo that leaves the service healthy while silently
escalating every year it was configured to assert is the kind of config
error that surfaces weeks later as an unexplained review backlog.

On this corpus the serial is redacted, so no check digit is computable and no
policy would fire anyway. The rules layer therefore asserts **no years at
all** here: 93.9% of labelled rows get an `ESCALATE` reading, 6.1%
`UNKNOWN`. The offered reading is right 96.5% of the time and still does not
get to be a `RULE`.

## Limitations

- **The shipped applicability policy still assumes something.** It
  establishes market from the WMI and consistency from the check digit, but
  the NHTSA rule is scoped by vehicle class and weight, which no VIN states.
  A registry mapping WMI to vehicle class would close that gap; until then
  the policy is opt-in and documented rather than default. A per-manufacturer
  first-production-year table would collapse most of the remaining cycle
  ambiguity — a 1984 BMW i3 is not a possibility — and belongs in the same
  rules layer.
- **Two makes, 3 000 rows.** Audi and BMW only. The WMI register in
  `deterministic.py` covers what the corpus contains and nothing else; a
  third make returns `UNKNOWN` for make until its WMIs are registered. This
  is a data-coverage limit, not a model limit, and it fails loudly.
- **Year coverage is partial.** Position 10 is a US/NHTSA requirement.
  European-market VINs may carry an unrelated character there, so the rule
  resolves 57.6% of the corpus outright and declares the rest unknown rather
  than inventing a year. Manufacturer-specific year encodings would lift
  this, and belong in the rules layer, not the model.
- **Trim ambiguity is structural.** `Q5`/`SQ5` cannot be separated from a
  masked VIN by any method. Unmasked serials, or a manufacturer build-data
  feed, would resolve it; more training data will not.
- **Calibration is global, not per class.** One temperature is shared across
  all classes. Per-class Platt scaling would be better in principle and is
  not fittable here: several classes have two or three examples, and
  scikit-learn's `CalibratedClassifierCV` raises rather than fit them. The
  shared scalar is the method that survives the class sparsity, not the
  method I would choose with ten times the data.
- **Repeated holdout, not k-fold.** Stratified k-fold at k=5 needs five
  examples of every class and this corpus has model classes with two.
  Dropping them to enable k-fold would improve every metric by deleting the
  hardest cases. Repeated holdout keeps them at the cost of overlapping test
  sets across repeats, so the standard deviations are mildly optimistic.
  Classes appearing once are excluded from training entirely.
- **Accuracy is unweighted.** A wrong `7 Series` and a wrong `1 Series` cost
  the same here. In production they do not.

## Where I would take it next

1. **Revisit calibration as data grows.** Per-field thresholds are in;
   per-class calibration is not, because the class sparsity forbids it. Once
   the thin classes carry enough examples, per-class Platt scaling should
   replace the shared temperature.
2. **Version the rules layer.** The WMI register and year table are data, not
   code; they should be a versioned artifact with their own review path so a
   register update does not need a deploy.
3. **Alert on the verdict mix, not on accuracy.** Ground truth arrives late
   or never. A rising `ESCALATE` share is an immediate, label-free signal that
   the input distribution has moved — the batch endpoint already returns the
   counts for this reason.
4. **Feed escalations back.** They are, by construction, the highest-value
   labelling queue available.
5. **Cache on the prefix.** Positions 1–9 are the entire feature; identical
   prefixes are identical decodes, which makes this trivially cacheable and
   is what carries it to high request volumes.

`docs/ARCHITECTURE.md` sketches the production shape.

## Layout

```
src/vindecode/
  deterministic.py   VIN parsing, WMI register, year rule
  labels.py          canonicalisation and conflict resolution
  model.py           training pipeline and predictor
  calibration.py     temperature scaling, ECE, reliability table
  gate.py            verdict assignment
  api.py             FastAPI service
  cli.py             training entry point
scripts/             corpus fetch, repeated evaluation
tests/               58 tests
```

## Licence

MIT, mikko-lab (WP Saavutettavuus) — see `LICENSE`. Covers the source only:
the training corpus belongs to carVertical and is fetched, not vendored.
