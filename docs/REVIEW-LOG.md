# Review log

This project was reviewed five times before publication. Twelve findings came
out of it, and most of them were not style issues — three were wrong answers
being served with authority, and one showed that the accuracy figure in the
README was inflated.

The log is here because the corrections are the most useful thing in the
repository. A decoder that argues for declaring uncertainty is not credible
if it hides its own.

## Attribution

**The reviews were produced by a large language model**, not by human
reviewers, working from the packaged source and running it where it could.
Findings were verified independently before being accepted: each one was
reproduced against the real corpus or a constructed VIN, and every fix landed
with a regression test. The evidence is in `tests/test_decoding.py`, so the
claims below can be checked without trusting either the reviewer or me.

The reviews ran on Windows. The project was developed on Linux. Round five
found a bug that only that difference could surface.

## Round 1 — five findings

| Finding | Outcome |
|---|---|
| Model year returned as `RULE` without resolving the 30-year cycle | Valid. Rewritten across rounds 1, 2 and 3 |
| Conflicts resolved per VIN while the model's key is the prefix | Valid, and worse than reported — see below |
| Two-class `decision_function` breaks calibration | Valid latent bug; `as_logits` added |
| Ambiguous model label discarded the row's usable body label | Valid; withholding is now per field |
| Quick start never made the package importable | Valid; `pyproject.toml` added |

The second finding prompted a measurement I had not thought to make.
Splitting on VINs put the same prefix in both halves: 30.6% of test rows had
their prefix in training and scored 98.3%, against 83.8% on genuinely unseen
prefixes. Reported accuracy fell from 85.9% to 82.9% once grouping moved to
the prefix. That is recall reported as generalisation, and it was mine.

The fourth finding was also understated. Counting withheld rows per field
showed body losing 74 prefixes, not the one the reviewer's small test showed.

## Round 2 — four findings

| Finding | Outcome |
|---|---|
| Applicability of the year rule still not established | Valid; a single plausible candidate proves nothing about whether position 10 encodes a year |
| Training grouped raw VINs while the decoder upper-cased them | Valid; a lower-case VIN produced an ambiguity key the gate could never find |
| Ten-character cache key insufficient for the whole response | Valid; response key is now the full VIN plus rule and model versions |
| README example showed `MODEL` below the threshold; corpus counts stale | Valid; both regenerated from real output |

## Round 3 — two findings

| Finding | Outcome |
|---|---|
| A valid check digit was treated as proof the rule applies | Valid; it is arithmetic consistency, satisfied by chance about one time in eleven |
| Candidates filtered before the cycle was chosen | Valid, and a genuine logic error |

The second was the worst defect found in the project. With an out-of-range
modern year filtered away, `max()` selected the surviving older-cycle value
and returned it as a fact: `5UXFC5CAXYK200000` yielded 2000 as a `RULE` when
position 7 indicates 2030 and the VIN is simply inconsistent. Cycle selection
now precedes the plausibility check, and an impossible year returns a
contradiction rather than the other cycle.

Applicability became a caller-supplied policy with no default. On this
corpus, where the serial is redacted and no check digit can be computed, the
rules layer asserts no years at all.

## Round 4 — one finding, three cleanups

| Finding | Outcome |
|---|---|
| A misspelled `VINDECODE_YEAR_POLICY` was ignored while `/health` reported it as active | Valid; unknown names now abort startup |
| `YearReading` and `has_valid_check_digit` defined twice | Valid; my own error while rewriting the module |
| "No policy supplied" and "policy declined this VIN" gave the same reason | Valid; separated |
| Regression test depended on the wall clock | Valid; the date is now injectable and pinned |

The configuration finding was the most operationally serious of the twelve. A
typo would have left the service running and reporting healthy while
escalating every year it was configured to assert — visible weeks later only
as an unexplained review backlog.

## Round 5 — one finding

| Finding | Outcome |
|---|---|
| Subprocess tests built a minimal environment and dropped `SystemRoot` | Valid; the environment is now inherited and overridden |

The three configuration tests added in round 4 failed on Windows before
reaching the code under test. This one could not have been found on the
development machine — "works here" was doing the arguing.

## What the pattern was

Every version of the year rule was directionally right and one step short.
First the year was a fact. Then the cycle was shown but uniqueness was
treated as proof. Then the check digit was required but mistaken for
evidence. The same error three times: *I cannot see an alternative* is not
*there is no alternative*.

That is the exact confusion this project exists to prevent, committed
repeatedly inside the code meant to prevent it. Which is a reasonable
argument for why the gate belongs in the design rather than in the
documentation.
