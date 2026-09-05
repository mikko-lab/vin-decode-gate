import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from vindecode.deterministic import (
    InvalidVin,
    decode_make,
    decode_year,
    has_valid_check_digit,
    north_american_passenger_car,
    parse,
)
from vindecode.gate import Verdict
from vindecode.labels import normalize_model, resolve


class TestVinParsing:
    def test_splits_into_iso_3779_sections(self):
        parts = parse("WBA3N3C50EK2XXXXX")
        assert (parts.wmi, parts.vds, parts.vis) == ("WBA", "3N3C50", "EK2XXXXX")

    def test_lowercase_input_is_accepted(self):
        assert parse("wba3n3c50ek2xxxxx").wmi == "WBA"

    @pytest.mark.parametrize(
        "bad, why",
        [
            ("WBA3N3C50EK2XXX", "too short"),
            ("WBA3N3C50EK2XXXXXX", "too long"),
            ("WBA3N3C50EK2XXXXI", "letter I is excluded by ISO 3779"),
            ("WBA3N3C50EK2-XXXX", "punctuation"),
        ],
    )
    def test_rejects_impossible_input(self, bad, why):
        with pytest.raises(InvalidVin):
            parse(bad)


class TestDeterministicRules:
    def test_make_comes_from_the_wmi_register(self):
        assert decode_make(parse("WAUZZZ4H1DN0XXXXX")) == "Audi"
        assert decode_make(parse("WBA3N3C50EK2XXXXX")) == "BMW"

    def test_unregistered_wmi_returns_none_rather_than_a_guess(self):
        assert decode_make(parse("ZZZ3N3C50EK2XXXXX")) is None

    def test_single_candidate_is_still_not_asserted(self):
        # 2037 is not plausible, so 2007 is the only reading -- but nothing
        # shows position 10 encodes a year on this European VIN at all.
        reading = decode_year(parse("WAUZZZ8P27A1XXXXX"), "WAUZZZ8P27A1XXXXX")
        assert reading.candidates == (2007,)
        assert reading.value == 2007
        assert reading.resolved is False

    def test_check_digit_alone_does_not_license_an_assertion(self):
        # Arithmetic consistency is not evidence about market or vehicle
        # class. A real North American VIN (2003 Honda Accord) still gets no
        # assertion without an applicability policy.
        vin = "1HGCM82633A004352"
        assert has_valid_check_digit(vin)
        assert decode_year(parse(vin), vin).resolved is False

    def test_an_explicit_policy_is_what_licenses_an_assertion(self):
        vin = "1HGCM82633A004352"
        reading = decode_year(parse(vin), vin, north_american_passenger_car)
        assert reading.value == 2003 and reading.resolved

    def test_policy_rejects_a_checksum_coincidence_outside_north_america(self):
        # Passes the check digit, but the WMI is not North American.
        vin = "ZZZ3N3C51EK200000"
        assert has_valid_check_digit(vin)
        assert decode_year(parse(vin), vin, north_american_passenger_car).resolved is False

    def test_impossible_cycle_never_falls_back_to_the_other_one(self):
        # Position 7 is alphabetic, so the modern cycle applies and code Y
        # means 2030 -- not a possible model year. Filtering the candidates
        # before choosing the cycle would have returned 2000 as a fact.
        import datetime

        vin = "5UXFC5CAXYK200000"
        assert has_valid_check_digit(vin)
        reading = decode_year(
            parse(vin), vin, north_american_passenger_car, today=datetime.date(2026, 9, 5)
        )
        assert reading.value is None
        assert reading.resolved is False
        assert "inconsistent" in reading.basis

    def test_letter_code_stays_ambiguous_across_the_thirty_year_cycle(self):
        # E is 2014 or 1984. Without a valid check digit nothing in the VIN
        # separates them, so the reading is offered, not asserted.
        reading = decode_year(parse("WBA3N3C50EK2XXXXX"), "WBA3N3C50EK2XXXXX")
        assert reading.resolved is False
        assert reading.value == 2014
        assert set(reading.candidates) == {2014, 1984}

    def test_non_year_character_yields_no_value(self):
        # European-market VINs may carry an unrelated character at position 10.
        reading = decode_year(parse("WBY2Z21040V3XXXXX"), "WBY2Z21040V3XXXXX")
        assert reading.value is None and reading.candidates == ()

    def test_redacted_serial_cannot_validate_a_check_digit(self):
        assert has_valid_check_digit("WBA3N3C50EK2XXXXX") is False


class TestLabelHygiene:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("4", "4 Series"),
            ("530D (EUR)", "5 Series"),
            ("428I (USA)", "4 Series"),
            ("630dx (630dx)", "6 Series"),
            ("rs 7", "RS7"),
            ("i3", "i3"),
            ("SQ5", "SQ5"),
        ],
    )
    def test_formatting_noise_is_canonicalised(self, raw, expected):
        assert normalize_model(raw) == expected

    def test_specific_label_beats_its_coarse_parent(self):
        assert resolve(["X5", "X Series"]) == ("X5", False)

    def test_siblings_are_ambiguous_not_majority_voted(self):
        # Q5 and SQ5 are different cars; the visible VIN cannot separate them.
        assert resolve(["Q5", "SQ5", "Q5"]) == (None, True)

    def test_equivalent_spellings_collapse_to_one_answer(self):
        assert resolve(["4", "4 Series"]) == ("4 Series", False)


class TestGate:
    def test_ambiguous_year_is_escalated_with_both_candidates(self, decoder):
        year = decoder.decode("WBA3N3C50EK2XXXXX").year
        assert year.verdict is Verdict.ESCALATE
        assert year.candidates == (2014, 1984)

    def test_rule_fields_carry_no_confidence_score(self, decoder):
        result = decoder.decode("WBA3N3C50EK2XXXXX")
        assert result.make.verdict is Verdict.RULE
        assert result.make.value == "BMW"
        assert result.make.confidence is None

    def test_learned_fields_are_labelled_model_or_escalate(self, decoder):
        result = decoder.decode("WAUZZZ4H1DN0XXXXX")
        assert result.model.verdict in {Verdict.MODEL, Verdict.ESCALATE}
        assert 0.0 <= result.model.confidence <= 1.0

    def test_low_confidence_escalates_instead_of_asserting(self, predictor):
        from vindecode.gate import Decoder

        strict = Decoder(predictor, threshold=1.01)  # nothing can clear this
        assert strict.decode("WBA3N3C50EK2XXXXX").model.verdict is Verdict.ESCALATE

    def test_unknown_make_is_declared_not_inferred(self, decoder):
        assert decoder.decode("ZZZ3N3C50EK2XXXXX").make.verdict is Verdict.UNKNOWN

    def test_malformed_vin_raises_rather_than_returning_a_partial_answer(self, decoder):
        with pytest.raises(InvalidVin):
            decoder.decode("NOT-A-VIN")


class TestCalibration:
    def test_temperature_scaling_cannot_change_the_predicted_label(self):
        # Monotone in the logits, so argmax is invariant. This is why
        # calibration is separable from modelling.
        import numpy as np

        from vindecode.calibration import apply_temperature

        logits = np.array([[2.0, 0.5, -1.0, 0.1]])
        cold = apply_temperature(logits, 0.5).argmax(axis=1)
        hot = apply_temperature(logits, 2.5).argmax(axis=1)
        assert cold == hot == logits.argmax(axis=1)

    def test_temperature_below_one_sharpens_confidence(self):
        import numpy as np

        from vindecode.calibration import apply_temperature

        logits = np.array([[2.0, 0.5, -1.0]])
        assert apply_temperature(logits, 0.5).max() > apply_temperature(logits, 2.0).max()

    def test_perfect_confidence_scores_zero_calibration_error(self):
        import numpy as np

        from vindecode.calibration import expected_calibration_error

        confidence = np.array([0.95, 0.95, 0.05, 0.05])
        correct = np.array([1.0, 1.0, 0.0, 0.0])
        assert expected_calibration_error(confidence, correct) < 0.06

    def test_systematic_overconfidence_is_detected(self):
        import numpy as np

        from vindecode.calibration import expected_calibration_error

        confidence = np.full(100, 0.95)
        correct = np.array([1.0] * 50 + [0.0] * 50)  # claims 95%, delivers 50%
        assert expected_calibration_error(confidence, correct) > 0.4

    def test_fitted_temperature_is_persisted_with_the_pipeline(self, predictor):
        temperatures = predictor.temperatures
        assert set(temperatures) == {"model", "body"}
        assert all(0.1 < t < 10.0 for t in temperatures.values())

    def test_predicted_probability_stays_a_probability(self, predictor):
        _, probability = predictor.predict("model", "WAUZZZ4H1")
        assert 0.0 <= probability <= 1.0


class TestPerFieldThresholds:
    def test_fields_carry_their_own_cutoffs(self, decoder):
        # Body's probability mass sits lower than model's; a shared cutoff
        # would escalate body decodes that are in fact reliable.
        assert decoder.threshold_for("model") > decoder.threshold_for("body")

    def test_partial_override_leaves_other_fields_at_default(self, predictor):
        from vindecode.gate import DEFAULT_THRESHOLDS, Decoder

        custom = Decoder(predictor, thresholds={"body": 0.55})
        assert custom.threshold_for("body") == 0.55
        assert custom.threshold_for("model") == DEFAULT_THRESHOLDS["model"]

    def test_single_threshold_applies_to_every_field(self, predictor):
        from vindecode.gate import Decoder

        uniform = Decoder(predictor, threshold=0.75)
        assert uniform.threshold_for("model") == uniform.threshold_for("body") == 0.75

    def test_escalation_reason_names_the_field_and_its_cutoff(self, predictor):
        from vindecode.gate import Decoder

        strict = Decoder(predictor, thresholds={"model": 1.01})
        field = strict.decode("WBA3N3C50EK2XXXXX").model
        assert "model" in field.reason and "1.01" in field.reason


class TestPrefixLevelHygiene:
    """Regressions for the review findings: the feature key is the prefix."""

    def test_conflicts_are_resolved_on_the_prefix_not_the_vin(self, tmp_path):
        # Two VINs, one prefix, incompatible models. Resolving per VIN would
        # emit two confident and contradictory training rows.
        from vindecode.model import build_corpus

        csv = tmp_path / "corpus.csv"
        csv.write_text(
            "vin,make,model,year,body\n"
            "WAUZZZ8R0AA1XXXXX,Audi,Q5,,SUV\n"
            "WAUZZZ8R0BB2XXXXX,Audi,SQ5,,SUV\n"
        )
        corpus = build_corpus(csv)
        assert len(corpus.frame) == 1
        assert corpus.frame.iloc[0]["model"] is None
        assert corpus.ambiguous["model"]["WAUZZZ8R0"] == ("Q5", "SQ5")

    def test_ambiguity_in_one_field_does_not_discard_the_other(self, tmp_path):
        # Q5 and SQ5 disagree on model but agree on body; the body label is
        # perfectly usable and must survive.
        from vindecode.model import build_corpus

        csv = tmp_path / "corpus.csv"
        csv.write_text(
            "vin,make,model,year,body\n"
            "WAUZZZ8R0AA1XXXXX,Audi,Q5,,SUV\n"
            "WAUZZZ8R0BB2XXXXX,Audi,SQ5,,SUV\n"
        )
        corpus = build_corpus(csv)
        assert corpus.frame.iloc[0]["body"] == "SUV"
        assert corpus.withheld == {"model": 1, "body": 0}

    def test_empty_result_still_has_columns(self, tmp_path):
        from vindecode.model import build_corpus

        csv = tmp_path / "empty.csv"
        csv.write_text("vin,make,model,year,body\n")
        assert list(build_corpus(csv).frame.columns) == ["prefix", "model", "body"]

    def test_contested_prefix_is_escalated_regardless_of_confidence(self, predictor):
        from vindecode.gate import Decoder

        contested = next(iter(predictor._artifacts["model"]["ambiguous_prefixes"]))
        permissive = Decoder(predictor, threshold=0.0)  # nothing may pass on merit
        field = permissive.decode(contested + "AA1XXXXX").model
        assert field.verdict is Verdict.ESCALATE
        assert field.candidates


class TestBinaryClassSupport:
    """Regression for the two-class decision_function shape."""

    def test_binary_scores_expand_to_a_logit_pair(self):
        import numpy as np

        from vindecode.calibration import as_logits

        expanded = as_logits(np.array([0.7, -1.2]))
        assert expanded.shape == (2, 2)
        assert (expanded[:, 0] == 0).all()

    def test_binary_probabilities_are_not_all_one(self):
        import numpy as np

        from vindecode.calibration import apply_temperature

        probabilities = apply_temperature(np.array([0.7, -1.2]), 1.0)
        assert probabilities.shape == (2, 2)
        assert probabilities.max() < 1.0

    def test_two_class_training_round_trips(self, tmp_path):
        import pandas as pd

        from vindecode.model import Predictor, train

        rows = []
        for i in range(20):
            rows.append({"vin": f"WAUZZZ4H{i:01d}DN0XXXXX"[:17], "make": "Audi",
                         "model": "A4", "year": None, "body": "sedan"})
            rows.append({"vin": f"WBA3N3C5{i:01d}EK2XXXXX"[:17], "make": "BMW",
                         "model": "X5", "year": None, "body": "SUV"})
        csv = tmp_path / "binary.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)

        train(csv, tmp_path / "artifacts")
        _, probability = Predictor(tmp_path / "artifacts").predict("model", "WAUZZZ4H0")
        assert 0.0 < probability < 1.0


class TestVinNormalisation:
    """Regression: training and decoding must agree on the form of a VIN."""

    def test_case_does_not_split_a_prefix_group(self, tmp_path):
        # Training grouped raw strings while the decoder upper-cases its
        # input, so a lower-case VIN produced a second group and an ambiguity
        # key the gate could never look up.
        from vindecode.model import build_corpus

        csv = tmp_path / "mixed.csv"
        csv.write_text(
            "vin,make,model,year,body\n"
            "WAUZZZ8R0AA1XXXXX,Audi,Q5,,SUV\n"
            "wauzzz8r0bb2xxxxx,Audi,SQ5,,SUV\n"
        )
        corpus = build_corpus(csv)
        assert len(corpus.frame) == 1
        assert corpus.ambiguous["model"] == {"WAUZZZ8R0": ("Q5", "SQ5")}

    def test_ambiguity_key_is_found_whatever_case_the_query_uses(self, tmp_path):
        import pandas as pd

        from vindecode.gate import Decoder, Verdict
        from vindecode.model import Predictor, train

        # Distinct prefixes per class: the corpus is keyed on prefixes, so
        # repeating one would give the classifier a single example.
        rows = [
            {"vin": "WAUZZZ8R0AA1XXXXX", "make": "Audi", "model": "Q5",
             "year": None, "body": "SUV"},
            {"vin": "wauzzz8r0bb2xxxxx", "make": "Audi", "model": "SQ5",
             "year": None, "body": "SUV"},
        ]
        for i in range(6):
            rows.append({"vin": f"WBA3N3C5{i}EK2XXXXX", "make": "BMW",
                         "model": "X5", "year": None, "body": "SUV"})
            rows.append({"vin": f"WAUZZZ4H{i}DN0XXXXX", "make": "Audi",
                         "model": "A4", "year": None, "body": "sedan"})
        csv = tmp_path / "mixed.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        train(csv, tmp_path / "artifacts")

        decoder = Decoder(Predictor(tmp_path / "artifacts"), threshold=0.0)
        for query in ("WAUZZZ8R0AA1XXXXX", "wauzzz8r0aa1xxxxx"):
            assert decoder.decode(query).model.verdict is Verdict.ESCALATE

    def test_malformed_rows_are_dropped_from_the_corpus(self, tmp_path):
        from vindecode.model import build_corpus

        csv = tmp_path / "junk.csv"
        csv.write_text(
            "vin,make,model,year,body\n"
            "WAUZZZ8R0AA1XXXXX,Audi,Q5,,SUV\n"
            "TOOSHORT,Audi,Q5,,SUV\n"
            "WAUZZZ8R0AA1XXXXI,Audi,Q5,,SUV\n"  # I is not in the VIN alphabet
        )
        assert len(build_corpus(csv).frame) == 1


class TestYearPolicyConfiguration:
    """The applicability policy must be unambiguous at startup."""

    def _health_and_verdict(self, tmp_path, value):
        import subprocess
        import sys

        script = (
            "from fastapi.testclient import TestClient\n"
            "from vindecode.api import app\n"
            "c = TestClient(app)\n"
            "print(c.get('/health').json()['year_rule_policy'])\n"
            "print(c.get('/decode/1HGCM82633A004352').json()['year']['verdict'])\n"
        )
        # Copy the real environment and override only what the test controls.
        # A hand-built env omits platform essentials -- on Windows, dropping
        # SystemRoot breaks asyncio's socket setup and the subprocess dies on
        # import, long before it reaches the configuration under test.
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(ROOT / "src"),
                "VINDECODE_ARTIFACTS": str(ROOT / "artifacts"),
                "VINDECODE_YEAR_POLICY": value,
            }
        )
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )

    def test_unset_policy_starts_and_asserts_nothing(self, tmp_path):
        result = self._health_and_verdict(tmp_path, "")
        assert result.returncode == 0
        assert result.stdout.split() == ["None", "ESCALATE"]

    def test_known_policy_is_applied_and_reported(self, tmp_path):
        result = self._health_and_verdict(tmp_path, "north-american-passenger-car")
        assert result.returncode == 0
        assert result.stdout.split() == ["north-american-passenger-car", "RULE"]

    def test_misspelled_policy_refuses_to_start(self, tmp_path):
        # Previously this started cleanly, reported the typo as active, and
        # silently escalated every year it was configured to assert.
        result = self._health_and_verdict(tmp_path, "north-american-passenger-carr")
        assert result.returncode != 0
        assert "not a known applicability policy" in result.stderr


class TestYearReasonsAndClock:
    def test_missing_policy_and_declining_policy_read_differently(self):
        vin = "ZZZ3N3C51EK200000"  # passes the check digit, not a NA WMI
        parts = parse(vin)
        assert "no applicability policy supplied" in decode_year(parts, vin).basis
        assert (
            "does not cover this VIN"
            in decode_year(parts, vin, north_american_passenger_car).basis
        )

    def test_impossible_cycle_test_does_not_depend_on_the_wall_clock(self):
        # Pinned: without this the 2030 expectation quietly expires in 2029.
        import datetime

        reading = decode_year(
            parse("5UXFC5CAXYK200000"),
            "5UXFC5CAXYK200000",
            north_american_passenger_car,
            today=datetime.date(2026, 9, 5),
        )
        assert reading.value is None and "2030" in reading.basis

    def test_the_same_vin_resolves_once_that_year_is_in_range(self):
        import datetime

        reading = decode_year(
            parse("5UXFC5CAXYK200000"),
            "5UXFC5CAXYK200000",
            north_american_passenger_car,
            today=datetime.date(2035, 1, 1),
        )
        assert reading.value == 2030 and reading.resolved
