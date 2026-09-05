"""Command-line entry point for training."""

from __future__ import annotations

import argparse

from .model import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vindecode-train", description="Train VIN decoding models.")
    parser.add_argument("--data", default="data/ml-engineer-challenge-redacted-data.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)

    for report in train(args.data, args.out):
        print(
            f"{report.field:6} classes={report.n_classes:3} "
            f"train={report.n_train:4} test={report.n_test:3} "
            f"accuracy={report.accuracy:.3f} T={report.temperature:.3f} ECE {report.ece_raw:.3f}->{report.ece_calibrated:.3f}"
            + (f" withheld_ambiguous={report.withheld_ambiguous}" if report.withheld_ambiguous else "")
        )
    print(f"\nArtifacts written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
