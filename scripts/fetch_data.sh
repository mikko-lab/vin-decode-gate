#!/usr/bin/env bash
# Fetch the training corpus from its upstream repository.
# The data is carVertical's and is not redistributed with this project.
set -euo pipefail

URL="https://raw.githubusercontent.com/carVertical/ml-engineering-homework/master/data/ml-engineer-challenge-redacted-data.csv"
DEST="$(dirname "$0")/../data/ml-engineer-challenge-redacted-data.csv"

curl -fsSL "$URL" -o "$DEST"
echo "Wrote $DEST ($(wc -l < "$DEST") lines)"
