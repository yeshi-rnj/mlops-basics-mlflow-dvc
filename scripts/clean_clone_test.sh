#!/usr/bin/env bash
# Prove reproducibility from a clean clone (commit everything first!).
# Works with the local DVC remote because the clone is a sibling folder.
set -euo pipefail

CLONE=../mlops-clean-clone
rm -rf "$CLONE"
git clone . "$CLONE"
cd "$CLONE"

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -q -r requirements.txt

dvc pull
python src/reproduce.py
