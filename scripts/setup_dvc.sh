#!/usr/bin/env bash
# One-time DVC setup: init, track dataset + models, local remote, push.
# Run from the repo root (Git Bash / WSL / macOS / Linux) AFTER:
#   git init && python src/run_experiments.py
set -euo pipefail

dvc init
dvc add data/breast_cancer.csv models           # creates *.dvc files + .gitignore entries
dvc remote add -d localstorage ../dvc_storage   # local folder next to the repo
dvc push

git add .dvc .dvcignore data/.gitignore data/breast_cancer.csv.dvc models.dvc .gitignore
git commit -m "Track dataset and models with DVC"
echo "DVC setup done. Remote:" && dvc remote list
