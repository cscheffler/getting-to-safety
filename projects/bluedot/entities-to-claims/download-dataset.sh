#!/usr/bin/env bash

set -euo pipefail

DATASET="$1"
LOCAL_DIR="${DATASET#*/}"

hf download "$DATASET" \
  --repo-type dataset \
  --local-dir "./$LOCAL_DIR"
