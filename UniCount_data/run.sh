#!/bin/bash
set -euo pipefail

# Default behavior for quick dataset prep.
# Override with CLI args, for example:
#   bash run.sh --max_samples 500 --output_dir count_GT
if [[ "$#" -gt 0 ]]; then
	python extract_count.py "$@"
else
	python extract_count.py --max_samples 10 --output_dir count_GT
fi
