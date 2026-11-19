#!/usr/bin/env bash
# End-to-end smoke test: tiny random models + synthetic data, every stage of the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
C=configs/smoke.yaml
python scripts/make_smoke_assets.py
python scripts/prepare_data.py --config $C
DETS=$(python -c "import sys; sys.path.insert(0,'src'); from pidetect.utils import load_config, enabled_detectors; print(' '.join(enabled_detectors(load_config('$C'))))")
for D in $DETS; do
  python scripts/run_detector.py --config $C --detector $D
done
python scripts/run_detector.py --config $C --detector distilbert --seed 42   # second seed -> mean±std path
DEV=cpu; python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && DEV=cuda
for D in $DETS; do
  python scripts/benchmark_latency.py --config $C --detector $D --device $DEV
done
[ "$DEV" = cuda ] && for D in $DETS; do CUDA_VISIBLE_DEVICES="" python scripts/benchmark_latency.py --config $C --detector $D --device cpu; done
python scripts/aggregate.py --config $C --paper-dir smoke/paper
cp paper/main.tex paper/references.bib smoke/paper/
python scripts/check_paper_macros.py smoke/paper || echo "(missing numbers above are expected in the smoke run: no SPML/jailbreak sets, GPU latency only on GPU nodes)"
if command -v latexmk >/dev/null 2>&1; then (cd smoke/paper && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 && echo "smoke paper -> smoke/paper/main.pdf") || true; fi
echo "SMOKE TEST PASSED -> smoke/results, smoke/paper"
