#!/usr/bin/env bash
# Generates RLMRec profiles (vLLM, .venv-vllm) and then encodes them (ghc2f venv), per dataset.
# Resumable: generate_rlmrec_profiles.py skips finished chunks.
#
#   mkdir -p logs
#   nohup bash run_rlmrec_profiles.bash all_beauty > logs/rlmrec_profiles_all_beauty.log 2>&1 &
#   tail -f logs/rlmrec_profiles_all_beauty.log
#
# Arguments are DATASET NAMES only (no stage names such as "profiles").
set -euo pipefail
cd "$(dirname "$0")"

KNOWN="all_beauty imdb musical_instruments tripadvisor"
if [ $# -eq 0 ]; then
    echo "usage: $0 <dataset> [dataset ...]   (one of: $KNOWN)" >&2
    exit 2
fi
for ds in "$@"; do
    case " $KNOWN " in
        *" $ds "*) ;;
        *) echo "unknown dataset '$ds' (one of: $KNOWN)" >&2; exit 2 ;;
    esac
done

VLLM_PY=$PWD/.venv-vllm/bin/python
GHC2F=${GHC2F:-$HOME/ghc2f}
EMB_PY=$GHC2F/.venv/bin/python
for py in "$VLLM_PY" "$EMB_PY"; do
    [ -x "$py" ] || { echo "python not found: $py" >&2; exit 1; }
done

# Only one generation at a time on the GPU (two vLLM instances = OOM / clobbered chunks)
exec 9>/tmp/rlmrec_profiles.lock
flock -n 9 || { echo "another run_rlmrec_profiles.bash is already running" >&2; exit 1; }

export PATH="$PWD/.venv-vllm/bin:$PATH"         # ninja for vLLM's compiled kernels
export VLLM_USE_FLASHINFER_SAMPLER=0             # its kernels don't build with the system nvcc
export RLMREC_QUANT=${RLMREC_QUANT:-fp8}         # ~2.9x faster than bf16 on the L40S
export RLMREC_GPU_UTIL=${RLMREC_GPU_UTIL:-0.85}
export PYTHONUNBUFFERED=1                        # prints reach the nohup log immediately

for ds in "$@"; do
    export IREV_SPLITS_DIR=$GHC2F/data/$ds/splits
    if [ ! -f "$IREV_SPLITS_DIR/train.parquet" ]; then
        echo "missing $IREV_SPLITS_DIR/train.parquet" >&2
        exit 1
    fi

    echo "== $ds: profiles from $IREV_SPLITS_DIR ($(date))"
    "$VLLM_PY" preprocess_data/generate_rlmrec_profiles.py "$ds"

    if [ -f "dataset/${ds}_data/user_ids.npy" ]; then
        echo "== $ds: embeddings ($(date))"
        "$EMB_PY" preprocess_data/build_rlmrec_fixed_split.py "$ds"
    else
        echo "== $ds: dataset/${ds}_data/ not built yet; run build_rlmrec_fixed_split.py after it is"
    fi
done
echo "== done ($(date))"