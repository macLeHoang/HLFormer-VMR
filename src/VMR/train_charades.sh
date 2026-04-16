#!/usr/bin/env bash
# Train HLFormer-VMR on Charades-STA
#
# Usage:
#   bash VMR/train_charades.sh                        # train on GPU 0
#   bash VMR/train_charades.sh --gpu 1                # train on GPU 1
#   bash VMR/train_charades.sh --eval --resume <ckpt> # eval only
#
# Run from: ICCV25-HLFormer/src/
# --------------------------------------------------------------------------

set -e

# ---------- defaults -------------------------------------------------------
GPU="0"
RESUME=""
EVAL=""

# ---------- parse args -----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU="$2";    shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --eval)   EVAL="--eval"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------- locate src/ directory ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"   # src/ is one level above VMR/
cd "$SRC_DIR"
echo "Working directory: $(pwd)"

# ---------- build command --------------------------------------------------
CMD="python VMR/main_vmr.py -d charades --gpu ${GPU}"

if [[ -n "$RESUME" ]]; then
    CMD="${CMD} --resume ${RESUME}"
fi

if [[ -n "$EVAL" ]]; then
    CMD="${CMD} --eval"
fi

# ---------- run ------------------------------------------------------------
echo "Running: ${CMD}"
echo "=========================================="
eval "$CMD"
