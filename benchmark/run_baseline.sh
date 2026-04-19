#!/bin/bash
# Runs the VMAF CUDA baseline matrix: {RTX 5070, RTX 4070} × {480p, 1080p, 2160p}.
# For each cell: runs under nsys profile, captures wall-time and per-frame
# stats plus an nsys-rep trace, and logs VMAF score for sanity.
#
# Results land under benchmark/results/<tag>/. The nsys-rep files are
# binary traces openable in nsys-ui for per-frame timeline inspection.

set -euo pipefail

VMAF_BIN="${VMAF_BIN:-/home/dsummer/VMAF/libvmaf/build/tools/vmaf}"
CLIPS_DIR="${CLIPS_DIR:-/tmp/vmaf-bench/clips}"
NSYS="${NSYS:-/usr/local/cuda/bin/nsys}"
TAG="${TAG:-$(date +%Y-%m-%d)-baseline}"
OUT_DIR="${OUT_DIR:-/home/dsummer/VMAF/benchmark/results/${TAG}}"
FRAMES="${FRAMES:-300}"

mkdir -p "${OUT_DIR}"

declare -A GPU_NAME=(
  [0]="rtx5070"
  [1]="rtx4070"
)

RESOLUTIONS=(480p 1080p 2160p)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_one() {
  local dev="$1" res="$2"
  local gpu="${GPU_NAME[$dev]}"
  local ref="${CLIPS_DIR}/ref_${res}.y4m"
  local dist="${CLIPS_DIR}/dist_${res}.y4m"
  local base="${OUT_DIR}/${gpu}_${res}"
  local log_file="${base}.log"
  local vmaf_json="${base}.vmaf.json"
  local nsys_rep="${base}.nsys-rep"

  if [[ ! -f "${ref}" || ! -f "${dist}" ]]; then
    log "SKIP ${gpu} ${res}: missing ${ref} or ${dist}"
    return
  fi

  log "run ${gpu} ${res}"
  local t0 t1
  t0=$(date +%s.%N)

  CUDA_VISIBLE_DEVICES="${dev}" \
  "${NSYS}" profile \
    --force-overwrite=true \
    --output="${base}" \
    --trace=cuda,nvtx,osrt \
    --sample=none --cpuctxsw=none \
    --stats=false \
    -- \
    "${VMAF_BIN}" \
      --gpumask 0 \
      -r "${ref}" -d "${dist}" \
      --model version=vmaf_v0.6.1 \
      --json -o "${vmaf_json}" \
      --frame_cnt "${FRAMES}" \
      --quiet \
    > "${log_file}" 2>&1
  t1=$(date +%s.%N)

  local elapsed fps vmaf_score
  elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
  fps=$(awk -v f="$FRAMES" -v t="$elapsed" 'BEGIN{printf "%.2f", f/t}')
  vmaf_score=$(python3 -c "import json; d=json.load(open('${vmaf_json}')); print(f\"{d['pooled_metrics']['vmaf']['mean']:.4f}\")" 2>/dev/null || echo "N/A")

  printf '  elapsed=%ss  fps=%s  vmaf=%s\n' "${elapsed}" "${fps}" "${vmaf_score}"
  printf '%s,%s,%s,%s,%s,%s\n' "${gpu}" "${res}" "${FRAMES}" "${elapsed}" "${fps}" "${vmaf_score}" \
    >> "${OUT_DIR}/summary.csv"
}

echo "gpu,resolution,frames,elapsed_s,fps,vmaf_mean" > "${OUT_DIR}/summary.csv"

for dev in 0 1; do
  for res in "${RESOLUTIONS[@]}"; do
    run_one "${dev}" "${res}"
  done
done

log "done. results in ${OUT_DIR}"
echo
column -ts, "${OUT_DIR}/summary.csv"
