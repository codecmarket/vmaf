#!/bin/bash
# Fetches Xiph derf reference clips, truncates to FRAMES frames, produces
# matching Vulkan-H.264-encoded distorted variants at 480p/1080p/2160p.
# Outputs land in $CLIPS_DIR (default /tmp/vmaf-bench/clips), which is
# deliberately ephemeral (tmpfs) — these are multi-GB files and should
# not enter git.

set -euo pipefail

FRAMES="${FRAMES:-300}"
CLIPS_DIR="${CLIPS_DIR:-/tmp/vmaf-bench/clips}"
FFMPEG="${FFMPEG:-/home/dsummer/ffmpeg-local/bin/ffmpeg}"
FFMPEG_LIB="${FFMPEG_LIB:-/home/dsummer/ffmpeg-local/lib}"
SRC_BASE="${SRC_BASE:-https://media.xiph.org/video/derf/y4m}"
CLIP="${CLIP:-crowd_run}"

export LD_LIBRARY_PATH="${FFMPEG_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "${CLIPS_DIR}"
cd "${CLIPS_DIR}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

fetch() {
  local name="$1"
  if [[ -f "${name}" ]]; then
    log "${name} already present, skipping fetch"
    return
  fi
  log "fetching ${name}"
  curl -fL --progress-bar -o "${name}" "${SRC_BASE}/${name}"
}

trim() {
  # Trim a Y4M to the first N frames and rewrite to a new Y4M.
  local in="$1" out="$2"
  if [[ -f "${out}" ]]; then
    log "${out} already present, skipping trim"
    return
  fi
  log "trimming ${in} -> ${out} (${FRAMES} frames)"
  "${FFMPEG}" -v error -y -i "${in}" -frames:v "${FRAMES}" "${out}"
}

scale_480p() {
  # Downscale 1080p Y4M to 854x480 (16:9), first N frames only.
  local in="$1" out="$2"
  if [[ -f "${out}" ]]; then
    log "${out} already present, skipping 480p scale"
    return
  fi
  log "scaling ${in} -> ${out} (480p, ${FRAMES} frames)"
  "${FFMPEG}" -v error -y -i "${in}" -frames:v "${FRAMES}" \
    -vf "scale=854:480:flags=lanczos" -pix_fmt yuv420p "${out}"
}

vulkan_encode_decode() {
  # Vulkan H.264 encode at the tier's bitrate, then decode back to a Y4M
  # of the same dimensions. The decoded Y4M is the "distorted" VMAF input.
  local in="$1" bitrate="$2" label="$3"
  local h264="dist_${label}.mkv"
  local out="dist_${label}.y4m"
  if [[ -f "${out}" ]]; then
    log "${out} already present, skipping encode+decode"
    return
  fi
  log "vulkan encoding ${in} @ ${bitrate} -> ${h264}"
  "${FFMPEG}" -v error -y \
    -init_hw_device vulkan=vk:0 -filter_hw_device vk \
    -i "${in}" \
    -vf "format=nv12,hwupload" \
    -c:v h264_vulkan -b:v "${bitrate}" \
    -frames:v "${FRAMES}" \
    "${h264}"
  log "decoding ${h264} -> ${out}"
  "${FFMPEG}" -v error -y -i "${h264}" -frames:v "${FRAMES}" -pix_fmt yuv420p "${out}"
}

# ---- 1080p tier ----
fetch "${CLIP}_1080p50.y4m"
trim "${CLIP}_1080p50.y4m" "ref_1080p.y4m"
vulkan_encode_decode "ref_1080p.y4m" "1M" "1080p"

# ---- 2160p tier ----
fetch "${CLIP}_2160p50.y4m"
trim "${CLIP}_2160p50.y4m" "ref_2160p.y4m"
vulkan_encode_decode "ref_2160p.y4m" "4M" "2160p"

# ---- 480p tier (derived from 1080p) ----
scale_480p "ref_1080p.y4m" "ref_480p.y4m"
vulkan_encode_decode "ref_480p.y4m" "300k" "480p"

log "done. clips in ${CLIPS_DIR}:"
ls -lh "${CLIPS_DIR}" | awk 'NR>1 {print "  " $5 "  " $9}'
