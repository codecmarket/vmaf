# CUDA Pinned Host Pictures — 2026-04-24

Pins the host-side fetch-ring buffers in `tools/vmaf.c` via
`vmaf_cuda_picture_alloc_pinned` (`cuMemHostAlloc`) instead of
`vmaf_picture_alloc` (pageable `aligned_malloc`). Adds a narrow
libvmaf-internal side-table keyed by host-pic pointer that records the
device picture's `ready` event on each upload, and a new public API
`vmaf_cuda_wait_host_picture_consumed` the tool calls before reusing a
ring slot so pinning doesn't break ring-depth-2 reuse safety.

## Motivation

Profiling on RTX 3080 (150 frames, `crowd_run`) showed 2160p was PCIe-
stalled, not GPU-bound:

| API call               | 480p  | 1080p  | 2160p  |
|------------------------|------:|-------:|-------:|
| `cuMemcpy2DAsync_v2`   | 12.6% |  36.6% | **68.8%** (207 ms) |
| `cuLaunchKernel`       | 40.8% |  25.9% |  11.4% |

`cuMemcpy2DAsync` from pageable host memory stages synchronously through
a pinned bounce buffer inside the API call before returning. The
pre-existing `tools/vmaf.c:156-158` comment relied on exactly that
synchronous-staging behavior to justify ring depth 2. Pinning makes the
copy truly async and lets it overlap with kernels.

## Results (RTX 3080, bare wall-clock, best-of-3, 300 frames)

| Tier  | Before (fps) | After (fps) |     Δ      |
|-------|-------------:|------------:|-----------:|
| 480p  |       726.39 |      765.31 |    +5.4%   |
| 1080p |       358.00 |      412.65 | **+15.3%** |
| 2160p |        98.72 |      122.40 | **+24.0%** |

Bit-exact on all three tiers (18.7483 / 18.7538 / 23.5754 before and
after, best-of-3 on the same clips).

## API-level confirmation (2160p, 150 frames)

| API call             | Before  | After  |  Δ   |
|----------------------|--------:|-------:|-----:|
| `cuMemcpy2DAsync_v2` | 207 ms  | 5.1 ms | -40× |
| `cuLaunchKernel`     | 34 ms   | 34 ms  | flat |

The upload stall is gone; kernel-launch overhead is now the dominant
residual per-API-time line item, as predicted. The new
`cuMemHostAlloc` / `cuMemFreeHost` line items are one-time setup /
teardown, not per-frame.

## Design

Three pieces:

1. **Pinned allocation in the tool ring** (`tools/vmaf.c`): when CUDA is
   active (`c.gpumask != ~0u` and `vmaf_cuda_state_init` succeeded), each
   of the 2 × FETCH_RING_DEPTH = 4 host picture slots is allocated via
   `vmaf_cuda_picture_alloc_pinned` (backed by `cuMemHostAlloc`). The
   pinned allocator already existed in `libvmaf/src/cuda/picture_cuda.c`
   but was only wired into the internal picture-pool path
   (`VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED`). Promoted to
   `libvmaf/include/libvmaf/libvmaf_cuda.h` for tool use.

2. **Upload-tracking side-table** (`libvmaf/src/cuda/common.{h,c}`):
   fixed 8-slot table on `VmafCudaState`, keyed by host-pic data pointer,
   stores the device picture's `ready` event recorded inside
   `vmaf_cuda_picture_upload_async`. Inserted from
   `translate_picture_host` in `libvmaf.c` immediately after the
   upload call; round-robin eviction when full.

3. **Public wait API** (`vmaf_cuda_wait_host_picture_consumed`):
   tool calls this on the ring slot about to be overwritten; looks up
   the stored `ready` event and `cuEventSynchronize`s on it. Returns
   immediately if the slot has no tracking entry (first K frames).

Ring depth stays at 2. The `ready` event is recorded on the upload
stream after the `cuMemcpy2DAsync`, so synchronizing on it is
exactly the "host buffer has been fully read" guarantee we need.

## Files changed

- `libvmaf/include/libvmaf/libvmaf_cuda.h` — exported
  `vmaf_cuda_picture_alloc_pinned` and `vmaf_cuda_wait_host_picture_consumed`.
- `libvmaf/src/cuda/common.h` — added `host_pic_slots[8]` + `host_pic_next`
  on `VmafCudaState`; declared `vmaf_cuda_host_pic_track`.
- `libvmaf/src/cuda/common.c` — `vmaf_cuda_host_pic_track` implementation
  (lookup-then-insert, round-robin evict).
- `libvmaf/src/libvmaf.c` — tracking call after
  `vmaf_cuda_picture_upload_async`; `vmaf_cuda_wait_host_picture_consumed`
  implementation.
- `tools/vmaf.c` — `FetchRing` gets `cu_state` field; uses
  `vmaf_cuda_picture_alloc_pinned` on the CUDA path; calls
  `vmaf_cuda_wait_host_picture_consumed` before `copy_picture_data`
  overwrites a slot.

## Reproduce

```sh
for res in 480p 1080p 2160p; do for r in 1 2 3; do
  t0=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES=0 /home/dsummer/VMAF/libvmaf/build/tools/vmaf \
    --gpumask 0 \
    -r /tmp/vmaf-bench/clips/ref_${res}.y4m \
    -d /tmp/vmaf-bench/clips/dist_${res}.y4m \
    --model version=vmaf_v0.6.1 --frame_cnt 300 --quiet >/dev/null
  t1=$(date +%s.%N)
  awk -v a=$t0 -v b=$t1 -v res=$res -v r=$r \
    'BEGIN{printf "res=%s run=%s elapsed=%.3fs fps=%.2f\n", res, r, b-a, 300/(b-a)}'
done; done
```

## Raw fps per run

```
480p:   753.77  765.31  742.57
1080p:  412.65  407.61  408.16
2160p:  121.75  122.15  122.40
```
