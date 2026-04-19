# CUDA Runtime Modernization (Scope A)

**Date:** 2026-04-19
**Scope:** Infra/runtime layer under `libvmaf/src/cuda/` and the CUDA feature extractors' use of it. No kernel changes. No public API changes.
**Goal:** Reduce per-frame CPU-side overhead in the CUDA pipeline by replacing sync allocations with a stream-ordered memory pool, eliminating context push/pop churn in hot paths, warming the pool at startup, and tightening the top-level sync. Raise the compiled compute baseline now that Turing support can be dropped.

## Motivation

Profiling the current code reveals two CPU-side tax sources on every frame:

1. **Allocator churn.** Feature extractors and picture allocators use `cuMemAlloc` / `cuMemAllocPitch`. Each call is a context-global, implicitly-synchronizing device allocation. For a VMAF run with many frames at modest resolution, this dominates CPU time between kernels.
2. **Context push/pop churn.** The codebase wraps almost every CUDA driver call with `cuCtxPushCurrent` / `cuCtxPopCurrent` — currently 93 such pairs, including around every `cuMemcpy*Async` and every tiny helper. On CUDA 13's driver these are cheap but not free, and on the hot path they create measurable overhead and obscure the actual per-frame latency.

Neither problem needs kernel changes. Both are solved by straightforward driver-API idiom updates that became standard in CUDA 11.2+ and are fully supported on the target architectures (Ada, Blackwell).

## Non-goals

- **No kernel changes.** `__shfl_down_sync`, the int64 warp-reduce, the emulated `atomicAdd_int64`, cooperative-groups adoption — all deferred to a potential Scope B.
- **No CUDA graphs.** Capturing the per-frame DAG would give further wins but adds material correctness risk; evaluate as a follow-up once this cleaner baseline is in and measured.
- **No public API change.** `libvmaf_cuda.h` surface stays identical. Callers recompile against the same symbols.
- **No algorithmic change.** Feature scores must be bit-exact with `master`.

## Current state (as of commit `966be8d5`)

- `libvmaf/src/cuda/common.c`
  - `vmaf_cuda_buffer_alloc` — sync `cuMemAlloc`.
  - `vmaf_cuda_buffer_free` — sync `cuMemFree`.
  - Every helper brackets its body with `cuCtxPushCurrent` / `cuCtxPopCurrent`.
  - `vmaf_cuda_sync` calls `cuCtxSynchronize` (blocks all streams in the context).
- `libvmaf/src/cuda/picture_cuda.c`
  - `vmaf_cuda_picture_alloc` uses `cuMemAllocPitch` (sync) per plane, behind push/pop.
  - `vmaf_cuda_picture_free` already uses `cuMemFreeAsync` on the picture's stream — good; consistent with the target model.
  - Pinned host alloc uses `cuMemHostAlloc` with portable flag only.
- `libvmaf/src/feature/cuda/*.c` — feature extractors allocate scratch via `vmaf_cuda_buffer_alloc`, so they inherit the sync behavior automatically.
- `VmafCudaBuffer` and `VmafCudaState` are defined in `libvmaf/src/cuda/common.h` and are not exposed externally. `VmafCudaBuffer`'s fields (`size`, `data`) can be extended without breaking callers.

## Design

### 1. Stream-ordered memory pool

At `vmaf_cuda_state_init` time, after the context is established, create a `CUmemoryPool` bound to the device:

- `cuMemPoolCreate(&pool, &props)` with `props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED`, `props.handleTypes = CU_MEM_HANDLE_TYPE_NONE`, `props.location = { CU_MEM_LOCATION_TYPE_DEVICE, cu_state->dev }`.
- `cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &UINT64_MAX)` so memory is retained across frames rather than handed back to the driver between streams going idle.

Store the pool on `VmafCudaState`:

```c
typedef struct VmafCudaState {
    CUcontext ctx;
    CUstream  str;
    CUdevice  dev;
    CUmemoryPool pool;      /* NEW: owned, destroyed on release */
    CudaFunctions *f;
    int release_ctx;
} VmafCudaState;
```

Extend `VmafCudaBuffer` to remember the stream it was allocated on (so frees go back to the same stream):

```c
typedef struct VmafCudaBuffer {
    size_t size;
    CUdeviceptr data;
    CUstream alloc_stream;  /* NEW */
} VmafCudaBuffer;
```

Route allocations through the pool:

- `vmaf_cuda_buffer_alloc(state, &buf, size)` → `cuMemAllocFromPoolAsync(&buf->data, size, state->pool, state->str)`; record `buf->alloc_stream = state->str`.
- `vmaf_cuda_buffer_free(state, buf)` → `cuMemFreeAsync(buf->data, buf->alloc_stream)`. Falls back to `cuMemFree` if `alloc_stream` is null (defensive, not expected).

Picture allocator (`vmaf_cuda_picture_alloc`):

- Replace `cuMemAllocPitch` with our own pitch math + `cuMemAllocFromPoolAsync`. Pitch = round up the plane-row size to `CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT` (query once per state and cache it on `VmafCudaState`). This matches the effective pitch `cuMemAllocPitch` returns on Ada/Blackwell for VMAF's plane sizes while letting us take the allocation from the pool.
- Allocate on the picture's own stream (`priv->cuda.str`) and free via `cuMemFreeAsync` on the same stream (already the pattern in `vmaf_cuda_picture_free`).

Pinned host allocations stay on `cuMemHostAlloc` — they already are stream-oblivious and not in the hot path per frame.

### 2. Context-bind once

Add an internal `static int ensure_ctx_current(VmafCudaState *s)` that:

1. Queries current ctx with `cuCtxGetCurrent(&cur)`.
2. If `cur != s->ctx`, calls `cuCtxSetCurrent(s->ctx)`.

Call sites that currently wrap a single driver call in `push/pop` are rewritten to rely on an already-bound context:

- **Workers that enter from a foreign thread** (the top-level thread pool for feature extraction and picture prealloc) call `ensure_ctx_current` once on entry.
- **Hot-path helpers** (`vmaf_cuda_buffer_alloc`, `vmaf_cuda_buffer_free`, `vmaf_cuda_buffer_upload_async`, `vmaf_cuda_buffer_download_async`, `vmaf_cuda_picture_upload_async`, `vmaf_cuda_picture_download_async`, `vmaf_cuda_picture_synchronize`, event record/wait, stream sync) drop their push/pop pairs outright. They trust the caller's thread has a current context — which it does, post-`ensure_ctx_current`.
- **Cold-path helpers** (`vmaf_cuda_state_init`, `vmaf_cuda_release`, one-time picture allocations) keep explicit `cuCtxPushCurrent` / `cuCtxPopCurrent` since they may run on threads that haven't yet bound.

Expected reduction: 93 push/pop pairs → under 10. Net overhead saved per frame ≈ (N_calls × 2 × driver-call cost). On the target hardware this is small per-call but cumulative across a feature DAG.

### 3. Pool warmup

At `vmaf_cuda_preallocate_pictures` (and inside `vmaf_cuda_picture_alloc` on the first call if prealloc wasn't used), estimate the frame-resident working set in bytes — sum of all plane allocations per picture × ring buffer depth, plus a configurable pad for feature scratch — and do a transient `cuMemAllocFromPoolAsync(size=budget)` followed by `cuMemFreeAsync` on the state stream, then `cuStreamSynchronize`. This forces the pool to grow its backing to the budget and, because the release threshold is `UINT64_MAX`, the freed memory stays in the pool for reuse.

Without warmup, the first ~5 frames pay allocator-growth cost as the pool expands incrementally. The implementation plan will determine whether the feature-scratch pad is a fixed multiplier of plane size or computed per-extractor.

### 4. Tighten top-level sync

Change `vmaf_cuda_sync` to call `cuStreamSynchronize(state->str)` instead of `cuCtxSynchronize`. Context-wide sync blocks every stream in the context — including unrelated streams owned by the host app embedding libvmaf (via the `VmafCudaConfiguration.cu_ctx` path). Stream-scoped sync is strictly narrower and correct for the only thing `vmaf_cuda_sync` is documented to do.

### 5. Build baseline

In `libvmaf/src/meson.build`, drop `sm_75` from the gencode list. The resulting set (CUDA ≥ 13):

```
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_100,code=sm_100
-gencode=arch=compute_120,code=sm_120
-gencode=arch=compute_120,code=compute_120
```

Turing (sm_75) support is dropped. CUDA 11.x toolchain paths and the <11.8 branch stay as-is for older-toolchain builds, but under CUDA 13 we target Ampere+ only. Document this in `CHANGELOG.md`.

### Loader compatibility

All newly-used driver entry points are present in `ffnvcodec >= 11.1.5.1`:

- `cuMemPoolCreate`, `cuMemPoolDestroy`, `cuMemPoolSetAttribute` — present.
- `cuMemAllocFromPoolAsync`, `cuMemFreeAsync` — present (11.2+ driver API; the loader table already exposes the latter, since `picture_cuda.c` uses it today).
- `cuCtxSetCurrent`, `cuCtxGetCurrent` — present.
- `cuDeviceGetAttribute` — present (needed for pitch alignment).

If any is missing from the dynlink table at build time, bump the `ffnvcodec` min version in `meson.build` and document it. Confirmed present against the nv-codec-headers 11.1.5.1 table at design time.

## File-level change list

- `libvmaf/src/cuda/common.h` — extend `VmafCudaState` (add `pool`, `pitch_align`), extend `VmafCudaBuffer` (add `alloc_stream`), add `vmaf_cuda_ctx_ensure` prototype.
- `libvmaf/src/cuda/common.c` — pool create/destroy, route allocs through pool, drop push/pop from hot-path helpers, `vmaf_cuda_sync` switched to stream-scoped.
- `libvmaf/src/cuda/picture_cuda.c` — replace `cuMemAllocPitch` with pool-backed pitched alloc, drop push/pop from memcpy/event/sync paths, `ensure_ctx_current` at picture alloc entry.
- `libvmaf/src/feature/cuda/*_cuda.c` — add `ensure_ctx_current` at each extractor's init entry point (thread entry), remove redundant push/pop around allocs and launches.
- `libvmaf/src/meson.build` — drop `sm_75`, tighten gencode.
- `CHANGELOG.md` — note dropped Turing support and runtime changes.

No new files.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Pool outlives its context → UB on destroy | `vmaf_cuda_release` destroys pool before `cuDevicePrimaryCtxRelease`. |
| Free on destroyed stream errors | `vmaf_cuda_buffer_free` checks `alloc_stream`; falls back to sync `cuMemFree` if the stream handle is invalid. Streams tied to state live for the state's full lifetime, so in practice this path is unreachable. |
| Cross-stream consumer of a pool-allocated buffer reads stale/pending memory | Stream-ordered allocations are ordered only on the allocating stream. Audit each extractor: any place a buffer is produced on one stream and consumed on another must already record+wait an event. Most extractors use a single stream; cross-stream points (picture → feature) already use `cuEventRecord`/`cuStreamWaitEvent`. Add missing event ordering where the audit surfaces gaps. |
| Pitch math mismatches `cuMemAllocPitch` and confuses kernels | Use `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT)` for the alignment and round up the plane-row byte count accordingly. Validate by comparing against `cuMemAllocPitch`-returned pitches on the target GPUs for the full resolution/bpc matrix used by tests. |
| Raising base to sm_80 breaks users on Turing/Volta | Users on those architectures must stay on CUDA 12.x / libvmaf ≤ current. Documented in `CHANGELOG.md` as a breaking change for the CUDA build. |
| Reduced push/pop coverage exposes an assumption that a specific helper entered from an unbound thread | Worker entry points call `ensure_ctx_current`. For any helper callable from both bound and unbound threads, keep push/pop. The audit must identify any such helper explicitly. |

## Validation

1. **Numerical equivalence (gating).** Run `libvmaf/test` on the full test clip set under `enable_cuda=true`. Every VMAF score must be bit-exact with `master`.
2. **Build matrix.** `enable_cuda=true` + `enable_nvcc=true` under CUDA 13.2 on the developer box (Ada/Blackwell present). Also verify the clang-as-CUDA-frontend path still builds (`enable_nvcc=false`).
3. **Perf measurement.**
   - Canned clip at 480p, 1080p, and 2160p.
   - Measure wall-time and per-frame CPU time before/after. Expect a measurable reduction at 480p (where launch/alloc overhead dominates) and a smaller but non-zero reduction at 1080p. 2160p may be within noise.
   - Capture `nsys profile` before/after; compare CPU-side gap between kernel launches. The reduction in that gap is the direct evidence of the optimization.
4. **Allocator check.** With the pool warmed, observe zero `cuMemAlloc*` calls after the first N frames (N = warmup window). Confirm via `nsys` trace.
5. **Leak check.** Run VMAF on a long clip, verify `cuMemPoolGetAttribute(CU_MEMPOOL_ATTR_USED_MEM_CURRENT)` returns to ~zero at shutdown, and that `vmaf_cuda_release` destroys the pool cleanly.

## Rollout

Single commit series on `master` (this is an internal repo with no external downstream users beyond users who pull from `master`). Breaking change note in `CHANGELOG.md`. No feature flag; the modernized runtime replaces the old one directly.
