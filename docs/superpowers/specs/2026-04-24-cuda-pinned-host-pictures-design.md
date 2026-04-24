# CUDA Pinned Host Pictures

**Date:** 2026-04-24
**Scope:** `tools/vmaf.c` fetch ring + narrow libvmaf CUDA API addition. No kernel changes. No picture-pool changes.
**Goal:** Let `cuMemcpy2DAsync` actually overlap with kernel execution by pinning the host ring buffers, eliminating the synchronous-staging stall that dominates 2160p CUDA API time.

## Motivation

Profiling the CUDA pipeline on RTX 3080, 150 frames, `crowd_run` at 480p / 1080p / 2160p (traces in `/tmp/vmaf-prof/`):

| API call               | 480p  | 1080p  | 2160p  |
|------------------------|------:|-------:|-------:|
| `cuMemcpy2DAsync_v2`   | 12.6% |  36.6% | **68.8%** (207 ms) |
| `cuLaunchKernel`       | 40.8% |  25.9% | 11.4%  |

At 2160p the pipeline is PCIe-stalled, not GPU-bound. The 207 ms is not bandwidth — it is synchronous staging. `vmaf_picture_alloc` uses `aligned_malloc` (pageable), and `cuMemcpy2DAsync` from pageable host memory stages through a pinned bounce buffer synchronously inside the API call before returning. The "Async" in the name only refers to the device-side copy after staging completes.

The existing comment in `tools/vmaf.c:156-158` relies on exactly this synchronous-staging behavior to justify ring depth 2. Fixing the stall therefore requires both pinning the ring buffers and re-proving reuse safety without that guarantee.

## Non-goals

- **No kernel changes.** Same exclusion as the 2026-04-19 Scope A design.
- **No picture-pool changes.** The libvmaf-internal picture pool and `VmafCudaPicture` device allocations are already well-behaved.
- **No public API semantic change to `vmaf_picture_alloc`.** We are not auto-pinning inside libvmaf when CUDA is active — that would couple the picture allocator to CUDA state and is a larger follow-up.
- **No configuration flag.** Pinning is the correct behavior on the CUDA path; there is no reason to make it opt-in.

## Current state (as of commit `979a57bc`)

- `tools/vmaf.c` maintains a per-stream `FetchRing` with depth 2 of `VmafPicture`s allocated via `vmaf_picture_alloc`. Ring depth justified by synchronous-staging semantics.
- `libvmaf/src/picture.c:89`: `vmaf_picture_alloc` calls `aligned_malloc(pic_size, DATA_ALIGN)` → pageable.
- `libvmaf/src/cuda/picture_cuda.c:80`: after each `cuMemcpy2DAsync`, `cuEventRecord(cuda_priv->cuda.ready, cuda_priv->cuda.str)` records a "ready" event on the upload stream.
- `libvmaf/src/cuda/picture_cuda.c:269`: `vmaf_cuda_picture_get_ready_event` exposes that event.
- `libvmaf/src/libvmaf.c:853`: `vmaf_read_pictures` calls `vmaf_cuda_picture_upload_async(pic_device, host_pic, 0x1)` — this is the single point where host → device copy is enqueued.

## Design

### 1. Pin the fetch ring

In `tools/vmaf.c`, after `vmaf_picture_alloc` succeeds for each ring slot, register the underlying host allocation with the CUDA driver:

```c
cuMemHostRegister(ring->pic[i].data[0],
                  total_alloc_size,
                  CU_MEMHOSTREGISTER_PORTABLE);
```

We register the plane-0 pointer (which is the start of the underlying `aligned_malloc` blob; planes 1 and 2 are offsets into the same blob) with `total_alloc_size` equal to the sum of all plane sizes. One register call per slot rather than per plane.

`total_alloc_size` is recomputed at ring-init time from the same formula `vmaf_picture_alloc` uses — we do not touch `vmaf_picture_alloc`. The formula is derived from `pic->stride[]`, `pic->h`, and the subsampling of the pixel format; it is deterministic given those inputs. The implementation plan locks down the exact expression.

At `fetch_ring_close`, `cuMemHostUnregister(ring->pic[i].data[0])` runs *before* `vmaf_picture_unref`.

**Gating:**
- `HAVE_CUDA` at compile time.
- At runtime, pin only when `c.gpumask != ~0u` and `vmaf_cuda_state_init` succeeded (i.e., the same condition under which `vmaf_cuda_import_state` is called today).
- If `cuMemHostRegister` returns an error, log a warning, leave the buffer unpinned, and continue — the ring depth 2 correctness argument in the absence of pinning still holds, and a failed pinning should not fail the run.

### 2. Correctness: event-gated reuse

Once pinned, `cuMemcpy2DAsync` returns before the copy completes. Overwriting a ring slot's host buffer while its last H2D is still draining is a correctness bug. Ring depth stays at 2; we gate reuse on the device picture's `ready` event.

**New libvmaf API** (CUDA-only header):

```c
/* Blocks until any in-flight H2D upload whose host source was `host_pic`
 * has completed. Safe to call when no such upload is tracked (no-op).
 * Returns 0 on success, negative on error. */
int vmaf_cuda_wait_host_picture_consumed(VmafContext *vmaf,
                                         VmafPicture *host_pic);
```

**Library-side implementation:**

`vmaf_read_pictures` already calls `vmaf_cuda_picture_upload_async(pic_device, host_pic, ...)` at `libvmaf.c:853`. Immediately after each such call we:

1. Look up the slot for `host_pic->data[0]` in an 8-entry side-table on `VmafContext`.
2. Store the device picture's `ready` event (`vmaf_cuda_picture_get_ready_event(pic_device)`) into that slot. Capture additionally the device picture handle so we can invalidate the slot when the device picture is released back to the pool.
3. The side-table lives on the context's CUDA state; capacity 8 (`FETCH_RING_DEPTH × 2` streams = 4, with 2× slack).

`vmaf_cuda_wait_host_picture_consumed` looks up by `host_pic->data[0]`; on hit, `cuEventSynchronize(stored_event)`; on miss, no-op. The call is always made from the tool thread which has the CUDA context bound.

**Slot invalidation** (avoiding stale events):
When the picture pool recycles a device picture, the `ready` event handle may be re-recorded against a different upload later. We invalidate the side-table slot when the corresponding device picture's `finished` event fires — piggyback on the existing `cuEventRecord(finished, …)` path at `libvmaf.c:1030/1037`. Simpler and equivalently correct: on each new `vmaf_read_pictures` call, we iterate the side-table and evict any entries whose device-pic handle matches a device pic about to be re-used for this frame. 4–8 entries, iteration is trivially cheap.

The implementation plan picks one of these two approaches; both are correct.

**Tool-side call site:**

In `fetch_picture`, immediately before `copy_picture_data` overwrites the selected ring slot (i.e., after the `vmaf_picture_ref(pic, &ring->pic[idx])` call at `tools/vmaf.c:203-207`, before the `copy_picture_data` at line 210):

```c
#ifdef HAVE_CUDA
if (cuda_active) {
    vmaf_cuda_wait_host_picture_consumed(vmaf, &ring->pic[idx]);
}
#endif
```

First two fetches per stream no-op through the "key not present" path. Steady state: `cuEventSynchronize` on an event that fired 1-2 frames ago — near-instant unless CPU-starved.

### 3. Why ring depth stays at 2

The upload of slot `i` at frame N is fully ordered against its corresponding kernel chain by the existing `ready`-event wait in the feature extractors. Frame N+2 reuses slot `i`; by the time we're about to overwrite it, frame N's kernel chain has drained (otherwise frame N+1 would still be blocked). In practice the `ready` event fires within 1-2 ms at 2160p; no additional ring depth is needed. Deepening the ring would add host memory pressure without a correctness win.

## File-level change list

- `libvmaf/src/cuda/common.h` — add the side-table struct on `VmafCudaState` (or a dedicated CUDA-internal header), prototype for the side-table helpers.
- `libvmaf/src/cuda/common.c` — side-table init/destroy, insert-on-upload helper, lookup-and-wait helper, invalidation helper.
- `libvmaf/src/libvmaf.c` — after each `vmaf_cuda_picture_upload_async` in `vmaf_read_pictures`, insert into side-table. Invalidate stale entries (strategy locked in implementation plan).
- `libvmaf/include/libvmaf/libvmaf_cuda.h` — export `vmaf_cuda_wait_host_picture_consumed`.
- `tools/vmaf.c` — `cuMemHostRegister` / `cuMemHostUnregister` around the fetch ring; `vmaf_cuda_wait_host_picture_consumed` call in `fetch_picture` before `copy_picture_data`; compute `total_alloc_size` from picture geometry; CUDA-active detection (already available via the `cu_state` local).

No new files.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `cuMemHostRegister` fails (fragmentation, duplicate registration) | Log warning, skip pinning for that slot, continue. Ring depth 2 correctness under pageable staging still holds. |
| Stored `ready` event handle becomes stale when picture pool recycles its device picture | Invalidate side-table entry when device picture is released; alternative strategy (evict on new upload by device-pic handle match) is equivalently correct. Covered in §2. |
| Host buffer overwritten before H2D completes | `vmaf_cuda_wait_host_picture_consumed` called immediately before `copy_picture_data`; `cuEventSynchronize` on the recorded `ready` event. |
| Side-table grows unboundedly | Fixed capacity 8; evict on insert when full (LRU over the 8 slots). Ring needs only 4 live entries at any time. |
| Pinned allocation survives `vmaf_picture_unref` and leaks pinning | `cuMemHostUnregister` called explicitly before `vmaf_picture_unref` in `fetch_ring_close`. |
| Non-CUDA builds break | All additions under `HAVE_CUDA`. Pure CPU path: no pinning, no side-table, no new API called. |
| Bit-exactness regression | Validation gate: test suite under `enable_cuda=true` + 3 clip tiers. Any score change fails the change. |

## Validation

1. **Numerical equivalence (gating).** `libvmaf/test` under `enable_cuda=true`. Plus the canned 300-frame `crowd_run` at 480p/1080p/2160p — pooled VMAF must be bit-exact against the current master run on the same clips.
2. **Perf measurement.** Bare wall-clock, best-of-3, 300 frames, three resolutions. Targets (vs. current master on RTX 3080 today: 726 / 358 / 99 fps):
   - 2160p: +30-50% fps (removing the 207 ms API stall).
   - 1080p: +10-20% fps (memcpy is 36.6% of API time there).
   - 480p: noise (launch-bound, not upload-bound).
3. **Profile re-measurement.** nsys on 2160p post-change. `cuMemcpy2DAsync` share of API time should drop sharply; GPU timeline should show H2D of frame N+1 overlapping with kernels of frame N.
4. **Registration-balance check.** Instrument `cuMemHostRegister` / `cuMemHostUnregister` counts at tool shutdown. They must balance.

## Rollout

Single commit series on `master`:

1. Library: side-table infra + `vmaf_cuda_wait_host_picture_consumed` export.
2. Tool: ring pinning + sync call.

No feature flag. `CHANGELOG.md` note under "CUDA pipeline."
