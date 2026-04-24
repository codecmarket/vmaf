/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef __VMAF_CUDA_H__
#define __VMAF_CUDA_H__

#include "libvmaf/libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafCudaState VmafCudaState;

typedef struct VmafCudaConfiguration {
    void *cu_ctx; ///< CUcontext
} VmafCudaConfiguration;

/**
 * Initialize VmafCudaState.
 * VmafCudaState can optionally be configured with VmafCudaConfiguration.
 *
 * @param cu_state The CUDA state to open.
 *
 * @param cfg      Optional configuration parameters.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_state_init(VmafCudaState **cu_state, VmafCudaConfiguration cfg);

/**
 * Import VmafCudaState for use during CUDA feature extraction.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()`.
 *
 * @param cu_state CUDA state allocated with `vmaf_cuda_state_init()`.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_import_state(VmafContext *vmaf, VmafCudaState *cu_state);

enum VmafCudaPicturePreallocationMethod {
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE = 0,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST,
    VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
};

typedef struct VmafCudaPictureConfiguration {
    struct {
        unsigned w, h;
        unsigned bpc;
        enum VmafPixelFormat pix_fmt;
    } pic_params;
    enum VmafCudaPicturePreallocationMethod pic_prealloc_method;
} VmafCudaPictureConfiguration;

/**
 * Config and preallocate VmafPictures for use during CUDA feature extraction.
 * The preallocated VmafPicture data buffers are set according to
 * cfg.pic_prealloc_method.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()`.
 *
 * @param cfg VmafPicture parameter configuration.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_preallocate_pictures(VmafContext *vmaf,
                                   VmafCudaPictureConfiguration cfg);

/**
 * Fetch a preallocated VmafPicture for use during CUDA feature extraction.
 * pictures are allocated during `vmaf_cuda_preallocate_pictures` and data
 * buffers are set according to cfg.pic_prealloc_method.
 *
 * @param vmaf VMAF context allocated with `vmaf_init()` and
 *             initialized with `vmaf_cuda_preallocate_pictures()`.
 *
 * @param pic Preallocated picture.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_fetch_preallocated_picture(VmafContext *vmaf, VmafPicture* pic);

/**
 * Allocate a VmafPicture backed by pinned (page-locked) host memory via
 * cuMemHostAlloc. Pinned buffers let cuMemcpy2DAsync actually run the
 * host-to-device copy asynchronously instead of staging through a bounce
 * buffer inside the API call. The allocation is released on the final
 * vmaf_picture_unref via cuMemFreeHost.
 *
 * @param pic       Picture to initialize.
 * @param pix_fmt   Pixel format.
 * @param bpc       Bits per component.
 * @param w, h      Picture dimensions in pixels.
 * @param cu_state  CUDA state previously initialized with
 *                  vmaf_cuda_state_init.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_picture_alloc_pinned(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                                   unsigned bpc, unsigned w, unsigned h,
                                   VmafCudaState *cu_state);

/**
 * Block until any in-flight host-to-device upload whose source was
 * `host_pic` has completed. Intended for tools that recycle host
 * picture buffers: once the source-side read is done, the host
 * buffer can be safely overwritten. Safe to call for a pic that has
 * never been uploaded (no-op) or when CUDA is not active (no-op).
 *
 * @param vmaf     VMAF context with an imported CUDA state.
 * @param host_pic The host-side VmafPicture that was most recently
 *                 passed to vmaf_read_pictures.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_cuda_wait_host_picture_consumed(VmafContext *vmaf,
                                         VmafPicture *host_pic);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_CUDA_H__ */
