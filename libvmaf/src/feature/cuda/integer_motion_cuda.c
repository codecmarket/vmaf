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

#include <errno.h>
#include <math.h>
#include <string.h>

#include "common.h"
#include "cpu.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_motion_cuda.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

struct MotionStateCuda;

typedef struct write_score_parameters_moco {
    VmafFeatureCollector *feature_collector;
    struct MotionStateCuda *s;
    unsigned h, w;
    unsigned index;
    uint64_t *sad_host;
} write_score_parameters_moco;

#define MOTION_CUDA_HOST_SLOTS 4

typedef struct MotionCudaSlot {
    VmafCudaBuffer *sad;
    uint64_t *sad_host;
    write_score_parameters_moco cpu_param;
    CUstream stream;
    CUevent consumed;
    CUevent kernel_done;
} MotionCudaSlot;

typedef struct MotionStateCuda {
    CUfunction funcbpc8, funcbpc16;
    VmafCudaBuffer* blur[2];
    MotionCudaSlot slots[MOTION_CUDA_HOST_SLOTS];
    unsigned next_slot;
    unsigned index;
    double score;
    bool debug;
    bool motion_force_zero;
    void (*calculate_motion_score)(const VmafPicture* src, VmafCudaBuffer* src_blurred,
            const VmafCudaBuffer* prev_blurred, VmafCudaBuffer* sad,
            unsigned width, unsigned height,
            ptrdiff_t src_stride, ptrdiff_t blurred_stride, unsigned src_bpc,
            CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions *cu_f, CUstream stream);
    VmafDictionary *feature_name_dict;
} MotionStateCuda;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionStateCuda, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

static int extract_force_zero(VmafFeatureExtractor *fex,
        VmafPicture *ref_pic, VmafPicture *ref_pic_90,
        VmafPicture *dist_pic, VmafPicture *dist_pic_90,
        unsigned index,
        VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;

    (void) fex;
    (void) ref_pic;
    (void) ref_pic_90;
    (void) dist_pic;
    (void) dist_pic_90;

    int err =
        vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict, "VMAF_integer_feature_motion2_score", 0.,
                index);

    if (!s->debug) return err;

    err = vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_motion_score", 0.,
            index);

    return err;
}

void calculate_motion_score(const VmafPicture* src, VmafCudaBuffer* src_blurred,
        const VmafCudaBuffer* prev_blurred, VmafCudaBuffer* sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t blurred_stride, unsigned src_bpc,
        CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions* cu_f, CUstream stream)
{
    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    if (src_bpc == 8){
        void *kernelParams[] = {(void*)src,   (void*) src_blurred, (void*)prev_blurred, (void*)sad,
            &width, &height,     &src_stride,  &blurred_stride};
        CHECK_CUDA(cu_f, cuLaunchKernel(funcbpc8, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    } else {
        void *kernelParams[] = {(void*)src,   (void*) src_blurred, (void*)prev_blurred, (void*)sad,
            &width, &height,     &src_stride,  &blurred_stride};
        CHECK_CUDA(cu_f, cuLaunchKernel(funcbpc16, grid_dim_x,
                    grid_dim_y, 1, block_dim_x, block_dim_y, 1, 0,
                    stream, kernelParams, NULL));
    }
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
        unsigned bpc, unsigned w, unsigned h)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    (void) pix_fmt;
    (void) bpc;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));

    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, motion_score_ptx));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_motion_score_kernel_16bpc"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_motion_score_kernel_8bpc"));

    if (s->motion_force_zero) {
        CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
        fex->extract = extract_force_zero;
        fex->flush = NULL;
        fex->close = NULL;
        return 0;
    }

    s->calculate_motion_score = calculate_motion_score;
    s->score = 0;
    s->next_slot = 0;

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->blur[0], sizeof(uint16_t) * w * h);
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->blur[1], sizeof(uint16_t) * w * h);
    if (ret) goto free_ref;

    for (unsigned i = 0; i < MOTION_CUDA_HOST_SLOTS; i++) {
        MotionCudaSlot *slot = &s->slots[i];
        slot->cpu_param.s = s;

        CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&slot->stream,
                    CU_STREAM_NON_BLOCKING, 0));
        CHECK_CUDA(cu_f, cuEventCreate(&slot->consumed, CU_EVENT_DEFAULT));
        CHECK_CUDA(cu_f, cuEventCreate(&slot->kernel_done, CU_EVENT_DEFAULT));

        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &slot->sad, sizeof(uint64_t));
        if (ret) goto free_ref;

        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state,
                (void**)&slot->sad_host, sizeof(uint64_t));
        if (ret) goto free_ref;
        slot->cpu_param.sad_host = slot->sad_host;
    }

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) { ret = -ENOMEM; goto free_ref; }

    return 0;

free_ref:
    if (s->blur[0]) {
        vmaf_cuda_buffer_free(fex->cu_state, s->blur[0]);
        free(s->blur[0]);
        s->blur[0] = NULL;
    }
    if (s->blur[1]) {
        vmaf_cuda_buffer_free(fex->cu_state, s->blur[1]);
        free(s->blur[1]);
        s->blur[1] = NULL;
    }
    for (unsigned i = 0; i < MOTION_CUDA_HOST_SLOTS; i++) {
        MotionCudaSlot *slot = &s->slots[i];
        if (slot->sad) {
            vmaf_cuda_buffer_free(fex->cu_state, slot->sad);
            free(slot->sad);
            slot->sad = NULL;
        }
        if (slot->sad_host) {
            vmaf_cuda_buffer_host_free(fex->cu_state, slot->sad_host);
            slot->sad_host = NULL;
        }
        if (slot->consumed) {
            CHECK_CUDA(cu_f, cuEventDestroy(slot->consumed));
            slot->consumed = NULL;
        }
        if (slot->kernel_done) {
            CHECK_CUDA(cu_f, cuEventDestroy(slot->kernel_done));
            slot->kernel_done = NULL;
        }
        if (slot->stream) {
            CHECK_CUDA(cu_f, cuStreamDestroy(slot->stream));
            slot->stream = NULL;
        }
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    return ret ? ret : -ENOMEM;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;
    for (unsigned i = 0; i < MOTION_CUDA_HOST_SLOTS; i++) {
        CHECK_CUDA(cu_f, cuStreamSynchronize(s->slots[i].stream));
    }

    if (s->index > 0) {
        ret = vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion2_score",
                s->score, s->index);
    }

    return (ret < 0) ? ret : !ret;
}

static inline double normalize_and_scale_sad(uint64_t sad,
        unsigned w, unsigned h)
{
    return (float) (sad / 256.) / (w * h);
}


static int write_scores(write_score_parameters_moco* params)
{
    MotionStateCuda *s = params->s;
    VmafFeatureCollector *feature_collector = params->feature_collector;

    double score_prev = s->score;

    s->score = normalize_and_scale_sad(*params->sad_host, params->w, params->h);
    int err = 0;
    if (s->debug) {
        err |= vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion_score",
                s->score, params->index);
    }
    if (err) return err;

    if (params->index == 1)
        return 0;

    err = vmaf_feature_collector_append(feature_collector,
            "VMAF_integer_feature_motion2_score",
            score_prev < s->score ? score_prev : s->score, params->index - 1);
    return err;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int err = 0;
    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    // Acquire a slot. Wait for the prior use of this slot's host state to
    // finish; first K frames pass through instantly (unrecorded events).
    const unsigned slot_idx = s->next_slot;
    s->next_slot = (s->next_slot + 1) % MOTION_CUDA_HOST_SLOTS;
    MotionCudaSlot *slot = &s->slots[slot_idx];
    CHECK_CUDA(cu_f, cuEventSynchronize(slot->consumed));

    s->index = index;
    const unsigned src_blurred_idx = (index + 0) % 2;
    const unsigned prev_blurred_idx = (index + 1) % 2;

    // The kernel reads blur[prev] and writes blur[curr], so frame N's kernel
    // must complete before frame N+1's kernel runs (both buffers alias across
    // frames). Chain via the prior slot's kernel_done event.
    const unsigned prev_slot_idx =
        (slot_idx + MOTION_CUDA_HOST_SLOTS - 1) % MOTION_CUDA_HOST_SLOTS;
    MotionCudaSlot *prev_slot = &s->slots[prev_slot_idx];
    CHECK_CUDA(cu_f, cuStreamWaitEvent(slot->stream, prev_slot->kernel_done,
                CU_EVENT_WAIT_DEFAULT));

    // Wait for the ref picture to be ready on this slot's stream.
    CHECK_CUDA(cu_f, cuStreamWaitEvent(slot->stream,
                vmaf_cuda_picture_get_ready_event(ref_pic), CU_EVENT_WAIT_DEFAULT));

    // Reset per-slot SAD.
    CHECK_CUDA(cu_f, cuMemsetD8Async(slot->sad->data, 0, sizeof(uint64_t),
                slot->stream));

    // Compute motion score on slot->stream.
    s->calculate_motion_score(ref_pic, s->blur[src_blurred_idx],
            s->blur[prev_blurred_idx], slot->sad,
            ref_pic->w[0], ref_pic->h[0], ref_pic->stride[0],
            sizeof(uint16_t) * ref_pic->w[0], ref_pic->bpc,
            s->funcbpc8, s->funcbpc16, cu_f, slot->stream);
    CHECK_CUDA(cu_f, cuEventRecord(slot->kernel_done, slot->stream));

    if (index == 0) {
        err = vmaf_feature_collector_append(feature_collector,
                "VMAF_integer_feature_motion2_score",
                0., index);
        if (s->debug) {
            err |= vmaf_feature_collector_append(feature_collector,
                    "VMAF_integer_feature_motion_score",
                    0., index);
        }
        // Slot had no D2H / host_func; mark consumed now so it can recycle.
        CHECK_CUDA(cu_f, cuEventRecord(slot->consumed, slot->stream));
        return err;
    }

    // D2H of 8-byte SAD + host-fn callback, all on this slot's stream.
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(slot->sad_host,
                (CUdeviceptr)slot->sad->data, sizeof(*slot->sad_host),
                slot->stream));

    slot->cpu_param.feature_collector = feature_collector;
    slot->cpu_param.h = ref_pic->h[0];
    slot->cpu_param.w = ref_pic->w[0];
    slot->cpu_param.index = index;
    CHECK_CUDA(cu_f, cuLaunchHostFunc(slot->stream, (CUhostFn*)write_scores,
                &slot->cpu_param));
    CHECK_CUDA(cu_f, cuEventRecord(slot->consumed, slot->stream));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    MotionStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = 0;

    for (unsigned i = 0; i < MOTION_CUDA_HOST_SLOTS; i++) {
        MotionCudaSlot *slot = &s->slots[i];
        if (slot->stream) {
            CHECK_CUDA(cu_f, cuStreamSynchronize(slot->stream));
        }
    }

    if (s->blur[0]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[0]);
        free(s->blur[0]);
    }
    if (s->blur[1]) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->blur[1]);
        free(s->blur[1]);
    }
    for (unsigned i = 0; i < MOTION_CUDA_HOST_SLOTS; i++) {
        MotionCudaSlot *slot = &s->slots[i];
        if (slot->sad) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, slot->sad);
            free(slot->sad);
        }
        if (slot->sad_host) {
            ret |= vmaf_cuda_buffer_host_free(fex->cu_state, slot->sad_host);
        }
        if (slot->kernel_done) {
            CHECK_CUDA(cu_f, cuEventDestroy(slot->kernel_done));
        }
        if (slot->consumed) {
            CHECK_CUDA(cu_f, cuEventDestroy(slot->consumed));
        }
        if (slot->stream) {
            CHECK_CUDA(cu_f, cuStreamDestroy(slot->stream));
        }
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);

    return ret;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_score", "VMAF_integer_feature_motion2_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion_cuda = {
    .name = "motion_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(MotionStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_CUDA,
};
