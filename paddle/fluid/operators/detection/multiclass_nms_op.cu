/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

limitations under the License. */

#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/detection/bbox_util.cu.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void Select(const int num, const T *scores, T min_threshold,
                       int *flags) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    if (scores[i] > min_threshold) {
      flags[i] = 1;
    } else {
      flags[i] = 0;
    }
  }
}

template <typename T>
__global__ void GatherScoresBBoxesAndFillLabels(const int num, const int *index,
                                                const T *scores,
                                                const T *bboxes, T *out_scores,
                                                int *out_labels, T *out_bboxes,
                                                const int label_fill_value) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    int id = index[i];
    out_scores[i] = scores[id];
    out_labels[i] = label_fill_value;
    out_bboxes[i * 4 + 0] = bboxes[id * 4 + 0];
    out_bboxes[i * 4 + 1] = bboxes[id * 4 + 1];
    out_bboxes[i * 4 + 2] = bboxes[id * 4 + 2];
    out_bboxes[i * 4 + 3] = bboxes[id * 4 + 3];
  }
}

template <typename T>
__global__ void GatherLabelsBBoxes(const int num, const int *index,
                                   const T *bboxes, const int *labels,
                                   T *out_bboxes, int *out_labels) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    int id = index[i];
    out_labels[i] = labels[id];
    out_bboxes[i * 4 + 0] = bboxes[id * 4 + 0];
    out_bboxes[i * 4 + 1] = bboxes[id * 4 + 1];
    out_bboxes[i * 4 + 2] = bboxes[id * 4 + 2];
    out_bboxes[i * 4 + 3] = bboxes[id * 4 + 3];
  }
}

template <typename T>
__global__ void AssignOutput(const int num, const int *labels, const T *scores,
                             const T *bboxes, T *outputs) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    outputs[i * 6 + 0] = static_cast<T>(labels[i]);
    outputs[i * 6 + 1] = scores[i];
    outputs[i * 6 + 2] = bboxes[i * 4 + 0];
    outputs[i * 6 + 3] = bboxes[i * 4 + 1];
    outputs[i * 6 + 4] = bboxes[i * 4 + 2];
    outputs[i * 6 + 5] = bboxes[i * 4 + 3];
  }
}

template <typename T>
__global__ void KeMultiClassNMS(const T *scores,  // N * C * M
                                const T *bboxes,  // N * M * 4
                                const T *work_space,
                                const int class_num,  // C
                                const int box_num,    // M
                                T) {
  // --- Shared memory allocation
  __shared__ float values[BLOCK_THREADS * ITEMS_PER_THREAD];
  __shared__ int keys[BLOCK_THREADS * ITEMS_PER_THREAD];
  // --- Specialize BlockStore and BlockRadixSort collective types
  typedef cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD, T>
      BlockRadixSortT;
  // --- Allocate type-safe, repurposable shared memory for collectives
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;

  for (int i = 0; i < class_num++ i) {
    // 1. sort
    int block_offset = bidx * (BLOCK_THREADS * ITEMS_PER_THREAD);
    // --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
      values[tidx * ITEMS_PER_THREAD + k] =
          d_values[block_offset + tidx * ITEMS_PER_THREAD + k];
      keys[tidx * ITEMS_PER_THREAD + k] =
          d_keys[block_offset + tidx * ITEMS_PER_THREAD + k];
    }
    __syncthreads();
    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage)
        .SortBlockedToStriped(
            *static_cast<int(*)[ITEMS_PER_THREAD]>(
                static_cast<void *>(keys + (tidx * ITEMS_PER_THREAD))),
            *static_cast<float(*)[ITEMS_PER_THREAD]>(
                static_cast<void *>(values + (tidx * ITEMS_PER_THREAD))));
    __syncthreads();

    // --- Write data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
      d_values_result[block_offset + tidx * ITEMS_PER_THREAD + k] =
          values[tidx * ITEMS_PER_THREAD + k];
      d_keys_result[block_offset + tidx * ITEMS_PER_THREAD + k] =
          keys[tidx * ITEMS_PER_THREAD + k];
    }
  }
}

template <typename T>
class MultiClassNMSCUDAKernel : public framework::OpKernel<T> {
 public:
  void MultiClassNMS(const framework::ExecutionContext &ctx,
                     const Tensor &scores, const Tensor &bboxes,
                     Tensor *out_scores, Tensor *out_labels,
                     Tensor *out_bboxes) const {
    int background_label = ctx.Attr<int>("background_label");
    int nms_top_k = ctx.Attr<int>("nms_top_k");
    int keep_top_k = ctx.Attr<int>("keep_top_k");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    int64_t class_num = scores.dims()[0];
    int64_t predict_dim = scores.dims()[1];

    Tensor scores_sort, index_sort;
    Tensor flags, sum_out, sum_temp_storage;
    Tensor pre_bboxes;
    Tensor keep_index;
    Tensor all_cls_scores, all_cls_labels, all_cls_bboxes;

    int max_num = class_num * nms_top_k;
    all_cls_scores.mutable_data<T>({max_num, 1}, ctx.GetPlace());
    all_cls_bboxes.mutable_data<T>({max_num, 4}, ctx.GetPlace());
    all_cls_labels.mutable_data<int>({max_num, 1}, ctx.GetPlace());

    int *d_sum = sum_out.mutable_data<int>(ctx.GetPlace(),
                                           memory::Allocator::kScratchpad, 1);
    const auto gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());

    int keep_num = 0;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;
      Tensor score = scores.Slice(c, c + 1);
      // 1. sort and pre nms
      SortDescending<T>(dev_ctx, score, &scores_sort, &index_sort);
      // filter scores less than score_threshold
      int *flags_data = flags.mutable_data<int>({predict_dim}, ctx.GetPlace());
      Select<<<GET_BLOCKS(predict_dim), CUDA_NUM_THREADS>>>(
          predict_dim, score.data<T>(), score_threshold, flags_data);
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, flags_data, d_sum,
                             predict_dim);
      sum_temp_storage.mutable_data<int8_t>(
          ctx.GetPlace(), memory::Allocator::kScratchpad, temp_storage_bytes);
      void *d_temp_storage =
          reinterpret_cast<void *>(sum_temp_storage.data<int8_t>());
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, flags_data,
                             d_sum, predict_dim);
      int pre_num;
      memory::Copy(platform::CPUPlace(), &pre_num, gpu_place, d_sum,
                   sizeof(int), 0);
      pre_num = std::min(pre_num, nms_top_k);
      if (pre_num == 0) continue;

      scores_sort.Resize({pre_num, 1});
      index_sort.Resize({pre_num, 1});
      pre_bboxes.mutable_data<T>({pre_num, 4}, ctx.GetPlace());
      GPUGather<T>(dev_ctx, bboxes, index_sort, &pre_bboxes);
      // dev_ctx.Wait();

      // 2. nms
      const int col_blocks = DIVUP(pre_num, kThreadsPerBlock);
      dim3 blocks(DIVUP(pre_num, kThreadsPerBlock),
                  DIVUP(pre_num, kThreadsPerBlock));
      dim3 threads(kThreadsPerBlock);
      framework::Vector<uint64_t> mask(pre_num * col_blocks);
      NMSKernel<<<blocks, threads, 0, dev_ctx.stream()>>>(
          pre_num, nms_threshold, pre_bboxes.data<T>(),
          mask.CUDAMutableData(gpu_place), true);

      std::vector<uint64_t> remv(col_blocks);
      memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

      std::vector<int> keep_vec;
      int num_to_keep = 0;
      for (int i = 0; i < pre_num; i++) {
        int nblock = i / kThreadsPerBlock;
        int inblock = i % kThreadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
          ++num_to_keep;
          keep_vec.push_back(i);
          uint64_t *p = &mask[0] + i * col_blocks;
          for (int j = nblock; j < col_blocks; j++) {
            remv[j] |= p[j];
          }
        }
      }
      int *keep = keep_index.mutable_data<int>({num_to_keep}, ctx.GetPlace());
      memory::Copy(gpu_place, keep, platform::CPUPlace(), keep_vec.data(),
                   sizeof(int) * num_to_keep, 0);

      GatherScoresBBoxesAndFillLabels<T><<<
          GET_BLOCKS(num_to_keep), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          num_to_keep, keep, scores_sort.data<T>(), pre_bboxes.data<T>(),
          all_cls_scores.data<T>() + keep_num,
          all_cls_labels.data<int>() + keep_num,
          all_cls_bboxes.data<T>() + keep_num * 4, c);
      keep_num += num_to_keep;
    }

    all_cls_scores.Resize({keep_num});
    all_cls_bboxes.Resize({keep_num, 4});
    all_cls_labels.Resize({keep_num});
    if (keep_num == 0) return;

    if (keep_top_k > -1 && keep_num > keep_top_k) {
      keep_num = keep_top_k;
      // 3. post nms
      SortDescending<T>(dev_ctx, all_cls_scores, out_scores, &index_sort);
      index_sort.Resize({keep_num});
      out_scores->Resize({keep_num});
      out_labels->mutable_data<int>({keep_num}, ctx.GetPlace());
      out_bboxes->mutable_data<T>({keep_num, 4}, ctx.GetPlace());
      GatherLabelsBBoxes<
          T><<<GET_BLOCKS(keep_num), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          keep_num, index_sort.data<int>(), all_cls_bboxes.data<T>(),
          all_cls_labels.data<int>(), out_bboxes->data<T>(),
          out_labels->data<int>());
    } else {
      out_scores->mutable_data<T>({keep_num}, ctx.GetPlace());
      out_bboxes->mutable_data<T>({keep_num, 4}, ctx.GetPlace());
      out_labels->mutable_data<T>({keep_num}, ctx.GetPlace());

      TensorCopySync(all_cls_scores, ctx.GetPlace(), out_scores);
      TensorCopySync(all_cls_bboxes, ctx.GetPlace(), out_bboxes);
      TensorCopySync(all_cls_labels, ctx.GetPlace(), out_labels);
    }
  }

  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *boxes = ctx.Input<Tensor>("BBoxes");
    auto *scores = ctx.Input<Tensor>("Scores");
    auto *outs = ctx.Output<LoDTensor>("Out");

    auto score_dims = scores->dims();

    int batch_size = score_dims[0];
    int class_num = score_dims[1];
    int predict_dim = score_dims[2];
    int box_dim = boxes->dims()[2];
    int out_dim = boxes->dims()[2] + 2;

    int max_num = scores->numel();
    int keep_top_k = ctx.Attr<int>("keep_top_k");
    if (keep_top_k > -1) {
      max_num = batch_size * keep_top_k;
    }
    outs->mutable_data<T>({max_num, 6}, ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    std::vector<size_t> batch_starts = {0};
    Tensor out_scores, out_bboxes, out_labels;
    for (int64_t i = 0; i < batch_size; ++i) {
      Tensor ins_score = scores->Slice(i, i + 1);
      ins_score.Resize({class_num, predict_dim});
      Tensor ins_boxes = boxes->Slice(i, i + 1);
      ins_boxes.Resize({predict_dim, box_dim});
      MultiClassNMS(ctx, ins_score, ins_boxes, &out_scores, &out_labels,
                    &out_bboxes);
      int num_det = out_scores.numel();
      if (num_det) {
        AssignOutput<<<GET_BLOCKS(num_det), CUDA_NUM_THREADS, 0,
                       dev_ctx.stream()>>>(
            num_det, out_labels.data<int>(), out_scores.data<T>(),
            out_bboxes.data<T>(), outs->data<T>() + batch_starts.back() * 6);
      }
      batch_starts.push_back(batch_starts.back() + num_det);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      outs->mutable_data<T>({1}, ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, T> setter;
      setter(dev_ctx, outs, static_cast<T>(-1));
    }
    PADDLE_ENFORCE_LE(batch_starts.back(), max_num);
    outs->Resize({static_cast<int>(batch_starts.back()), 6});
    framework::LoD lod;
    lod.emplace_back(batch_starts);
    outs->set_lod(lod);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(multiclass_nms, ops::MultiClassNMSCUDAKernel<float>);
