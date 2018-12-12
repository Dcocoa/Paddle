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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS) __global__
    void KeSortByScores(const T* in_scores,  // N * C * M
                        const int background_label,
                        const int class_num,  // C
                        const int box_num,    // M
                        int* sort_inds        // N * C * N
                        ) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;  // class id
  int bidy = blockIdx.y;  // batch id

  if (bidx == background_label) return;

  typedef cub::BlockRadixSort<T, BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockRadixSortT;
  __shared__ typename BlockRadixSortT::TempStorage temp_storage;

  T keys[ITEMS_PER_THREAD];
  int inds[ITEMS_PER_THREAD];

  int offset = bidy * class_num * box_num + bidx * box_num;
  const T* scores = in_scores + offset;
  for (int k = 0; k < ITEMS_PER_THREAD; k++) {
    int id = k * BLOCK_THREADS + tidx;
    T key = id < box_num ? scores[id] : -1.;
    keys[k] = key;
    inds[k] = id < box_num ? id : -1;
  }
  __syncthreads();
  BlockRadixSortT(temp_storage).SortDescendingBlockedToStriped(keys, inds);
  __syncthreads();

  // store global memory
  int* inds_space = sort_inds + offset;
  for (int k = 0; k < ITEMS_PER_THREAD; k++) {
    int id = k * BLOCK_THREADS + tidx;
    if (id < box_num) {
      inds_space[id] = inds[k];
    }
  }
}

// __launch_bounds__ (BLOCK_THREADS)
template <typename T,
          int BLOCK_THREADS>
__global__ void KeNMSPerClass(const T* in_scores,  // N * C * M
                              const T* in_bboxes, const int background_label,
                              const int class_num,  // C
                              const int box_num,    // M
                              const T pre_nms_thresh, const T nms_thresh,
                              const int nms_top_k,
                              int* sort_inds,      // N * C * N
                              unsigned char* mask  // N * C * N
                              ) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;  // class id
  int bidy = blockIdx.y;  // batch id

  if (bidx == background_label) return;
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int pre_nms_num;

  int offset = bidy * class_num * box_num + bidx * box_num;
  const T* scores = in_scores + offset;
  int* inds_space = sort_inds + offset;

  int flag = 0;
  for (int k = tidx; k < box_num; k += BLOCK_THREADS) {
    flag += scores[inds_space[k]] > pre_nms_thresh ? 1 : 0;
  }
  __syncthreads();
  int sum_num = BlockReduce(temp_storage).Reduce(flag, cub::Sum());
  __syncthreads();

  if (tidx == 0) {
    pre_nms_num = sum_num;
    if (nms_top_k > -1 && nms_top_k < pre_nms_num) {
      pre_nms_num = nms_top_k;
    }
  }
  __syncthreads();
  for (int j = threadIdx.x; j < box_num; j += BLOCK_THREADS) {
    if (j < pre_nms_num) {
      mask[offset + j] = 1;
    } else {
      mask[offset + j] = 0;
    }
  }
  __syncthreads();

  // nms
  const T* bboxes = in_bboxes + bidy * box_num * 4;
  int col = 0;
  while (col < pre_nms_num - 1) {
    for (int i = tidx; i < pre_nms_num - 1; i += BLOCK_THREADS) {
      if (i >= col) {
        T iou = IoU(bboxes + 4 * inds_space[col],
                    bboxes + 4 * inds_space[i + 1], true);
        mask[offset + i + 1] *= (iou > nms_thresh) ? 0 : 1;
      }
    }
    __syncthreads();
    ++col;
    // 1 is surviving and 0 is removing
    while ((col < pre_nms_num - 1) && (mask[offset + col] == 0)) {
      ++col;
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
// __launch_bounds__ (BLOCK_THREADS)
__global__ void KeMultiClassNMSPost(const T* in_scores,  // N * C * M
                                    const T* in_bboxes,  // N * M * 4
                                    const int background_label,
                                    const int class_num,  // C
                                    const int box_num,    // M
                                    const int nms_top_k, const int keep_top_k,
                                    const int max_out_len,
                                    int* sort_inds,           // N * C * N
                                    unsigned char* nms_mask,  // N * C * N
                                    T* outs, int* outs_num) {
  typedef cub::BlockRadixSort<T, BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockRadixSortT;
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduce;
  __shared__ union {
    typename BlockReduce::TempStorage reduce;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;
  __shared__ int keep_nms_num;

  T keys[ITEMS_PER_THREAD];
  int inds[ITEMS_PER_THREAD];

  int bidx = blockIdx.x;  // batch id
  int tidx = threadIdx.x;

  // compute total surviving number
  int cnt = 0;
  for (int i = tidx; i < class_num * box_num; i += BLOCK_THREADS) {
    int tmp_cnt =
        (i / box_num) == background_label
            ? 0
            : static_cast<int>(nms_mask[bidx * class_num * box_num + i]);
    cnt += tmp_cnt;
  }
  __syncthreads();
  int surviving_nms_num =
      BlockReduce(temp_storage.reduce).Reduce(cnt, cub::Sum());
  if (tidx == 0) {
    keep_nms_num = surviving_nms_num;
    if (keep_top_k > -1 && keep_top_k < surviving_nms_num) {
      keep_nms_num = keep_top_k;
    }
    outs_num[bidx] = keep_nms_num;
  }
  __syncthreads();

  int min_num = nms_top_k < box_num ? nms_top_k : box_num;
  for (int c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    cnt = bidx * class_num * box_num + c * box_num;
    const T* scores = in_scores + cnt;
    unsigned char* mask = nms_mask + cnt;
    // sort
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
      int id = k * BLOCK_THREADS + tidx;
      int flg = id < min_num ? mask[id] : 0;
      inds[k] = flg ? sort_inds[cnt + id] : -1;
      keys[k] = flg ? scores[sort_inds[cnt + id]] : 0.;
    }
    __syncthreads();
    BlockRadixSortT(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, inds);
    __syncthreads();
    if (!(background_label == 0 && class_num == 2)) {
      int min_num2 = keep_nms_num < min_num ? keep_nms_num : min_num;
      for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int id = k * BLOCK_THREADS + tidx;
        if (id < min_num) {
          sort_inds[cnt + id] = id < min_num2 ? inds[k] : -1;
        }
      }
      __syncthreads();
    }
  }

  const T* bboxes = in_bboxes + bidx * box_num * 4;
  T* out = outs + bidx * max_out_len * 6;

  // post process: sort and pruning
  if (background_label == 0 && class_num == 2) {
    // write to output
    for (int i = tidx; i < keep_nms_num; i += BLOCK_THREADS) {
      int k = i / BLOCK_THREADS;
      int id = inds[k];
      out[i * 6 + 0] = 1;
      out[i * 6 + 1] = keys[k];
      out[i * 6 + 2] = bboxes[id * 4 + 0];
      out[i * 6 + 3] = bboxes[id * 4 + 1];
      out[i * 6 + 4] = bboxes[id * 4 + 2];
      out[i * 6 + 5] = bboxes[id * 4 + 3];
    }
  } else {
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
      keys[k] = 0.;
      inds[k] = -1;
    }
    __syncthreads();
    int offset = bidx * class_num * box_num;
    int min_num2 = keep_nms_num < min_num ? keep_nms_num : min_num;
    for (int k = tidx; k < 2 * keep_nms_num; k += BLOCK_THREADS) {
      int c = k < keep_nms_num ? 0 : 1;
      if (c == background_label) continue;

      int i = k < keep_nms_num ? k : k - keep_nms_num;
      if (i >= min_num2) continue;
      int id = sort_inds[offset + c * box_num + i];
      if (id != -1) {
        int kk = k / BLOCK_THREADS;
        keys[kk] = in_scores[offset + c * box_num + id];
        inds[kk] = c * box_num + id;
      }
    }

    __syncthreads();
    BlockRadixSortT(temp_storage.sort)
        .SortDescendingBlockedToStriped(keys, inds);
    __syncthreads();
    for (int c = 2; c < class_num; ++c) {
      if (c == background_label) continue;
      for (int k = tidx; k < 2 * keep_nms_num; k += BLOCK_THREADS) {
        if (k < keep_nms_num) continue;
        if ((k - keep_nms_num) >= min_num2) break;
        int id = sort_inds[offset + c * box_num + k - keep_nms_num];
        if (id == -1) continue;
        int kk = k / BLOCK_THREADS;
        keys[kk] = in_scores[offset + c * box_num + id];
        inds[kk] = c * box_num + id;
      }
      __syncthreads();
      BlockRadixSortT(temp_storage.sort)
          .SortDescendingBlockedToStriped(keys, inds);
      __syncthreads();
    }

    // write to output
    for (int i = tidx; i < keep_nms_num; i += BLOCK_THREADS) {
      int k = i / BLOCK_THREADS;
      int box_id = (inds[k] % box_num) * 4;
      out[i * 6 + 0] = inds[k] / box_num;
      out[i * 6 + 1] = keys[k];
      out[i * 6 + 2] = bboxes[box_id + 0];
      out[i * 6 + 3] = bboxes[box_id + 1];
      out[i * 6 + 4] = bboxes[box_id + 2];
      out[i * 6 + 5] = bboxes[box_id + 3];
    }
  }
}

template <typename T, int BLOCK_THREADS>
__global__ void KeShifftBatchData(T* outs, const int* outs_num,
                                  const int in_stride, const int batch_size) {
  __shared__ T val[BLOCK_THREADS];
  int dst_offset = outs_num[0] * 6;
  int tidx = threadIdx.x;
  for (int b = 1; b < batch_size; ++b) {
    int in_offset = b * in_stride * 6;
    for (int i = tidx; i < outs_num[b] * 6; ++i) {
      val[tidx] = outs[in_offset + i];
      __syncthreads();
      outs[dst_offset + i] = val[tidx];
    }
    dst_offset += outs_num[b] * 6;
  }
}

template <typename T>
class MultiClassNMSCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bboxes = ctx.Input<Tensor>("BBoxes");
    auto* scores = ctx.Input<Tensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");

    int bg_label = ctx.Attr<int>("background_label");
    int nms_top_k = ctx.Attr<int>("nms_top_k");
    int keep_top_k = ctx.Attr<int>("keep_top_k");
    T nms_thresh = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T score_thresh = static_cast<T>(ctx.Attr<float>("score_threshold"));

    auto score_dims = scores->dims();
    int batch_size = score_dims[0];
    int class_num = score_dims[1];
    int box_num = score_dims[2];

    int box_dim = bboxes->dims()[2];
    int out_dim = bboxes->dims()[2] + 2;

    int scores_size = scores->numel();

    int max_num = (nms_top_k > -1 && nms_top_k < box_num) ? nms_top_k : box_num;
    max_num = (bg_label > -1 && bg_label < class_num)
                  ? (class_num - 1) * max_num
                  : class_num * max_num;
    max_num = (keep_top_k > -1 && keep_top_k < max_num) ? keep_top_k : max_num;

    outs->mutable_data<T>({max_num * batch_size, 6}, ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());

    int work_space_bytes = (sizeof(int) + sizeof(unsigned char)) * scores_size;
    work_space_bytes += batch_size * sizeof(int);
    // void* work_space = memory::Alloc(place,
    //     work_space_bytes, memory::Allocator::kScratchpad)->ptr();
    void* work_space = memory::Alloc(place, work_space_bytes)->ptr();
    unsigned char* mask = reinterpret_cast<unsigned char*>(
        work_space + sizeof(int) * (scores_size + batch_size));
    int* sort_inds = reinterpret_cast<int*>(work_space);
    int* outs_num = sort_inds + scores_size;

    const int kThreads = 1024;
    dim3 blocks(kThreads, 1);
    dim3 grids(class_num, batch_size);

// #define EXPAND_CASE_BAES(dim, num, ...) \
//   }                                     \
//   else if ((dim) < kThreads * (num)) {  \
//     const int kItermsPerThread = (num); \
//     __VA_ARGS__;  // NOLINT

#define EXPAND_CASE(dim, ...)                                         \
  if ((dim) < kThreads) {                                             \
    const int kItermsPerThread = 1;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 2) {                                  \
    const int kItermsPerThread = 2;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 3) {                                  \
    const int kItermsPerThread = 3;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 4) {                                  \
    const int kItermsPerThread = 4;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 5) {                                  \
    const int kItermsPerThread = 5;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 6) {                                  \
    const int kItermsPerThread = 6;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 7) {                                  \
    const int kItermsPerThread = 7;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 8) {                                  \
    const int kItermsPerThread = 8;                                   \
    __VA_ARGS__;                                                      \
  } else if ((dim) < kThreads * 9) {                                  \
    const int kItermsPerThread = 9;                                   \
    __VA_ARGS__;                                                      \
  } else {                                                            \
    PADDLE_THROW("The number (%d) is too large, unspported.", (dim)); \
  }

    EXPAND_CASE(
        box_num,
        KeSortByScores<
            T, kThreads,
            kItermsPerThread><<<grids, blocks, 0, dev_ctx.stream()>>>(
            scores->data<T>(), bg_label, class_num, box_num, sort_inds));

    KeNMSPerClass<T, kThreads><<<grids, blocks, 0, dev_ctx.stream()>>>(
        scores->data<T>(), bboxes->data<T>(), bg_label, class_num, box_num,
        score_thresh, nms_thresh, nms_top_k, sort_inds, mask);

    int max_sort_num = std::max(std::min(box_num, nms_top_k), 2 * keep_top_k);
    EXPAND_CASE(
        max_sort_num,
        KeMultiClassNMSPost<
            T, kThreads,
            kItermsPerThread><<<batch_size, kThreads, 0, dev_ctx.stream()>>>(
            scores->data<T>(), bboxes->data<T>(), bg_label, class_num, box_num,
            nms_top_k, keep_top_k, max_num, sort_inds, mask, outs->data<T>(),
            outs_num));

    int* h_outs_num = reinterpret_cast<int*>(
        memory::Alloc(platform::CPUPlace(), batch_size * sizeof(int),
                      memory::Allocator::kScratchpad)
            ->ptr());
    memory::Copy(platform::CPUPlace(), h_outs_num, place, outs_num,
                 batch_size * sizeof(int), dev_ctx.stream());

    std::vector<size_t> batch_starts = {0};
    for (int i = 0; i < batch_size; ++i) {
      batch_starts.push_back(batch_starts.back() + h_outs_num[i]);
    }
    int num_kept = batch_starts.back();

    if (num_kept == 0) {
      outs->mutable_data<T>({1}, ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, T> setter;
      setter(dev_ctx, outs, static_cast<T>(-1));
    } else if (batch_size == 1) {
      PADDLE_ENFORCE_LE(batch_starts.back(), max_num);
      outs->Resize({static_cast<int>(batch_starts.back()), 6});
    } else {
      // batch size > 1
      KeShifftBatchData<T, kThreads><<<1, kThreads>>>(outs->data<T>(), outs_num,
                                                      max_num, batch_size);
      outs->Resize({static_cast<int>(batch_starts.back()), 6});
    }
    framework::LoD lod;
    lod.emplace_back(batch_starts);
    outs->set_lod(lod);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(multiclass_nms, ops::MultiClassNMSCUDAKernel<float>);
