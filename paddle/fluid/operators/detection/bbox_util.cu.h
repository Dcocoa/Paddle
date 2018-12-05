/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <paddle/fluid/memory/allocation/allocator.h>
#include <algorithm>
#include "cub/cub.cuh"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

#define CUDA_NUM_THREADS 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

int const kThreadsPerBlock = sizeof(uint64_t) * 8;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

struct RangeInitFunctor {
  int start_;
  int delta_;
  int *out_;
  __device__ void operator()(size_t i) { out_[i] = start_ + i * delta_; }
};

template <typename T>
inline void SortDescending(const platform::CUDADeviceContext &ctx,
                           const framework::Tensor &value,
                           framework::Tensor *value_out,
                           framework::Tensor *index_out) {
  int num = static_cast<int>(value.numel());
  framework::Tensor index_in_t;
  int *idx_in = index_in_t.mutable_data<int>({num}, ctx.GetPlace());
  platform::ForRange<platform::CUDADeviceContext> for_range(ctx, num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  int *idx_out = index_out->mutable_data<int>({num}, ctx.GetPlace());

  const T *keys_in = value.data<T>();
  T *keys_out = value_out->mutable_data<T>({num}, ctx.GetPlace());

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      nullptr, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out, num);
  // Allocate temporary storage
  auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  auto d_temp_storage =
      memory::Alloc(place, temp_storage_bytes, memory::Allocator::kScratchpad);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      d_temp_storage->ptr(), temp_storage_bytes, keys_in, keys_out, idx_in,
      idx_out, num);
}

__device__ inline float IoU(const float *a, const float *b, bool normalized) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float norm = normalized ? 0. : 1.;
  float width = max(right - left + norm, 0.f),
        height = max(bottom - top + norm, 0.f);
  float inter_s = width * height;
  float s_a = (a[2] - a[0] + norm) * (a[3] - a[1] + norm);
  float s_b = (b[2] - b[0] + norm) * (b[3] - b[1] + norm);
  return inter_s / (s_a + s_b - inter_s);
}

__global__ inline void NMSKernel(const int n_boxes,
                                 const float nms_overlap_thresh,
                                 const float *dev_boxes, uint64_t *dev_mask,
                                 const bool normalized) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  __shared__ float block_boxes[kThreadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 4, normalized) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, kThreadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

}  // namespace operators
}  // namespace paddle
