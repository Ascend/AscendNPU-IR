//===- BufferUtils.h ------------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_UTILS_BUFFERUTILS_H
#define BISHENGIR_DIALECT_HFUSION_UTILS_BUFFERUTILS_H

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace utils {

// Value comparator for std::map
inline bool isLessValue(const Value &a, const Value &b) {
  return a.getImpl() < b.getImpl();
}

struct ValueComparator {
  bool operator()(const Value &a, const Value &b) const {
    return isLessValue(a, b);
  }
};

struct BufferAnalysisOptions {
  using MultiBufferMap = std::map<Value, size_t, ValueComparator>;

  /// Mapping from `value` to the multi-buffer count.
  MultiBufferMap multiBufferCount;
  /// If enabled, the buffer used by DMA operations will not be reused by Vector
  /// operations.
  bool enableDmaOpt{false};
  bool printLiveRange{false};
};

int64_t countMaxBuffer(func::FuncOp func,
                       const BufferAnalysisOptions &options = {});

} // namespace utils
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_BUFFERUTILS_H
