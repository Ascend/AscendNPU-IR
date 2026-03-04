//===- BufferUtils.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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