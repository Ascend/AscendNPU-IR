//===- ExtraBuffer.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H
#define BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hfusion {
namespace util {

constexpr static unsigned int REDUCE_DEFAULT_FACTOR = 1;

enum class BufferSizeUnit {
  ELEMENT, // the buffer size is in unit of element
  FACTOR   // the buffer size is a factor of the input tensor/buffer size
};

/// Get extra buffer size needed for VBrcOp.
///
/// \param op `linalg.broadcast` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor of destination tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForBroadcastOp(Operation *op,
                                                        BufferSizeUnit unit);

/// Get extra buffer size needed for VReduceOp.
///
/// \param op `linalg.reduce` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor reduction op's tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit);

} // namespace util
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H
