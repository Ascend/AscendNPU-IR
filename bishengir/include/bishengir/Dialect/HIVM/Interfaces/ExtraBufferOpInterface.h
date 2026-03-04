//===- ExtraBufferOpInterface.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

namespace hivm {
namespace util {

constexpr static unsigned int REDUCE_DEFAULT_FACTOR = 1;

enum class BufferSizeUnit {
  ELEMENT, // the buffer size is in unit of element
  FACTOR   // the buffer size is a factor of the input tensor/buffer size
};

/// Get extra buffer size needed for VBrcOp.
///
/// \param op `hivm.vbrc` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor of destination tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForBroadcastOp(Operation *op,
                                                        BufferSizeUnit unit);

/// Get extra buffer size needed for VReduceOp.
///
/// \param op `hivm.vreduce` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor reduction op's tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit);

std::optional<int64_t>
getExtraBufferSizeForReduceOpSingleDim(Operation *op, BufferSizeUnit unit,
                                       int64_t reductionDim);

std::optional<int64_t> refineReduceExtraBufferSize(ShapedType srcType,
                                                   int64_t srcAllocTotalSize,
                                                   int64_t reductionDim);

} // namespace util
} // namespace hivm
} // namespace mlir

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_H
