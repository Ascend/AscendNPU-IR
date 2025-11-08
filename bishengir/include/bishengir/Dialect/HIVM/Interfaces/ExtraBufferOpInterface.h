//===- ExtraBufferOpInterface.h -------------------------------------------===//
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
