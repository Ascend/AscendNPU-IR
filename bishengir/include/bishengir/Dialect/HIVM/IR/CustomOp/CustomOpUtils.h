//===- CustomOpUtils.h - Shared utilities for HIVM custom ops -------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_UTILS_H
#define BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {

/// Returns true when `op` is `hivm.hir.custom` or `hivm.hir.custom_macro`.
inline bool isCustomLikeOp(Operation *op) {
  return llvm::isa<CustomOp>(op) || llvm::isa<CustomMacroOp>(op);
}

/// Collect execution pipe(s) for a custom-like op.
SmallVector<PIPE> getCustomLikePipes(Operation *op);

/// Returns true when `operand` is a temp buffer of a custom-like op.
bool isCustomLikeTempBuffer(Operation *op, OpOperand *operand);

/// Invokes `fn` for each temp buffer operand of a custom-like op.
void forEachCustomLikeTempBuffer(Operation *op,
                                 llvm::function_ref<void(OpOperand &)> fn);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_CUSTOMOP_UTILS_H
