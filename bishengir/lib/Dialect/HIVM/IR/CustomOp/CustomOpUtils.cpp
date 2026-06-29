//===- CustomOpUtils.cpp - Shared utilities for HIVM custom ops -----------===//
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

#include "bishengir/Dialect/HIVM/IR/CustomOp/CustomOpUtils.h"

#include "llvm/ADT/TypeSwitch.h"

#include <type_traits>

namespace mlir {
namespace hivm {

namespace {
template <typename CustomOpT,
          std::enable_if_t<std::is_same_v<CustomOpT, CustomOp>, int> = 0>
SmallVector<PIPE> getCustomLikePipesFor(CustomOpT op) {
  return {op.getPipe()};
}

template <typename CustomOpT,
          std::enable_if_t<std::is_same_v<CustomOpT, CustomMacroOp>, int> = 0>
SmallVector<PIPE> getCustomLikePipesFor(CustomOpT op) {
  return {op.getInPipe(), op.getOutPipe()};
}

template <typename CustomOpT>
bool isCustomLikeTempBufferFor(CustomOpT op, OpOperand *operand) {
  for (auto &tempBuffer : op.getTempBuffersMutable())
    if (&tempBuffer == operand)
      return true;
  return false;
}

template <typename CustomOpT>
void forEachCustomLikeTempBufferFor(CustomOpT op,
                                    llvm::function_ref<void(OpOperand &)> fn) {
  for (auto &tempBuffer : op.getTempBuffersMutable())
    fn(tempBuffer);
}
} // namespace

SmallVector<PIPE> getCustomLikePipes(Operation *op) {
  return TypeSwitch<Operation *, SmallVector<PIPE>>(op)
      .Case<CustomOp>(
          [](CustomOp customOp) { return getCustomLikePipesFor(customOp); })
      .Case<CustomMacroOp>(
          [](CustomMacroOp macroOp) { return getCustomLikePipesFor(macroOp); })
      .Default([](Operation *) { return SmallVector<PIPE>{}; });
}

bool isCustomLikeTempBuffer(Operation *op, OpOperand *operand) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<CustomOp>([=](CustomOp customOp) {
        return isCustomLikeTempBufferFor(customOp, operand);
      })
      .Case<CustomMacroOp>([=](CustomMacroOp macroOp) {
        return isCustomLikeTempBufferFor(macroOp, operand);
      })
      .Default([](Operation *) { return false; });
}

void forEachCustomLikeTempBuffer(Operation *op,
                                 llvm::function_ref<void(OpOperand &)> fn) {
  TypeSwitch<Operation *>(op)
      .Case<CustomOp>([=](CustomOp customOp) {
        forEachCustomLikeTempBufferFor(customOp, fn);
      })
      .Case<CustomMacroOp>([=](CustomMacroOp macroOp) {
        forEachCustomLikeTempBufferFor(macroOp, fn);
      });
}

} // namespace hivm
} // namespace mlir
