//===- CustomOpExtraBuffer.cpp - Custom op extra buffer interface ---------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
Value allocExtraBuffer(Operation *op, const SmallVector<int64_t> &bufSize,
                       Type elemType) {
  mlir::IRRewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(op);
  return rewriter.create<memref::AllocOp>(op->getLoc(),
                                          MemRefType::get(bufSize, elemType));
}

template <typename CustomOpT>
LogicalResult allocExtraBuffersForCustomOp(CustomOpT op) {
  if (!llvm::to_vector(op.getTempBuffers()).empty()) {
    op.emitWarning("already has extra temp buffers");
    return success();
  }

  SmallVector<Value> buffs;
  for (const auto &[type, size] : op.getExtraBuffersInfo()) {
    Value extraBuf = allocExtraBuffer(op.getOperation(), {size}, type);
    buffs.push_back(extraBuf);
  }

  op.getTempBuffersMutable().assign(buffs);
  return success();
}
} // namespace

LogicalResult CustomOp::allocExtraBuffersIfPossible() {
  return allocExtraBuffersForCustomOp(*this);
}

LogicalResult CustomMacroOp::allocExtraBuffersIfPossible() {
  return allocExtraBuffersForCustomOp(*this);
}

OperandRange CustomOp::getExtraBuffers() { return getTempBuffersMutable(); }

OperandRange CustomMacroOp::getExtraBuffers() {
  return getTempBuffersMutable();
}
