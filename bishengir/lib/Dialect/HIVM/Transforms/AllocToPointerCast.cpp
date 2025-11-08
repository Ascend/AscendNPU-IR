//===- AllocToPointerCast.cpp - convert memref.AllocOp to hivm.pointercastOp.//
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

#include "bishengir/Dialect/HIVM/Transforms/AllocToPointerCast.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {} // namespace

LogicalResult MemrefAllocaOpToPointerCastOpPattern::matchAndRewrite(
    memref::AllocOp op, PatternRewriter &rewriter) const {
  const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
  auto iter = buffer2Offsets.find(op.getResult());
  if (iter == buffer2Offsets.end()) {
    op.emitOpError() << "error: read before first write";
    return failure();
  }

  SmallVector<Value> addrs;
  for (auto &offset : iter->second) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }
  auto hivmPointerCastOp = rewriter.create<hivm::PointerCastOp>(
      op.getLoc(), currentMemRefType, ValueRange(addrs),
      ValueRange(op.getDynamicSizes()));
  rewriter.replaceOp(op, hivmPointerCastOp->getResults());
  return success();
}

LogicalResult UpdateWorkSpaceAllocaOpOffsetPattern::matchAndRewrite(
    bishengir::memref_ext::AllocWorkspaceOp op,
    PatternRewriter &rewriter) const {
  if (!op.getOffset().empty()) {
    return failure();
  }
  auto iter = buffer2Offsets.find(op.getResult());
  if (iter == buffer2Offsets.end()) {
    op.emitOpError() << "error: read before first write";
    return failure();
  }

  SmallVector<Value> argOffset;
  for (auto &offset : iter->second) {
    Value newOffset =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), offset)
            .getResult();
    argOffset.push_back(newOffset);
  }
  auto allocWorkspaceOp =
      rewriter.create<bishengir::memref_ext::AllocWorkspaceOp>(
          op.getLoc(), op->getResultTypes(), op.getWorkspaceArg(),
          op.getDynamicSize(), argOffset);
  rewriter.replaceOp(op, allocWorkspaceOp->getResults());
  return success();
}