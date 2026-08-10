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

LogicalResult mlir::hivm::walkAllocToPointerCast(
    func::FuncOp funcOp,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets) {
  IRRewriter rewriter(funcOp);
  // Cache of already-created offset constants, keyed by the offset value. All
  // constants are created at the start of the function so that they dominate
  // every use, and an offset is materialized only once.
  DenseMap<uint64_t, Value> offset2Constant;
  auto result = funcOp.walk([&](memref::AllocOp op) {
    // Convert the alloc into an hivm.hir.pointer_cast bound to its planned
    // address(es). Fail with a diagnostic if the alloc is not covered by the
    // memory plan (read before first write).
    const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
    auto iter = buffer2Offsets.find(op.getResult());
    if (iter == buffer2Offsets.end()) {
      op.emitOpError() << "error: read before first write";
      return WalkResult::interrupt();
    }
    SmallVector<Value> addrs;
    for (auto &offset : iter->second) {
      // if the offset has not been materialized, materialize it.
      if (offset2Constant.find(offset) == offset2Constant.end()) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        auto constantIntOffsetOp =
            rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
        offset2Constant[offset] = constantIntOffsetOp.getResult();
      }
      addrs.push_back(offset2Constant.at(offset));
    }
    rewriter.setInsertionPoint(op);
    auto hivmPointerCastOp = rewriter.create<hivm::PointerCastOp>(
        op.getLoc(), currentMemRefType, ValueRange(addrs),
        ValueRange(op.getDynamicSizes()));
    rewriter.replaceOp(op, hivmPointerCastOp->getResults());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult mlir::hivm::walkUpdateAllocWorkspaceOffset(
    func::FuncOp funcOp,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets) {
  IRRewriter rewriter(funcOp);
  DenseMap<uint64_t, Value> offset2Constant;
  auto result = funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
    // Attach the planned offset(s) to the workspace alloc. Fail with a
    // diagnostic if it is not covered by the memory plan.
    if (!op.getOffset().empty())
      return WalkResult::advance();
    auto iter = buffer2Offsets.find(op.getResult());
    if (iter == buffer2Offsets.end()) {
      op.emitOpError() << "error: read before first write";
      return WalkResult::interrupt();
    }
    SmallVector<Value> argOffset;
    for (auto &offset : iter->second) {
      // if the offset has not been materialized, materialize it.
      if (offset2Constant.find(offset) == offset2Constant.end()) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        auto constantIndexOp =
            rewriter.create<arith::ConstantIndexOp>(op->getLoc(), offset);
        offset2Constant[offset] = constantIndexOp.getResult();
      }
      argOffset.push_back(offset2Constant.at(offset));
    }
    rewriter.setInsertionPoint(op);
    auto newAllocWorkspaceOp =
        rewriter.create<bishengir::memref_ext::AllocWorkspaceOp>(
            op.getLoc(), op->getResultTypes(), op.getWorkspaceArg(),
            op.getDynamicSize(), argOffset);
    rewriter.replaceOp(op, newAllocWorkspaceOp->getResults());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}