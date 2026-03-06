//===----- LowerMemRefExt.cpp - Lower Extended MemRef Dialect ---*- C++ -*-===//
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

#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LOWERMEMREFEXT
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "lower-memref-ext"

namespace {
struct MemrefExtLowering : public impl::LowerMemRefExtBase<MemrefExtLowering> {
  using Base::Base;
  void runOnOperation() override;
};

class LowerAllocWorkSpace
    : public OpRewritePattern<bishengir::memref_ext::AllocWorkspaceOp> {
public:
  LowerAllocWorkSpace(MLIRContext *context, int64_t localWorkSpaceSize)
      : OpRewritePattern(context), localWorkSpaceSize(localWorkSpaceSize) {}

  LogicalResult matchAndRewrite(bishengir::memref_ext::AllocWorkspaceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto offset = op.getOffset();
    if (offset.empty())
      return op->emitOpError("only support lower AllocWorkspaceOp with offset");

    // 1. Get offset inner workspace per block
    assert(offset.size() <= 2); // Making sure offset is not more than 2
    Value localOffset = offset.back();
    // Consider loop double buffer state
    if (offset.size() == 2) { // loop of double buffer
      auto loopOp = op->getParentOfType<LoopLikeOpInterface>();
      if (!loopOp)
        llvm_unreachable("Illegal state where DB workspace is not in loop");

      Value selectCounter;
      {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        selectCounter = createNestedIndexModular(
            rewriter, op, 2); // nested modular value is 2 for this case
      }

      assert(selectCounter);
      Value selectCondition = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI1Type(), selectCounter);
      localOffset = rewriter.create<arith::SelectOp>(
          loc, rewriter.getIndexType(), selectCondition, offset[1], offset[0]);
    }

    // 2. For workspace of current block, here get start address offset from
    // total workspace pool(which points to function argument).
    auto blockIdx = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(),
        // hivm::GetBlockIdxOp has pure trait and CSE will make simplification
        rewriter.create<hivm::GetBlockIdxOp>(loc, rewriter.getI64Type())
            .getResult());

    // 3. Accumulate 1&2
    auto blockIdxDim = rewriter.getAffineDimExpr(0);
    auto localOffsetSym = rewriter.getAffineSymbolExpr(0);
    auto viewOffset = rewriter.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(
            1, 1, (blockIdxDim * this->localWorkSpaceSize) + localOffsetSym),
        ValueRange{blockIdx, localOffset});

    // TODO: Currently, there is only stride = 1 scenario, which may be enhanced
    // in the future
    auto viewOp = rewriter.create<memref::ViewOp>(
        loc, op.getType(), op.getWorkspaceArg(),
        /*byte_shift*/ viewOffset, /*dynamic_sizes*/ ValueRange{});
    rewriter.replaceOp(op, viewOp);
    return success();
  }

private:
  const int64_t localWorkSpaceSize;
};

std::optional<int64_t> getLocalWorkSpaceSize(ModuleOp moduleOp) {
  if (!hacc::existHost(moduleOp))
    return std::nullopt;

  std::optional<int64_t> localWorkSpaceSize = std::nullopt;
  moduleOp.walk([&](func::FuncOp func) {
    std::optional<hacc::HostFuncType> hostType =
        hacc::utils::getHostFuncType(func);
    if (!hostType.has_value() ||
        hostType.value() != hacc::HostFuncType::kInferWorkspaceShapeFunction)
      return WalkResult::advance();

    // Here aims for function which just exits one terminator with a static
    // index val.
    // While it may match `HoistTensorEmpty`'s generation which hasn't
    // AllocWorkspaceOP, there's no side effect for that.
    if (func.getNumResults() == 1 && isa<IndexType>(func.getResultTypes()[0])) {
      auto returnOp = utils::getAssumedUniqueReturnOp(func);
      assert(returnOp);
      auto candidate = getConstantIntValue(returnOp.getOperands()[0]);
      if (candidate.has_value())
        localWorkSpaceSize = candidate.value();
    }
    // Assume there's only one `InferWorkspaceShapeFunction`.
    return WalkResult::interrupt();
  });

  return localWorkSpaceSize;
}

// TODO: Currently it only supports to lower workspace inner each block.
// There exits reduce state where workspace shoule be perceived by all blocks,
// potential related improvements include workspace creation, mem plan, inter
// block sync injection and this lower step.
void MemrefExtLowering::runOnOperation() {
  ModuleOp moduleOp = cast<ModuleOp>(getOperation());

  auto localWorkSpaceSize = getLocalWorkSpaceSize(moduleOp);
  if (!localWorkSpaceSize.has_value())
    return;

  RewritePatternSet patterns(&getContext());

  patterns.add<LowerAllocWorkSpace>(&getContext(), localWorkSpaceSize.value());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::createMemrefExtLoweringPass() {
  return std::make_unique<MemrefExtLowering>();
}
