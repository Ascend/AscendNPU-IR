//===------------- HIVMOptSinglePoint.cpp - optimize single point op-------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <queue>

namespace mlir {
#define GEN_PASS_DEF_HIVMOPTSINGLEPOINTOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-opt-single-point"

using namespace mlir;
using namespace mlir::hivm;
using namespace utils;
namespace {
struct None {};

struct HIVMOptSinglePointOpPass
    : public impl::HIVMOptSinglePointOpBase<HIVMOptSinglePointOpPass> {
  void runOnOperation() override;
};

template <
    typename HIVMOP, typename ARITHOP,
    typename = typename std::enable_if<!std::is_same_v<ARITHOP, None>>::type>
static void replaceByScalarOp(PatternRewriter &rewriter, HIVMOP hivmOp) {
  rewriter.setInsertionPoint(hivmOp);

  auto getScalarValueFunc = [&](Value v) -> Value {
    return getScalarValue(rewriter, hivmOp->getLoc(), v);
  };

  llvm::SmallVector<Value, 4> scalarInputs;
  llvm::transform(hivmOp.getDpsInputs(), std::back_inserter(scalarInputs),
                  getScalarValueFunc);

  auto arithOp = rewriter.create<ARITHOP>(hivmOp.getLoc(), scalarInputs);
  auto mem = hivmOp.getDpsInits()[0];
  assert(isa<MemRefType>(mem.getType()));
  assert(hivmOp.getDpsInits().size() == 1);
  createSinglePointStore(rewriter, hivmOp.getLoc(), arithOp.getResult(), mem);

  rewriter.eraseOp(hivmOp);
}

template <typename HIVMOP, typename ARITHFOP, typename ARITHIOP,
          typename ARITHUIOP>
struct SinglePointEltVecOp : public OpRewritePattern<HIVMOP> {
  using OpRewritePattern<HIVMOP>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIVMOP op,
                                PatternRewriter &rewriter) const final {
    if (!isa<hivm::HIVMStructuredOp>(op.getOperation())) {
      return failure();
    }

    if (!op.hasPureBufferSemantics()) {
      return failure();
    }

    if (op.getDpsInits().size() != 1) {
      return failure();
    }

    auto dstType = op.getDpsInits()[0].getType();
    if (!isa<MemRefType>(dstType)) {
      return failure();
    }

    auto dstShape = cast<MemRefType>(dstType).getShape();
    if (llvm::any_of(dstShape, [](int64_t s) { return s != 1; })) {
      return failure();
    }

    auto elementType = getElementTypeOrSelf(op->getOperand(0).getType());
    if (!elementType.isF32() && !elementType.isInteger(64)) {
      // hardware scalar operations only support f32 and integer 64
      // TODO : get support types from platform info
      return failure();
    }

    if (isa<FloatType>(elementType)) {
      if constexpr (!std::is_same_v<ARITHFOP, None>) {
        replaceByScalarOp<HIVMOP, ARITHFOP>(rewriter, op);
      }
    } else if (elementType.isSignedInteger() ||
               elementType.isSignlessInteger()) {
      if constexpr (!std::is_same_v<ARITHIOP, None>) {
        replaceByScalarOp<HIVMOP, ARITHIOP>(rewriter, op);
      }
    } else if (elementType.isUnsignedInteger()) {
      if constexpr (!std::is_same_v<ARITHUIOP, None>) {
        replaceByScalarOp<HIVMOP, ARITHUIOP>(rewriter, op);
      }
    }

    return success();
  }
};

struct SinglePointVBrcOp : public OpRewritePattern<VBrcOp> {
  using OpRewritePattern<VBrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VBrcOp brcOp,
                                PatternRewriter &rewriter) const final {
    if (!brcOp.hasPureBufferSemantics()) {
      return failure();
    }

    auto dstType = brcOp.getDst().getType();
    assert(isa<MemRefType>(dstType));
    auto dstShape = cast<MemRefType>(dstType).getShape();
    if (llvm::any_of(dstShape, [](int64_t s) { return s != 1; })) {
      return failure();
    }

    // optimize to use scalar operation to do instead of vector operation
    rewriter.setInsertionPoint(brcOp);
    auto src = brcOp.getSrc();
    Value scalarSrc;
    if (isa<MemRefType>(src.getType())) {
      scalarSrc = createSinglePointLoad(rewriter, brcOp->getLoc(), src);
    } else {
      scalarSrc = src;
    }

    createSinglePointStore(rewriter, brcOp->getLoc(), scalarSrc,
                           brcOp.getDst());

    rewriter.eraseOp(brcOp);
    return success();
  }
};

bool isAllMemoryUsersValid(Value gm, bool isWrite) {
  std::queue<Value> queue;
  queue.push(gm);

  while (!queue.empty()) {
    auto cur = queue.front();
    queue.pop();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        queue.push(castOp.getResult());
      } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
        queue.push(subviewOp.getResult());
      } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        queue.push(collapseOp->getResult(0));
      } else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) {
        queue.push(expandOp->getResult(0));
      } else if (auto hivmOp = dyn_cast<HIVMStructuredOp>(user)) {
        bool isReadOperand = hivmOp.isDpsInput(&use);
        // not optimize if has write op
        if ((isReadOperand && isWrite) || (!isReadOperand)) {
          return false;
        }
      } else if (isa<memref::LoadOp>(user)) {
        // continue to check other users
        continue;
      } else {
        // not optimize if has unknown op
        return false;
      }
    }
  }
  return true;
}

template <typename CopyOpType>
struct SinglePointVCopyLikeOp : public OpRewritePattern<CopyOpType> {
  using OpRewritePattern<CopyOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOpType copyLikeOp,
                                PatternRewriter &rewriter) const final {
    if (!copyLikeOp.hasPureBufferSemantics()) {
      return failure();
    }

    if constexpr (std::is_same_v<CopyOpType, hivm::StoreOp>) {
      return rewriter.notifyMatchFailure(copyLikeOp,
                                         "not optimize if store from ub to gm");
    }

    if constexpr (std::is_same_v<CopyOpType, hivm::LoadOp>) {
      func::FuncOp funcOp =
          copyLikeOp->template getParentOfType<func::FuncOp>();
      if (!hacc::utils::hasNoIOAlias(funcOp)) {
        return rewriter.notifyMatchFailure(
            copyLikeOp, "only optimize load if func has no_alias attr");
      }
    }

    Value copySrc = copyLikeOp.getSrc();
    Value copyDst = copyLikeOp.getDst();
    MemRefType srcType = dyn_cast_or_null<MemRefType>(copySrc.getType());
    MemRefType dstType = dyn_cast_or_null<MemRefType>(copyDst.getType());
    if (!srcType || !dstType) {
      return failure();
    }

    if (!srcType.getMemorySpace() || !dstType.getMemorySpace()) {
      return failure();
    }

    auto dstShape = dstType.getShape();
    if (llvm::any_of(dstShape, [](int64_t s) { return s != 1; })) {
      return failure();
    }

    auto gmOperand = copyLikeOp.getSrc();
    auto ubOperand = copyLikeOp.getDst();
    auto gmArg = utils::tracebackMemRef(gmOperand);
    if (!isAllMemoryUsersValid(gmArg, /*isWrite=*/false)) {
      return failure();
    }

    auto loadOp =
        createSinglePointLoad(rewriter, copyLikeOp->getLoc(), gmOperand);
    createSinglePointStore(rewriter, copyLikeOp->getLoc(), loadOp.getResult(),
                           ubOperand);
    rewriter.eraseOp(copyLikeOp);
    return success();
  }
};
} // namespace

void HIVMOptSinglePointOpPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<
      SinglePointVBrcOp, SinglePointVCopyLikeOp<hivm::CopyOp>,
      SinglePointVCopyLikeOp<hivm::LoadOp>,
      SinglePointEltVecOp<hivm::VAddOp, arith::AddFOp, arith::AddIOp, None>,
      SinglePointEltVecOp<hivm::VSubOp, arith::SubFOp, arith::SubIOp, None>,
      SinglePointEltVecOp<hivm::VMulOp, arith::MulFOp, arith::MulIOp, None>,
      SinglePointEltVecOp<hivm::VDivOp, arith::DivFOp, arith::DivSIOp,
                          arith::DivUIOp>,
      SinglePointEltVecOp<hivm::VAbsOp, math::AbsFOp, math::AbsIOp, None>,
      SinglePointEltVecOp<hivm::VSqrtOp, math::SqrtOp, math::SqrtOp, None>,
      SinglePointEltVecOp<hivm::VMaxOp, arith::MaxNumFOp, arith::MaxSIOp, None>,
      SinglePointEltVecOp<hivm::VMinOp, arith::MinNumFOp, arith::MinSIOp,
                          None>>(&getContext());

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> hivm::createHIVMOptSinglePointPass() {
  return std::make_unique<HIVMOptSinglePointOpPass>();
}
