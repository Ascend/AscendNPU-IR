//===---------------------- InlineFixpipe.cpp -----------------------------===//
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

// This pass converts ops to hivm.fixpipe.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Contracts/FixpipePreQuantContract.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeFallback.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeFusion.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_INLINEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-fixpipe"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

struct InlineFixpipe : public impl::InlineFixpipeBase<InlineFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

/// when all the activationOps are ready, there should be relu, leaky-relu and
/// p-relu
bool isActivationOp(Operation *op) { return isa<hivm::VReluOp>(op); }

static int64_t getSiftedUsersNum(Value v) {
  const DenseSet<Operation *> container(v.getUsers().begin(),
                                        v.getUsers().end());
  auto filteredRange = llvm::make_filter_range(container, [](Operation *op) {
    return !isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(op);
  });
  return DenseSet<Operation *>(filteredRange.begin(), filteredRange.end())
      .size();
}

//===----------------------------------------------------------------------===//
// InlineFixpipeOpPattern
//===----------------------------------------------------------------------===//
// Fixpipe can complete 3 inner action with origin matrixC operand following
// conditions
//   1. cast or quantization
//   2. relu and other activation function
//   3. store or layout
// Potential optimization is to fuse condition 1&2&3 into fixpipe.

struct InlineFixpipeOpPattern : public OpRewritePattern<FixpipeOp> {
public:
  InlineFixpipeOpPattern(MLIRContext *context, bool inlineQuantScale)
      : OpRewritePattern<FixpipeOp>(context),
        inlineQuantScale(inlineQuantScale) {}

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResultTensor())
      return failure();

    auto fixpipeResTensor = op.getResultTensor();
    if (fixpipeResTensor.getUsers().empty())
      return failure();

    if (getSiftedUsersNum(fixpipeResTensor) != 1)
      return failure();

    return inlineFixpipeOp(rewriter, op);
  }

private:
  LogicalResult inlineFixpipeOp(PatternRewriter &rewriter, FixpipeOp op) const {
    bool matched = false;
    auto loc = op.getLoc();
    Operation *curOp = nullptr;
    for (Operation *maybeDebugOp : op.getResultTensor().getUsers()) {
      if (isa<hivm::DebugOp>(maybeDebugOp) && !op->getAttr(usedForDebugOp)) {
        rewriter.setInsertionPoint(maybeDebugOp);
        FixpipeOp clonedFixpipeOp = cast<FixpipeOp>(rewriter.clone(*op));
        clonedFixpipeOp->setAttr(usedForDebugOp, rewriter.getBoolAttr(true));
        Value clonedResult = clonedFixpipeOp->getResult(0);
        hivm::DebugOp debugOp = cast<hivm::DebugOp>(maybeDebugOp);
        rewriter.modifyOpInPlace(
            debugOp, [&]() { debugOp.getArgMutable().assign(clonedResult); });
      } else {
        curOp = maybeDebugOp;
      }
    }
    // FixPipe followed by debugOp only, no need to inline
    if (curOp == nullptr)
      return success();
    // Avoid fusing fixpipe into a VECTOR-scope store (mix AIC/AIV).
    if (isInsideVectorScope(curOp))
      return failure();

    if (isOnRegBasedArch(op) && op.getDmaMode() != FixpipeDMAMode::NZ2NZ) {
      if (all_of(op->getUsers(), [](auto *user) {
            if (isa<annotation::MarkOp>(user))
              return true;
            return isa<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
                >(user);
          })) {

        convertNz2NdFixpipeToNz2Nz(rewriter, op);
        return success();
      }
    }

    // 1. cast or quantization (single VCast or a single-use VCast chain).
    //    Quant mode is decided from first-cast input type to last-cast output
    //    type so decomposed casts (e.g. i32->i16->i8) still fuse as S322I8.
    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp) {
      SmallVector<hivm::VCastOp> castChain = collectVCastChain(castOp);
      if (getQuantModeForCastChain(castChain).has_value()) {
        matched = true;
        inlineFixpipeWithQuant(rewriter, loc, op, castChain,
                               op.getDpsInputOperand(0)->get());
      }
    } else if (op.getFixpipeState() <= op.needFixpipePreFuse() &&
               isActivationOp(curOp)) {
      // 2. relu and other activation function
      matched = true;
      auto reluOp = llvm::dyn_cast_if_present<hivm::VReluOp>(curOp);
      inlineFixpipeWithRelu(rewriter, loc, op, reluOp);
    } else if (auto storeOp = llvm::dyn_cast_if_present<hivm::StoreOp>(curOp)) {
      //   3. store or layout
      auto storeAttr = storeOp.getAtomicKindAttr();
      hivm::AtomicKind atomicKind = hivm::AtomicKind::NONE;
      if (storeAttr)
        atomicKind = storeAttr.getValue();
      if (atomicKind == AtomicKind::NONE || atomicKind == AtomicKind::ADD ||
          atomicKind == AtomicKind::MAX || atomicKind == AtomicKind::MIN) {
        matched = true;
        inlineFixpipeWithStore(rewriter, loc, op, storeOp,
                               op.getDpsInputOperand(0)->get());
      }
    } else if (auto vMulOp = dyn_cast<hivm::VMulOp>(curOp);
               vMulOp && isOnRegBasedArch(op) &&
               (inlineQuantScale || hasQuantScaleCompileHint(vMulOp)) &&
               isUserQuantScaleInlinable(op, vMulOp)) {
      matched = true;
      inlineFixpipeWithQuantScale(rewriter, op, vMulOp);
    } else if (isUserTransposeInlinable(op, curOp)) {
      matched = true;
      inlineFixpipeWithTranspose(rewriter, op, cast<hivm::VTransposeOp>(curOp));
    } else if (auto extractSliceOp =
                   dyn_cast_if_present<tensor::ExtractSliceOp>(curOp);
               isUserSliceSwappable(op, extractSliceOp)) {
      // change to fixpipe op + extract_slice to extract_slice + fixpipe op
      if (op->getBlock() == extractSliceOp->getBlock()) {
        // only swap when fixpipe op and extract slice op are in same block,
        // otherwise, extract slice op may be in sub block loop and fixpipe
        // cannot be fused into.
        matched = true;
        swapFixpipeWithExtractSlice(rewriter, loc, op, extractSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp()) &&
               !op->getAttr(fixpipeDoNotMoveOutOfScfFor) &&
               !op.getSource().getDefiningOp<hivm::BatchMmadL1Op>()) {
      // Move fixpipe out of scf.for.
      // The BatchMmadL1 case is excluded because TileBatchMMIntoLoop requires
      // `BatchMmadL1 -> Fixpipe` use chain to stay in the same block.
      // Do not hoist a type-converting fixpipe out of a loop-carried value:
      // hoisting rewires the yield to the fixpipe source, which would change
      // the loop yield type.
      if (op.getSource().getType() != op.getResultTensor().getType())
        return rewriter.notifyMatchFailure(
            op, "fixpipe converts the loop-carried value type");
      matched = true;
      auto scfForOp = dyn_cast_if_present<scf::ForOp>(curOp->getParentOp());
      moveFixpipeOutOfScfFor(rewriter, loc, op, scfForOp, op.getResultTensor());
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::IfOp>(curOp->getParentOp()) &&
               curOp->getParentOp()->hasAttr(kFallbackNotExec)) {
      // Sink the outer consumer chain into both fallback_not_exec branches.
      // Cast / transpose / extract_slice / store fusion is left to the
      // existing greedy InlineFixpipe patterns on the Cube-side fixpipe.
      if (succeeded(sinkFixpipeIntoFallbackIf(
              rewriter, loc, op, cast<scf::IfOp>(curOp->getParentOp()))))
        matched = true;
    }
    return matched ? success() : failure();
  }

  const bool inlineQuantScale;
};

void populateInlineFixpipePatterns(RewritePatternSet &patterns,
                                   bool inlineQuantScale) {
  patterns.add<InlineFixpipeOpPattern>(patterns.getContext(), inlineQuantScale);
}

void eraseInlinableQuantScaleMarkOps(Operation *op) {
  SmallVector<annotation::MarkOp> inlinableQuantScaleMarkOps;
  op->walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttrOfType<UnitAttr>(utils::kInlinableQuantScaleAttr))
      inlinableQuantScaleMarkOps.push_back(markOp);
  });
  for (annotation::MarkOp markOp : inlinableQuantScaleMarkOps)
    markOp.erase();
}

void InlineFixpipe::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateInlineFixpipePatterns(patterns, inlineQuantScale);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  eraseInlinableQuantScaleMarkOps(getOperation());
}

std::unique_ptr<Pass>
createInlineFixpipePass(const InlineFixpipeOptions &options) {
  return std::make_unique<InlineFixpipe>(options);
}

} // namespace mlir::hivm
