//===- AdaptTritonKernel.cpp - Adapt triton kernel                       -===//
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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Transforms/Transform.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ADAPTTRITONKERNEL
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "adapt-triton-kernel"

using namespace mlir;

namespace {

/// This pass labels the triton entry kernel
struct AdaptTritonKernelPass
    : public impl::AdaptTritonKernelBase<AdaptTritonKernelPass> {
  using AdaptTritonKernelBase<AdaptTritonKernelPass>::AdaptTritonKernelBase;

public:
  AdaptTritonKernelPass(std::string hivmcVersion)
      : AdaptTritonKernelBase(AdaptTritonKernelOptions{hivmcVersion}) {}
  void runOnOperation() override;
};

void markWorkspaceArgument(func::FuncOp funcOp) {
  constexpr StringRef workspaceArgName = "WorkspaceArgIdx";
  if (!funcOp->hasAttr(workspaceArgName))
    return;
  int64_t workspaceArgIdx =
      funcOp->getAttrOfType<IntegerAttr>(workspaceArgName).getInt();
  assert(funcOp.getNumArguments() > workspaceArgIdx);

  funcOp.setArgAttrs(
      workspaceArgIdx,
      SmallVector<NamedAttribute>{hacc::createHACCKernelArgAttr(
          funcOp.getContext(), hacc::KernelArgType::kWorkspace)});
}

void markSyncBlockLockArgument(func::FuncOp funcOp) {
  constexpr StringRef lockArgName = "SyncBlockLockArgIdx";
  if (!funcOp->hasAttr(lockArgName))
    return;
  int64_t syncBlockLockArgIdx =
      funcOp->getAttrOfType<IntegerAttr>(lockArgName).getInt();
  assert(funcOp.getNumArguments() > syncBlockLockArgIdx);

  funcOp.setArgAttrs(
      syncBlockLockArgIdx,
      SmallVector<NamedAttribute>{hacc::createHACCKernelArgAttr(
          funcOp.getContext(), hacc::KernelArgType::kSyncBlockLock)});
}

/// triton print op is converted triton_print func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_print".
struct TritonPrintToHFusionPrintPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef printFuncName = "triton_print";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(printFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix triton_print");
    }
    auto printOpNameStr = hfusion::PrintOp::getOperationName();
    auto printOpName =
        mlir::OperationName(printOpNameStr, rewriter.getContext());
    auto prefixAttrName = hfusion::PrintOp::getPrefixAttrName(printOpName);
    bool hasPrefixAttr = funcOp->hasAttr(prefixAttrName);
    if (!hasPrefixAttr) {
      return rewriter.notifyMatchFailure(
          funcOp, funcName + " has no attribute of prefix");
    }
    auto hexAttrName = hfusion::PrintOp::getHexAttrName(printOpName);
    bool hasHexAttr = funcOp->hasAttr(hexAttrName);
    if (!hasHexAttr) {
      return rewriter.notifyMatchFailure(funcOp,
                                         funcName + " has no attribute of hex");
    }

    auto prefixAttr = funcOp->getAttrOfType<StringAttr>(prefixAttrName);
    auto hexAttr = funcOp->getAttrOfType<BoolAttr>(hexAttrName);
    for (Value operand : callOp.getArgOperands()) {
      rewriter.create<hfusion::PrintOp>(callOp.getLoc(), prefixAttr, hexAttr,
                                        operand);
    }
    rewriter.eraseOp(callOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton assert op is converted triton_assert func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_assert".
struct TritonAssertToHFusionAssertPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef assertFuncName = "triton_assert";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(assertFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp, funcName + " does not starts with the prefix triton_assert");
    }
    auto assertOpNameStr = hfusion::AssertOp::getOperationName();
    auto assertOpName =
        mlir::OperationName(assertOpNameStr, rewriter.getContext());
    auto msgAttrName = hfusion::AssertOp::getMsgAttrName(assertOpName);
    bool hasMsgAttr = funcOp->hasAttr(msgAttrName);
    if (!hasMsgAttr) {
      return rewriter.notifyMatchFailure(funcOp,
                                         funcName + " has no attribute of msg");
    }
    auto msgAttr = funcOp->getAttrOfType<StringAttr>(msgAttrName);
    if (callOp.getArgOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          callOp,
          "calling " + funcName + " with wrong number of args (expecting 1)");
    }
    auto originArg = callOp.getArgOperands()[0];
    auto ty = originArg.getType();
    if (isa<RankedTensorType>(ty)) {
      auto i1TensorTy = cast<RankedTensorType>(ty);
      SmallVector<int64_t> shape(i1TensorTy.getShape());
      RankedTensorType i8TensorTy =
          RankedTensorType::get(shape, rewriter.getI8Type());
      auto arg = rewriter.create<arith::ExtUIOp>(callOp.getLoc(), i8TensorTy,
                                                 originArg);
      rewriter.create<hfusion::AssertOp>(callOp.getLoc(), msgAttr, arg);
    } else {
      rewriter.create<hfusion::AssertOp>(callOp.getLoc(), msgAttr, originArg);
    }
    rewriter.eraseOp(callOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton gather op is converted triton_gather func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_gather".
struct TritonGatherToHFusionGatherPattern
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef gatherFuncName = "triton_gather";

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(gatherFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + gatherFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value index = callOp.getOperand(1);
    Value axisVal = callOp.getOperand(2);
    auto axis = mlir::utils::getArithConstantOpValue<int64_t>(axisVal);
    if (failed(axis)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the gather axis.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto elemTy = srcTy.getElementType();
    auto indexTy = cast<RankedTensorType>(index.getType());
    auto resShape = indexTy.getShape();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resShape, elemTy);
    auto gatherOp =
        rewriter.create<hfusion::GatherOp>(loc, src, index, init, *axis);
    rewriter.replaceOp(callOp, gatherOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

/// triton cumsum/cumprod op is converted triton_cumsum func call in triton
/// adaptor, therefore this pattern match func call of which func name starts
/// with "triton_cumsum/triton_cumprod".
struct TritonCumToHFusionCumPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef cumsumFuncName = "triton_cumsum";
  static constexpr StringRef cumprodFuncName = "triton_cumprod";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    mlir::hfusion::CumOpType cumOpType = mlir::hfusion::CumOpType::UNDEFINED;
    if (funcName.starts_with(cumsumFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMSUM;
    } else if (funcName.starts_with(cumprodFuncName)) {
      cumOpType = mlir::hfusion::CumOpType::CUMPROD;
    } else {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + cumsumFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value dimVals = callOp.getOperand(1);
    auto cumDim = mlir::utils::getArithConstantOpValue<int64_t>(dimVals);
    if (failed(cumDim)) {
      return callOp->emitError("Failed to extract the value of arith.constant "
                               "defining the cum dims.");
    }
    Value reverseVals = callOp.getOperand(2);
    auto reverse = mlir::utils::getArithConstantOpValue<bool>(reverseVals);
    if (failed(reverse)) {
      return callOp->emitError("Failed to extract the value of arith.constant"
                               "defining the reverse.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    llvm::SmallVector<int64_t> cumDims{*cumDim};
    if (cumOpType == mlir::hfusion::CumOpType::CUMSUM) {
      rewriter.replaceOp(
          callOp, rewriter.create<hfusion::CumsumOp>(loc, srcTy, src, cumDims, *reverse));
    } else if (cumOpType == mlir::hfusion::CumOpType::CUMPROD) {
      rewriter.replaceOp(callOp, rewriter.create<hfusion::CumprodOp>(
                                     loc, srcTy, src, cumDims, *reverse));
    } else {
      llvm_unreachable("unsupport cumulative function");
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// triton sort op is converted triton_sort func call in triton adaptor,
/// therefore this pattern match func call of which func name starts with
/// "triton_sort".
struct TritonSortToHFusionSortPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef sortFuncName = "triton_sort";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(sortFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + sortFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value sortAxisVals = callOp.getOperand(1);
    auto sortAxis = mlir::utils::getArithConstantOpValue<int64_t>(sortAxisVals);
    if (failed(sortAxis)) {
      return callOp->emitError("Failed to extract the value of arith.constant"
                               "defining the sort axis.");
    }
    Value descendingVals = callOp.getOperand(2);
    auto descending =
        mlir::utils::getArithConstantOpValue<bool>(descendingVals);
    if (failed(descending)) {
      return callOp->emitError("Failed to extract the value of arith.constant"
                               "defining the descending.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto sortOp = rewriter.create<hfusion::SortOp>(loc, srcTy, src, *descending,
                                                   *sortAxis);
    rewriter.replaceOp(callOp, sortOp);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct TritonFlipToHFusionFlipPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  static constexpr StringRef flipFuncName = "triton_flip";
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(callOp);
    auto funcName = funcOp.getSymName();
    if (!funcName.starts_with(flipFuncName)) {
      return rewriter.notifyMatchFailure(
          callOp,
          funcName + " does not starts with the prefix " + flipFuncName);
    }

    auto loc = callOp.getLoc();
    Value src = callOp.getOperand(0);
    Value flipAxisVals = callOp.getOperand(1);
    auto flipAxis = mlir::utils::getArithConstantOpValue<int64_t>(flipAxisVals);
    if (failed(flipAxis)) {
      return callOp->emitError("Failed to extract the value of arith.constant"
                               "defining the Flip axis.");
    }

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto flipOp = rewriter.create<hfusion::FlipOp>(loc, srcTy, src, *flipAxis);
    rewriter.replaceOp(callOp, flipOp);
    rewriter.eraseOp(funcOp);

    return success();
  }
};

struct TritonBindSubBlockAttrToHFusionPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  static constexpr llvm::StringLiteral kSplitVectorAttrName = "bind_sub_block";

  static bool hasBindSubBlockAttr(PatternRewriter &rewriter, scf::ForOp op) {
    if (!op->hasAttr(kSplitVectorAttrName))
      return false;
    return op->getAttr(kSplitVectorAttrName) == rewriter.getBoolAttr(true);
  }

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasBindSubBlockAttr(rewriter, op))
      return failure();

    // The reason for normalizing is that for changing into parallel blocks,
    // only index type is acceptable
    auto newForOp = bishengir::scf::normalizeToIndex(rewriter, op);
    rewriter.modifyOpInPlace(
        newForOp, [&]() { newForOp->removeAttr(kSplitVectorAttrName); });
    rewriter.modifyOpInPlace(newForOp, [&]() {
      newForOp->setAttr(hfusion::BindSubBlockAttr::name,
                        UnitAttr::get(newForOp->getContext()));
    });
    return success();
  }
};

} // end anonymous namespace

void AdaptTritonKernelPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ModuleOp module = getOperation();
  patterns
      .add<TritonPrintToHFusionPrintPattern, TritonAssertToHFusionAssertPattern,
           TritonGatherToHFusionGatherPattern, TritonCumToHFusionCumPattern,
           TritonBindSubBlockAttrToHFusionPattern,
           TritonSortToHFusionSortPattern, TritonFlipToHFusionFlipPattern>(
          patterns.getContext());
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
  module.walk([&](func::FuncOp funcOp) {
    std::string globalKernelStr{"global_kernel"};
    auto attr = funcOp->getAttr(globalKernelStr);
    if (!attr) {
      return;
    }
    hacc::utils::setDeviceEntry(funcOp);
    funcOp->removeAttr(globalKernelStr);

    // Extract workspace and sync-block-lock argument info if it exists and then
    // remark HACC attribute
    markWorkspaceArgument(funcOp);
    markSyncBlockLockArgument(funcOp);
  });
  // Add metadata.
  if (version_utils::isCompatibleHIVMCVersion(this->hivmcVersion)) {
    module->setAttr(utils::kMemrefAsPtr, UnitAttr::get(context));
  }
}

std::unique_ptr<Pass>
mlir::hfusion::createAdaptTritonKernelPass(std::string hivmcVersion) {
  return std::make_unique<AdaptTritonKernelPass>(hivmcVersion);
}
