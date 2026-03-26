//===--------- LegalizeBool.cpp - Legalize BF16 type Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Analysis/ReshapeAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-legalize-bool"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEBOOLPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::utils::debugger;
using namespace hfusion;

static const std::string generatedMarkOpAttr = "generated_by_legalize_bool";

struct CastOpFold : public OpRewritePattern<hfusion::CastOp> {
  using OpRewritePattern<hfusion::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto parentCastOp = dyn_cast_if_present<hfusion::CastOp>(
        op.getInputs().front().getDefiningOp());
    if (!parentCastOp || !utils::getAnnotateOpWithAttr(
                             parentCastOp.getResult(0), generatedMarkOpAttr)) {
      return failure();
    }

    // In Legalize Bool Pass
    // We want to fold:
    // %1 = cast %0 type A -> type i1
    // %2 = cast %1 type i1 -> type B
    // to
    // %2 = cast %0 type A -> type B
    rewriter.setInsertionPointAfter(op);
    auto foldedCast = hfusion::castTo(
        rewriter, parentCastOp.getInputs().front(), op.getResultTypes().front(),
        hfusion::RoundMode::RINT, op.getOutputs().front(), true, false, hfusion::TypeFn{});
    rewriter.replaceOp(op, foldedCast);
    return success();
  }
};

struct DeleteCreatedMarkOp : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override {
    if (utils::isAnnotationWithAttr(markOp, generatedMarkOpAttr)) {
      rewriter.eraseOp(markOp);
      return success();
    }
    return failure();
  }
};

// Identify if the input value carries the pseudo boolean attribute
static bool isPseudoBool(Value val) {
  if (auto defOp = val.getDefiningOp()) {
    return defOp->hasAttr("was_bool_to_int8");
  }
  return false;
}

template <typename OpTy>
struct ClampPseudoBoolArithOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    bool hasPseudoBool = false;
    for (auto operand : op->getOperands()) {
      if (isPseudoBool(operand)) {
        hasPseudoBool = true;
        break;
      }
    }

    if (!hasPseudoBool) {
      return failure();
    }

    // Avoid re-applying normalization to already processed operations
    if (op->hasAttr("is_clamped")) {
      return failure();
    }

    if (op->getNumResults() != 1) {
      return failure();
    }

    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType || !tensorType.getElementType().isInteger(8)) {
      return failure();
    }

    Location location = op.getLoc();

    // Clone operation to preserve original metadata
    OpTy newMathOp = cast<OpTy>(rewriter.clone(*op.getOperation()));
    newMathOp->setAttr("is_clamped", rewriter.getBoolAttr(true));

    Value unClampedResult = newMathOp->getResult(0);

    // Prepare i32 tensor type for intermediate calculation
    Type i32Type = rewriter.getI32Type();
    auto tensorTypeI32 = RankedTensorType::get(tensorType.getShape(), i32Type);

    // Sign extend addition result to i32
    Value extendedResult = rewriter.create<arith::ExtSIOp>(
        location, tensorTypeI32, unClampedResult);

    // Construct i32 zero tensor
    Value zeroScalarI32 = rewriter.create<arith::ConstantOp>(
        location, rewriter.getI32IntegerAttr(0));
    Value emptyTensorI32 = rewriter.create<tensor::EmptyOp>(
        location, tensorTypeI32.getShape(), i32Type);
    Value zeroTensorI32 = rewriter.create<linalg::FillOp>(
        location, zeroScalarI32, emptyTensorI32).getResult(0);

    // Compare in i32 precision
    Value cmpTensor = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ne, extendedResult, zeroTensorI32);

    // Zero extend boolean result back to i8
    Value clampedTensor = rewriter.create<arith::ExtUIOp>(
        location, tensorType, cmpTensor);

    // Propagate semantic marker
    if (auto extOp = clampedTensor.getDefiningOp()) {
      extOp->setAttr("was_bool_to_int8", rewriter.getBoolAttr(true));
    }

    rewriter.replaceOp(op, clampedTensor);
    return success();
  }
};

void populateLegalizeBoolFoldPatterns(RewritePatternSet &patterns) {
  patterns.add<CastOpFold>(patterns.getContext());
}

void populateLegalizeBoolCleanPatterns(RewritePatternSet &patterns) {
  patterns.add<DeleteCreatedMarkOp>(patterns.getContext());
}

void populateClampPseudoBoolPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // Register the template pattern for both Addition and Subtraction
  patterns.add<ClampPseudoBoolArithOp<arith::AddIOp>,
               ClampPseudoBoolArithOp<arith::SubIOp>>(context);
}

class LegalizeBoolPass : public impl::LegalizeBoolPassBase<LegalizeBoolPass> {
public:
  LegalizeBoolPass() = default;
  
  explicit LegalizeBoolPass(const LegalizeBoolPassOptions &options)
      : impl::LegalizeBoolPassBase<LegalizeBoolPass>(options) {}
  
  void runOnOperation() override;

private:
  std::shared_ptr<hfusion::detail::ReshapeAnalyzer> reshapeAnalyzer;

  /// Modifies a function's type signature by converting boolean types to int8.
  /// Updates both input arguments and result types, and also updates block
  /// arguments in the function body to match the new types.
  ///
  /// @param func The function operation to modify
  /// @param builder The OpBuilder to use for creation of new operations
  /// @return LogicalResult indicating success or failure
  LogicalResult modifyFunctionType(func::FuncOp func, OpBuilder &builder) {
    FunctionType oldType = func.getFunctionType();
    llvm::SmallVector<Type, 4> newInputTypes;
    llvm::SmallVector<Type, 4> newResultTypes;

    // Convert Input Type
    for (Type type : oldType.getInputs()) {
      newInputTypes.push_back(convertBoolToInt8(type));
    }

    // Convert Result Type
    for (Type type : oldType.getResults()) {
      newResultTypes.push_back(convertBoolToInt8(type));
    }

    // Create new function type
    FunctionType newType =
        builder.getFunctionType(newInputTypes, newResultTypes);
    func.setType(newType);

    // Update function body
    if (!func.empty()) {
      Block &entryBlock = func.getBody().front();

      // Update block argument types
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        BlockArgument arg = entryBlock.getArgument(i);
        arg.setType(newInputTypes[i]);
      }

      builder.setInsertionPointToStart(&entryBlock);
    }
    return success();
  }

  /// Overloaded version of modifyFunctionType that handles a function and all
  /// its call sites. Updates both the caller function and all call sites to use
  /// i8 instead of boolean types.
  ///
  /// @param callerInfo Information about the caller function and its call sites
  /// @param builder The OpBuilder to use for creation of new operations
  /// @return LogicalResult indicating success or failure
  LogicalResult modifyFunctionType(const tiling::CallerInfo &callerInfo,
                                   OpBuilder &builder) {
    if (failed(modifyFunctionType(callerInfo.caller, builder))) {
      return failure();
    }
    for (auto callSite : callerInfo.callSites) {
      for (auto opr : callSite->getOperands()) {
        if (isI1ElemType(opr.getType())) {
          opr.setType(convertBoolToInt8(opr.getType()));
        }
      }
      for (auto res : callSite->getResults()) {
        if (isI1ElemType(res.getType())) {
          res.setType(convertBoolToInt8(res.getType()));
        }
      }
    }
    return success();
  }

  /// Inserts casting operations to convert i8 input arguments back to i1 for
  /// use in the function body. Handles reshape operations by updating types
  /// throughout the reshape chain and inserting appropriate casts.
  ///
  /// @param inputArgument The function argument to convert
  /// @param builder The OpBuilder to use for inserting casting operations
  void castInputToI8(Value inputArgument, OpBuilder &builder) {
    Type i1Type = builder.getI1Type();
    Type i8Type = builder.getI8Type();
    SmallVector<hfusion::detail::ReshapeValue> reshapedValues;
    if (isa<TensorType>(inputArgument.getType())) {
      reshapeAnalyzer->getReshapeDescendants(inputArgument, reshapedValues);
    } else {
      for (auto &use : inputArgument.getUses())
        reshapedValues.emplace_back(inputArgument, use, 0);
    }
    for (auto descendant : reshapedValues) {
      auto descVal = descendant.endTarget->get();
      auto reshapeChain = reshapeAnalyzer->getReshapeChain(descVal);
      if (reshapeChain.empty())
        continue;
      LDBG("Casting descendant " << descVal
                                 << " chain length: " << reshapeChain.size());
      assert(!isReshapeOp(reshapeChain.back().getDefiningOp()));
      auto *edPtr = std::prev(reshapeChain.end());
      for (auto *reshapeVal = reshapeChain.begin(); reshapeVal != edPtr;
           reshapeVal++) {
        reshapeVal->setType(convertBoolToInt8(reshapeVal->getType()));
      }
      auto mode =
          mlir::utils::selectRoundMode<hfusion::RoundMode>(i8Type, i1Type);
      LDBG("descVal " << descVal);
      assert(isa<BlockArgument>(descVal) ||
             isReshapeOp(descVal.getDefiningOp()));
      assert(!isReshapeOp(descendant.endTarget->getOwner()));
      builder.setInsertionPointAfterValue(descVal);
      auto castResult = hfusion::castTo(builder, /*src=*/reshapeChain.front(),
                                        /*targetElemType=*/i1Type,
                                        /*roundMode=*/mode);
      if (isa<TensorType>(descVal.getType())) {
        auto newMarkOp =
            builder.create<annotation::MarkOp>(castResult.getLoc(), castResult);
        newMarkOp->setAttr(generatedMarkOpAttr, builder.getBoolAttr(true));
      }
      descendant.endTarget->set(castResult);
    }
  }

  /// Main method to convert a function kernel from using boolean types to int8.
  /// Performs the full conversion process:
  /// 1. Updates function signature
  /// 2. Converts input arguments with appropriate casts
  /// 3. Converts return values with appropriate casts
  /// 4. Updates reshape operations throughout the function
  ///
  /// @param func The function operation to convert
  /// @param builder The OpBuilder to use for creating operations
  /// @return LogicalResult indicating success or failure
  LogicalResult convertKernel(func::FuncOp func, OpBuilder &builder) {
    FunctionType oldType = func.getFunctionType();
    if (failed(modifyFunctionType(func, builder))) {
      signalPassFailure();
    }

    reshapeAnalyzer = std::make_shared<hfusion::detail::ReshapeAnalyzer>(func);
    // Update function body
    if (!func.empty()) {
      // Cast updated i8 input argument to Int1
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (isI1ElemType(oldType.getInput(i)))
          castInputToI8(func.getArgument(i), builder);
      }

      // Sign extend boolean return value to Int8
      Type i1Type = builder.getI1Type();
      Type i8Type = builder.getI8Type();
      func.walk([&](func::ReturnOp returnOp) {
        builder.setInsertionPoint(returnOp);
        llvm::SmallVector<Value, 4> newOperands;
        for (auto &operand : returnOp->getOpOperands()) {
          if (!isI1ElemType(operand.get().getType()))
            continue;
          // Op -> Reshape -> Reshape -> return
          // This will be converted to
          // Op -> Cast -> Reshape -> Reshape -> return
          auto returnChain = reshapeAnalyzer->getReshapeChain(operand.get());
          if (returnChain.empty())
            continue;
          auto mode =
              mlir::utils::selectRoundMode<hfusion::RoundMode>(i1Type, i8Type);
          builder.setInsertionPointAfterValue(returnChain.back());
          auto castResult = hfusion::castTo(builder, /*src=*/returnChain.back(),
                                            /*targetElemType=*/i8Type,
                                            /*roundMode=*/mode);
          OpOperand *lastOpOperand = &operand;
          if (returnChain.size() > 1) {
            lastOpOperand = &reshapeAnalyzer->getFirstReshape(returnChain)
                                 .getDefiningOp()
                                 ->getOpOperands()
                                 .front();
          }
          lastOpOperand->set(castResult);
          for (auto reshapeVal : returnChain) {
            if (isReshapeOp(reshapeVal.getDefiningOp())) {
              // TODO: clone the chain instead of converting it
              reshapeVal.setType(convertBoolToInt8(reshapeVal.getType()));
            }
          }
        }
      });
    }
    return success();
  }

  bool isIntegerElemType(Type type, unsigned width) const {
    auto elemTy = getElementTypeOrSelf(type);
    return elemTy.isInteger(width);
  }

  bool isI1ElemType(Type type) const { return isIntegerElemType(type, 1); }
  bool isI8ElemType(Type type) const { return isIntegerElemType(type, 8); }

  Type convertBoolToInt8(Type type) {
    if (!isI1ElemType(type)) {
      return type;
    }
    Type typeInt8 = IntegerType::get(type.getContext(), 8);
    if (type.isInteger(1)) {
      // scalar type i1
      return typeInt8;
    }
    auto shapedType = dyn_cast_or_null<ShapedType>(type);
    if (!shapedType) {
      return type;
    }
    // shaped type i1
    return shapedType.clone(typeInt8);
  }
};

void LegalizeBoolPass::runOnOperation() {
  MLIRContext *context = &getContext();
  OpBuilder builder(context);

  ModuleOp mod = getOperation();
  
  // Conditional Execution Branch
  if (this->enableClamp) {
    RewritePatternSet clampPatterns(context);
    clampPatterns.add<ClampPseudoBoolArithOp<arith::AddIOp>,
                      ClampPseudoBoolArithOp<arith::SubIOp>>(context);
                      
    if (failed(applyPatternsGreedily(mod, std::move(clampPatterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Legalize Bool Arithmetic Clamp Failed\n");
    }
    
    // Exit early if only clamp is requested
    return;
  }

  // Standard LegalizeBool logic executes when enableClamp is false
  SmallVector<func::FuncOp> deviceEntryFuncs;
  mod.walk([&](func::FuncOp func) {
    if (hacc::utils::isDeviceEntry(func)) {
      if (failed(convertKernel(func, builder)))
        signalPassFailure();

      deviceEntryFuncs.push_back(func);
    }
  });

  for (auto &deviceEntry : deviceEntryFuncs) {
    DenseMap<func::FuncOp, tiling::CallerInfo> workList;
    tiling::getCallerInfo(deviceEntry, mod, workList);
    // If there is no caller, just modify the entry kernel.
    if (workList.empty())
      continue;

    for (auto &[caller, callInfo] : workList) {
      if (failed(modifyFunctionType(callInfo, builder)))
        signalPassFailure();
    }
  }
  RewritePatternSet patterns(&getContext());
  populateLegalizeBoolFoldPatterns(patterns);
  if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Legalize Bool Cast Fold Failed");
  }

  RewritePatternSet clearUpPatterns(&getContext());
  populateLegalizeBoolCleanPatterns(clearUpPatterns);
  if (failed(applyPatternsGreedily(mod, std::move(clearUpPatterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Legalize Bool Clear Failed");
  }
}

std::unique_ptr<Pass> hfusion::createLegalizeBoolPass() {
  return std::make_unique<LegalizeBoolPass>();
}

std::unique_ptr<Pass> hfusion::createLegalizeBoolPass(const LegalizeBoolPassOptions &options) {
  return std::make_unique<LegalizeBoolPass>(options);
}