//===--------- LegalizeBool.cpp - Legalize Bool type Pass -----------------===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
        hfusion::RoundMode::RINT, op.getOutputs().front(), true, false,
        hfusion::TypeFn{});
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
static bool hasPseudoBoolAttr(Operation *op) {
  return op && op->hasAttr("was_bool_to_int8");
}

static Value getPseudoBoolSource(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return {};
  }

  if (hfusion::isReshapeOrSliceOp(defOp)) {
    return hfusion::getReshapeOrSliceSource(defOp);
  }

  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(defOp)) {
    return broadcastOp.getInput();
  }

  if (auto absOp = dyn_cast<math::AbsIOp>(defOp)) {
    return absOp.getOperand();
  }

  if (auto unaryOp = dyn_cast<linalg::ElemwiseUnaryOp>(defOp)) {
    if (unaryOp.getFun() == linalg::UnaryFn::abs &&
        unaryOp.getNumDpsInputs() == 1) {
      return unaryOp.getInputs()[0];
    }
    return {};
  }

  if (auto unaryOp = dyn_cast<hfusion::ElemwiseUnaryOp>(defOp)) {
    if ((unaryOp.getFun() == hfusion::UnaryFn::absi ||
         unaryOp.getFun() == hfusion::UnaryFn::relu) &&
        unaryOp.getNumDpsInputs() == 1) {
      return unaryOp.getInputs()[0];
    }
  }

  return {};
}

static bool isPseudoBool(Value val) {
  if (auto defOp = val.getDefiningOp()) {
    if (hasPseudoBoolAttr(defOp)) {
      return true;
    }

    Value source = getPseudoBoolSource(val);
    if (source) {
      return isPseudoBool(source);
    }
  }
  return false;
}

template <typename OpTy>
struct ClampPseudoBoolArithOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
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

    Type i8Type = tensorType.getElementType();
    auto shape = tensorType.getShape();

    // When both operands are canonical pseudo-bools, "integer arith +
    // clamp-to-nonzero" is exactly a bitwise logical op, so the whole
    // sign-extend / fill / compare / extend sequence collapses to:
    //   clamp(a + b) == a | b   (0+0=0, 0+1=1, 1+1=2 all map to OR over {0,1})
    //   clamp(a - b) == a ^ b   ((a - b) != 0  <=>  a != b)
    // This matches the existing bool-add -> vor / bool-mul -> vand convention
    // (see BoolBinaryFnToLogicOp above).
    hfusion::BinaryFn logicFn = std::is_same_v<OpTy, arith::AddIOp>
                                    ? hfusion::BinaryFn::vor
                                    : hfusion::BinaryFn::vxor;
    Value logicInit = rewriter.create<tensor::EmptyOp>(location, shape, i8Type);
    Operation *newOp = createBinaryOp<hfusion::ElemwiseBinaryOp,
                                      hfusion::BinaryFn, hfusion::BinaryFnAttr>(
        rewriter, location, logicFn,
        ValueRange{op->getOperand(0), op->getOperand(1)},
        ValueRange{logicInit});
    Value logicResult = newOp->getResult(0);
    newOp->setAttr("is_clamped", rewriter.getBoolAttr(true));

    // Canonicalize the result to {0, 1} with a bitwise AND against 1. This is a
    // no-op when the inputs are already {0, 1} (the normal case), but it keeps
    // the {0, 1} output guarantee of the original compare path even if a "true"
    // ever arrives as all-ones (0xFF / -1).
    Value oneScalar = rewriter.create<arith::ConstantOp>(
        location, rewriter.getIntegerAttr(i8Type, 1));
    Value oneInit = rewriter.create<tensor::EmptyOp>(location, shape, i8Type);
    Value oneTensor =
        rewriter.create<linalg::FillOp>(location, oneScalar, oneInit)
            .getResult(0);
    Value andInit = rewriter.create<tensor::EmptyOp>(location, shape, i8Type);
    Operation *clampedOp =
        createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                       hfusion::BinaryFnAttr>(
            rewriter, location, hfusion::BinaryFn::vand,
            ValueRange{logicResult, oneTensor}, ValueRange{andInit});
    clampedOp->setAttr("was_bool_to_int8", rewriter.getBoolAttr(true));
    rewriter.replaceOp(op, clampedOp->getResult(0));
    return success();
  }
};

// Convert hfusion.cast i1 → fXX to select(cond, 1.0, 0.0) to avoid the
// unsupported uitofp i1→fXX path that AVE hardware cannot handle.
struct CastI1ToSelectPattern : public OpRewritePattern<hfusion::CastOp> {
  using OpRewritePattern<hfusion::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    auto inputType = mlir::dyn_cast<RankedTensorType>(
        linalgOp.getDpsInputs().front().getType());
    auto outputType = mlir::dyn_cast<RankedTensorType>(
        linalgOp.getDpsInitOperand(0)->get().getType());
    if (!inputType || !outputType)
      return failure();

    // Only apply on RegBasedArch (register-based architectures)
    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod || !hacc::utils::isRegBasedArch(mod))
      return failure();

    if (!inputType.getElementType().isSignlessInteger(1))
      return failure();
    if (!mlir::isa<FloatType>(outputType.getElementType()))
      return failure();

    auto loc = op.getLoc();
    auto elemType = outputType.getElementType();

    // Reuse the original cast's init tensor — naturally supports dynamic shapes
    Value init = linalgOp.getDpsInitOperand(0)->get();

    // Get dynamic dimension sizes from the init tensor (if any)
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
      if (outputType.isDynamicDim(i)) {
        Value dimIdx =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
        dynamicSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, init, dimIdx));
      }
    }

    // Create scalar constants 1.0 and 0.0
    ArrayRef<NamedAttribute> attrs = op->getAttrs();
    TypeFn castVal = TypeFn::cast_unsigned;
    auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                                  return attr.getName() == "cast"; });
    if (castIter != attrs.end()) {
      if (auto attr = llvm::dyn_cast<TypeFnAttr>(castIter->getValue())) {
        castVal = attr.getValue();
      }
    }

    Value oneScalar = (castVal == TypeFn::cast_unsigned) ?
      rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getFloatAttr(elemType, 1.0)) :
      rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getFloatAttr(elemType, -1.0));
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, elemType, rewriter.getFloatAttr(elemType, 0.0));

    // Create true-constant tensor: fill 1.0
    Value trueInit = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), elemType, dynamicSizes);
    Value trueVal =
        rewriter.create<linalg::FillOp>(loc, oneScalar, trueInit).getResult(0);

    // Create false-constant tensor: fill 0.0
    Value falseInit = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), elemType, dynamicSizes);
    Value falseVal = rewriter.create<linalg::FillOp>(loc, zeroScalar, falseInit)
                         .getResult(0);

    // Create hfusion.select(cond, true_val, false_val) outs(init)
    Value cond = linalgOp.getDpsInputs().front();
    auto selectOp = rewriter.create<hfusion::SelectOp>(
        loc, TypeRange{init.getType()}, ValueRange{cond, trueVal, falseVal},
        ValueRange{init});

    rewriter.replaceOp(op, selectOp.getResult(0));
    return success();
  }
};

bool isIntegerElemType(Type type, unsigned width) {
  auto elemTy = getElementTypeOrSelf(type);
  return elemTy.isInteger(width);
}

bool isI1ElemType(Type type) { return isIntegerElemType(type, 1); }
bool isI8ElemType(Type type) { return isIntegerElemType(type, 8); }

template <auto OpFn, auto LogicOpFn>
struct BoolBinaryFnToLogicOp
    : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getFun() != OpFn) {
      return failure();
    }

    if (op->getNumResults() != 1) {
      return failure();
    }

    auto inputs = op.getInputs();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    if (!isI1ElemType(lhs.getType()) || !isI1ElemType(rhs.getType())) {
      return failure();
    }

    Location location = op.getLoc();

    Value logicResult =
        createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                       hfusion::BinaryFnAttr>(rewriter, location, LogicOpFn,
                                              ValueRange{lhs, rhs},
                                              ValueRange{op.getOutputs()[0]})
            ->getResult(0);

    rewriter.replaceOp(op, logicResult);
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

  // Convert i1→fXX hfusion.cast to select(cond, 1.0, 0.0)
  RewritePatternSet castI1Patterns(context);
  castI1Patterns.add<CastI1ToSelectPattern>(context);
  if (failed(applyPatternsGreedily(mod, std::move(castI1Patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Legalize Bool CastI1ToSelect Failed\n");
  }

  // Convert bool MulI and AddI to AndI and OrI
  RewritePatternSet logicPatterns(context);
  logicPatterns.add<
      BoolBinaryFnToLogicOp<linalg::BinaryFn::mul, hfusion::BinaryFn::vand>,
      BoolBinaryFnToLogicOp<linalg::BinaryFn::add, hfusion::BinaryFn::vor>>(
      context);
  if (failed(applyPatternsGreedily(mod, std::move(logicPatterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Legalize Bool on linalg::BinaryFn Failed\n");
  }

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

std::unique_ptr<Pass>
hfusion::createLegalizeBoolPass(const LegalizeBoolPassOptions &options) {
  return std::make_unique<LegalizeBoolPass>(options);
}
