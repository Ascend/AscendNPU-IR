//===- SplitMixKernel.cpp ---------------------------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/DistributedTransformUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DECL_SPLITMIXKERNEL
#define GEN_PASS_DEF_SPLITMIXKERNEL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "hivm-split-mix-kernel"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

FailureOr<bool> isCoreTypeOp(Operation *op, enum TCoreType coreType) {
  auto res = getCoreType(op);
  if (failed(res))
    return {};
  return res == coreType;
}

// mark the operands if the defining op is of given core type
void annotateOpOperand(OpBuilder builder, Operation *op,
                       enum TCoreType coreType) {
  size_t numInputOperands;

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op))
    numInputOperands = static_cast<size_t>(dpsOp.getNumDpsInputs());
  else
    numInputOperands = op->getNumOperands();

  auto propagateAnnotation = [&coreType, &builder](Value val) -> void {
    Operation *definingOp = val.getDefiningOp();
    if (!definingOp)
      return;
    auto opCoreType = getCoreType(definingOp);
    if (failed(opCoreType))
      return;
    if (opCoreType.value() == coreType) {
      builder.setInsertionPointAfter(definingOp);
      builder.create<annotation::MarkOp>(definingOp->getLoc(), val);
    } else if (opCoreType.value() == TCoreType::CUBE_OR_VECTOR) {
      annotateOpOperand(builder, definingOp, coreType);
    }
  };

  for (size_t i = 0; i < numInputOperands; i++) {
    Value operand = op->getOperand(i);
    propagateAnnotation(operand);
  }

  // For loop, here should consider its body.
  // Also recurse into scf.if yields: covers master's mixed-if path when
  // SplitMixedIfConditionals was not run. After SplitMixedIf, cube_only /
  // vec_only ifs are typed CUBE/VECTOR so this recurse is rarely hit.
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    for (Value val : loopOp.getYieldedValues()) {
      propagateAnnotation(val);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (auto val : ifOp.thenYield().getResults())
      propagateAnnotation(val);
    if (!ifOp.getElseRegion().empty()) {
      for (auto val : ifOp.elseYield().getResults())
        propagateAnnotation(val);
    }
  }
}

static bool isDefinedOutside(Value value, Region &region) {
  if (Operation *defOp = value.getDefiningOp()) {
    return !region.isAncestor(defOp->getParentRegion());
  }
  if (BlockArgument arg = dyn_cast<BlockArgument>(value)) {
    return !region.isAncestor(arg.getOwner()->getParent());
  }
  return false;
}

static Value createZeroOrEmptyStub(OpBuilder &builder, Location loc,
                                   Type type) {
  return llvm::TypeSwitch<Type, Value>(type)
      .Case<TensorType>([&](TensorType t) {
        return builder.create<tensor::EmptyOp>(loc, t.getShape(),
                                               t.getElementType());
      })
      .Case<IntegerType>([&](IntegerType t) {
        return builder.create<arith::ConstantOp>(loc, t,
                                                 builder.getIntegerAttr(t, 0));
      })
      .Case<FloatType>([&](FloatType t) {
        return builder.create<arith::ConstantOp>(loc, t,
                                                 builder.getFloatAttr(t, 0.0));
      })
      .Default([&](Type t) { return Value(); });
}

static SmallVector<Value> getOutOperands(Operation *op) {
  if (op->getResults().empty()) {
    return {};
  }

  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    return dpsOp.getDpsInits();
  }
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    SmallVector<Value> outOperands;
    auto funcOp =
        mlir::utils::getCalledFunction<hacc::HACCFunction, func::CallOp>(
            callOp);
    if (!funcOp) {
      callOp.emitError() << "callee func not found when filtering mix";
      return {};
    }

    // VF calls: recover output operands via transfer_write destinations in the
    // callee. Falls back to kernel-arg attrs / stubs when tracing finds
    // nothing or does not cover every result.
    if (isVFCall(op)) {
      SmallVector<unsigned> writeArgIds = traceVFWriteOpArgIds(callOp);
      if (writeArgIds.size() == op->getNumResults()) {
        for (unsigned i : writeArgIds)
          outOperands.push_back(op->getOperand(i));
        return outOperands;
      }
    }

    if (funcOp.getArgAttrsAttr()) {
      auto funcParamSize = funcOp.getNumArguments();
      for (size_t i = 0; i < funcParamSize; i++) {
        if (funcOp.isKernelArg(i, hacc::KernelArgType::kOutput) ||
            funcOp.isKernelArg(i, hacc::KernelArgType::kInputAndOutput))
          outOperands.push_back(op->getOperand(i));
      }
      if (outOperands.size() == op->getNumResults())
        return outOperands;
      outOperands.clear();
    }

    // VF results may not map 1:1 onto call operands; fabricate typed stubs so
    // the call can be erased from the opposite core. Non-VF calls must expose
    // outputs via kernel-arg attrs — fail closed instead of inventing values.
    if (!isVFCall(op)) {
      op->emitError()
          << "cannot recover out operands for non-VF call when filtering mix "
             "kernel";
      return {};
    }
    OpBuilder localBuilder(op->getContext());
    localBuilder.setInsertionPoint(op);
    for (auto result : op->getResults()) {
      Value stub =
          createZeroOrEmptyStub(localBuilder, op->getLoc(), result.getType());
      if (!stub) {
        op->emitError()
            << "unsupported VF call result type for stub replacement: "
            << result.getType();
        return {};
      }
      outOperands.push_back(stub);
    }
    return outOperands;
  }

  // For scope ops: derive replacement values from the terminator's operands.
  // Values defined outside the scope are used directly; values defined inside
  // are replaced with zero/empty stubs created immediately before the scope op.
  if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
    Operation *terminator = scopeOp.getBody()->getTerminator();
    SmallVector<Value> outVals;
    for (auto [index, result] : llvm::enumerate(scopeOp.getResults())) {
      if (result.use_empty()) {
        // Sentinel: result has no uses, no replacement needed.
        outVals.push_back(Value());
        continue;
      }
      Value yieldedVal = terminator->getOperand(index);
      if (isDefinedOutside(yieldedVal, scopeOp.getRegion())) {
        outVals.push_back(yieldedVal);
      } else {
        // Insert stub immediately before the scope op so it is visible to
        // all users of the scope result after the scope is erased.
        OpBuilder localBuilder(op->getBlock(), Block::iterator(op));
        Value stub = createZeroOrEmptyStub(localBuilder, scopeOp.getLoc(),
                                           result.getType());
        if (!stub)
          scopeOp.emitError()
              << "Failed to create replacement stub for scope result #" << index
              << " with unsupported type: " << result.getType();
        outVals.push_back(stub); // null on failure; caller checks
      }
    }
    return outVals;
  }

  // Uniform-core scf.if (cube_only / vec_only): same idea as scope — reuse
  // yielded values defined outside the if, otherwise stub before the if.
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    SmallVector<Value> outVals;
    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (result.use_empty()) {
        outVals.push_back(Value());
        continue;
      }
      Value thenVal = ifOp.thenYield().getOperand(index);
      if (isDefinedOutside(thenVal, ifOp.getThenRegion())) {
        outVals.push_back(thenVal);
        continue;
      }
      if (!ifOp.getElseRegion().empty()) {
        Value elseVal = ifOp.elseYield().getOperand(index);
        if (isDefinedOutside(elseVal, ifOp.getElseRegion())) {
          outVals.push_back(elseVal);
          continue;
        }
      }
      OpBuilder localBuilder(op->getBlock(), Block::iterator(op));
      Value stub =
          createZeroOrEmptyStub(localBuilder, ifOp.getLoc(), result.getType());
      if (!stub)
        ifOp.emitError() << "Failed to create replacement stub for if result #"
                         << index
                         << " with unsupported type: " << result.getType();
      outVals.push_back(stub);
    }
    return outVals;
  }

  LDBG("unsupported op: " << *op);
  // TODO: should we get the last operands as out operands by default?
  llvm_unreachable("unsupported op to get out operands");
}

LogicalResult replaceResultWithInitOperand(Operation *op) {
  // replace uses of op result with out operand
  auto numResults = op->getNumResults();
  if (numResults == 0 || isa<tensor::EmptyOp, memref::AllocOp>(op)) {
    return success();
  }

  SmallVector<Value> outOperands = getOutOperands(op);
  if (outOperands.size() != numResults) {
    return op->emitError()
           << "out operands and numResults mismatch when replacing results ("
           << outOperands.size() << " vs " << numResults << ")";
  }

  for (size_t i = 0; i < numResults; i++) {
    OpResult res = op->getResult(i);
    if (!outOperands[i]) {
      // Null sentinel is only valid when the result has no uses.
      if (!res.use_empty())
        return failure();
      continue;
    }
    res.replaceAllUsesWith(outOperands[i]);
  }
  return success();
}

struct SplitMixKernelPass
    : public impl::SplitMixKernelBase<SplitMixKernelPass> {
  void filterMixFunc(OpBuilder &builder, func::FuncOp mixedFunc,
                     enum TCoreType filterCoreType);
  void splitMixKernel(func::FuncOp &funcOp);
  void runOnOperation() override;
  void generateMixKernelDecl(func::FuncOp &funcOp);
};

struct PostCubeReplacement : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  constexpr static llvm::StringRef visitedLabel =
      "PostCubeReplacement::visitedLabel";

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // check if it has already been visited
    if (extractOp.getOperation()->hasAttr(visitedLabel)) {
      return failure();
    }
    extractOp.getOperation()->setAttr(visitedLabel,
                                      rewriter.getI32IntegerAttr(1));

    // do replacements
    for (Operation *userOp : extractOp.getResult().getUsers()) {
      if (annotation::MarkOp markOp = dyn_cast<annotation::MarkOp>(userOp)) {
        if (markOp.getOperation()->hasAttr(
                "DuplicateTensorExtractForCube::replacementLabel")) {
          Value source = markOp.getSrc();
          Value destination = markOp.getValues()[0];
          rewriter.replaceAllUsesWith(source, destination);
          break;
        }
      }
    }
    return success();
  }
};

struct FoldEmptyInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSource().getDefiningOp<tensor::EmptyOp>()) {
      rewriter.replaceOp(op, op.getDest());
      return success();
    }
    return failure();
  }
};

struct RemoveUselessMarkOps : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override {
    if (!markOp.isAttrEmpty()) {
      return failure();
    }

    Value src = markOp.getSrc();
    // TODO: The subsequent logic should be handled correctly for mark, rather
    // than applying special processing.
    Operation *defOp = src.getDefiningOp();
    if (isa<hivm::FixpipeOp, hivm::StoreOp>(defOp)) {
      return failure();
    }

    bool isOnlyMark = llvm::all_of(src.getUsers(), [](Operation *user) {
      return isa<annotation::MarkOp>(user);
    });

    if (isOnlyMark) {
      return failure();
    }

    rewriter.eraseOp(markOp);
    return success();
  }
};

/// Erase ops carrying `attrName` only when all results are unused.
template <typename OpType>
struct RemoveOpWithAttr : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  explicit RemoveOpWithAttr(MLIRContext *ctx, StringRef attrName)
      : OpRewritePattern<OpType>(ctx), attrName(attrName) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr(attrName))
      return failure();

    for (Value result : op->getResults()) {
      if (!result.use_empty())
        return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  StringRef attrName;
};

void postProcessCubeFunc(func::FuncOp func) {
  auto *ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<PostCubeReplacement>(ctx);
  patterns.insert<FoldEmptyInsertSlice>(ctx);
  patterns.insert<RemoveUselessMarkOps>(ctx);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (failed(applyPatternsGreedily(func.getOperation(), std::move(patterns))))
    llvm::report_fatal_error("postProcessCubeFunc failed");

  RewritePatternSet removePatterns(ctx);
  removePatterns.add<RemoveOpWithAttr<bufferization::ToTensorOp>>(
      ctx, "DuplicateTensorExtractForCube::cubeErasureLabel");
  removePatterns.add<RemoveOpWithAttr<hivm::LoadOp>>(
      ctx, "DuplicateTensorExtractForCube::cubeErasureLabel");
  removePatterns.add<RemoveOpWithAttr<memref::AllocOp>>(
      ctx, "DuplicateTensorExtractForCube::cubeErasureLabel");
  if (failed(applyPatternsGreedily(func.getOperation(),
                                   std::move(removePatterns))))
    llvm::report_fatal_error("postProcessCubeFunc remove patterns failed");
}

void postProcessVectorFunc(func::FuncOp func) {
  auto *ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  patterns.insert<RemoveUselessMarkOps>(ctx);
  patterns.add<RemoveOpWithAttr<annotation::MarkOp>>(
      ctx, "DuplicateTensorExtractForCube::replacementLabel");
  patterns.add<RemoveOpWithAttr<tensor::ExtractOp>>(
      ctx, "DuplicateTensorExtractForCube::newExtractLabel");
  if (failed(applyPatternsGreedily(func.getOperation(), std::move(patterns))))
    llvm::report_fatal_error("postProcessVectorFunc failed");
}

} // namespace

static bool isLoopOfCoreType(scf::ForOp forOp, TCoreType coreType) {
  if (auto coreTypeAttr = forOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::kPipelinedLoopCoreTypeAttrName)) {
    return coreType == coreTypeAttr.getTcoretype();
  }
  FailureOr<TCoreType> inferredCoreType = getCoreType(forOp);
  return llvm::succeeded(inferredCoreType) &&
         coreType == inferredCoreType.value();
}

static void inferDistributedCoreType(func::FuncOp mixedFunc) {
  auto rectifyCoreType = [mixedFunc](hivm::TCoreType &cur) {
    auto funcCoreType = mixedFunc
                            ->getAttrOfType<hivm::TFuncCoreTypeAttr>(
                                hivm::TFuncCoreTypeAttr::name)
                            .getFuncCoreType();
    if (cur == TCoreType::CUBE_AND_VECTOR) {
      return kTFuncCoreType2TCoreType.at(funcCoreType);
    }
    return cur;
  };
  mixedFunc->walk([&rectifyCoreType](Operation *op) {
    if (isDistributedTypeCustomOp(op)) {
      if (auto res = hivm::detail::queryCoreTypeHelper(op)) {
        auto coreType = rectifyCoreType(res.value());
        auto coreTypeAttr =
            hivm::TCoreTypeAttr::get(op->getContext(), coreType);
        op->setAttr(hivm::TCoreTypeAttr::name, coreTypeAttr);
      }
    }
  });
}

/// Tag SIMT scopes as VECTOR and erase empty scopes belonging to the filtered
/// core type (pre-order so nested scopes are handled outer-first).
static void filterEmptyScopesPreOrder(OpBuilder &builder,
                                      func::FuncOp mixedFunc,
                                      TCoreType keepCoreType) {
  mixedFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto scopeOp = dyn_cast<scope::ScopeOp>(op);
    if (!scopeOp)
      return WalkResult::advance();

    if (auto vectorType = scopeOp->getAttrOfType<StringAttr>("vector_type")) {
      if (vectorType.getValue() == "simt") {
        scopeOp->setAttr(
            hivm::TCoreTypeAttr::name,
            hivm::TCoreTypeAttr::get(builder.getContext(), TCoreType::VECTOR));
      }
    }

    if (scopeOp->getNumResults() != 0)
      return WalkResult::advance();

    auto attr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!attr)
      return WalkResult::advance();

    // Erase scopes that belong to the core type being filtered out.
    if (attr.getTcoretype() != keepCoreType) {
      op->erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

// erase ops of given core type from function
void SplitMixKernelPass::filterMixFunc(OpBuilder &builder,
                                       func::FuncOp mixedFunc,
                                       enum TCoreType filterCoreType) {
  // `coreType` is the core that remains after filtering `filterCoreType` out.
  const enum TCoreType coreType =
      filterCoreType == TCoreType::CUBE ? TCoreType::VECTOR : TCoreType::CUBE;

  // Distributed CustomOps need a concrete core type before erase walks.
  inferDistributedCoreType(mixedFunc);
  filterEmptyScopesPreOrder(builder, mixedFunc, coreType);

  mixedFunc.walk<WalkOrder::PostOrder>([&](Operation *op) {
    LDBG("filterMixFunc visiting: " << *op);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (isLoopOfCoreType(forOp, filterCoreType)) {
        // Pipelined loops: zero-trip so structure/anchors stay; otherwise skip
        // the body (preserve loop for multibuffer anchors on RegBase).
        if (forOp->hasAttr(hivm::kPipelinedLoopCoreTypeAttrName)) {
          forOp.setUpperBound(forOp.getLowerBound());
          return WalkResult::advance();
        }
        return WalkResult::skip();
      }
    }

    // Scope pipelined-loop core type is already honored by getCoreType.
    FailureOr<bool> res = isCoreTypeOp(op, filterCoreType);
    if (failed(res)) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (!res.value())
      return WalkResult::advance();

    // Region ops: replace nested filtered ops' results before erasing parent.
    if (op->getNumRegions() > 0) {
      WalkResult innerWalk =
          op->walk<WalkOrder::PostOrder>([&](Operation *innerOp) {
            if (innerOp == op)
              return WalkResult::advance();
            FailureOr<bool> innerRes = isCoreTypeOp(innerOp, filterCoreType);
            if (failed(innerRes)) {
              signalPassFailure();
              return WalkResult::interrupt();
            }
            if (innerRes.value()) {
              annotateOpOperand(builder, innerOp, coreType);
              if (failed(replaceResultWithInitOperand(innerOp))) {
                signalPassFailure();
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
      if (innerWalk.wasInterrupted())
        return WalkResult::interrupt();
    }

    annotateOpOperand(builder, op, coreType);
    if (failed(replaceResultWithInitOperand(op))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    op->erase();
    return WalkResult::advance();
  });
}

void SplitMixKernelPass::generateMixKernelDecl(func::FuncOp &funcOp) {
  auto module = funcOp->getParentOfType<ModuleOp>();
  std::optional<SymbolTable::UseRange> maybeUses = funcOp.getSymbolUses(module);
  if (!maybeUses || maybeUses.value().empty()) {
    // only generate decl if there exists at least one caller
    return;
  }

  if (!llvm::all_of(maybeUses.value(), [&](SymbolTable::SymbolUse use) {
        auto call = cast<func::CallOp>(use.getUser());
        auto caller = call->getParentOfType<hacc::HACCFunction>();
        return caller.isHost();
      })) {
    funcOp.emitError()
        << "Currently, MIX kernels can only be called by host functions!";
    return signalPassFailure();
  }

  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  auto funcDeclOp = builder.create<func::FuncOp>(
      funcOp->getLoc(), funcOp.getSymName(), funcOp.getFunctionType());
  funcDeclOp.setPrivate();
  funcDeclOp->setAttr(
      hacc::HACCFuncTypeAttr::name,
      hacc::HACCFuncTypeAttr::get(&getContext(), hacc::HACCFuncType::DEVICE));
  funcDeclOp->setAttr(
      hivm::TFuncCoreTypeAttr::name,
      hivm::TFuncCoreTypeAttr::get(&getContext(), hivm::TFuncCoreType::MIX));
  if (hacc::utils::isDeviceEntry(funcOp))
    funcDeclOp->setAttr(
        hacc::stringifyEnum(hacc::HACCToLLVMIRTranslateAttr::MIX_ENTRY),
        UnitAttr::get(&getContext()));
}

void SplitMixKernelPass::splitMixKernel(func::FuncOp &func) {
  StringRef funcName = func.getSymName();
  auto funcCoreTypeAttr = func->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  if (!funcCoreTypeAttr ||
      funcCoreTypeAttr.getFuncCoreType() != TFuncCoreType::MIX)
    return;

  // generate a Mix function declaration for host callers
  generateMixKernelDecl(func);

  // clone the function and add mix-aic / mix-aiv post-fix name
  OpBuilder builder(func);
  builder.setInsertionPointAfter(func.getOperation());
  auto vecFunc = cast<func::FuncOp>(builder.clone(*func.getOperation()));

  func.setSymNameAttr(builder.getStringAttr(funcName + kMixFuncAicSuffix));
  vecFunc.setSymNameAttr(builder.getStringAttr(funcName + kMixFuncAivSuffix));

  func->setAttr(hivm::TPartOfMixAttr::name, builder.getUnitAttr());
  vecFunc->setAttr(hivm::TPartOfMixAttr::name, builder.getUnitAttr());

  func->setAttr(hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(func->getContext(),
                                             hivm::TFuncCoreType::AIC));
  vecFunc->setAttr(hivm::TFuncCoreTypeAttr::name,
                   hivm::TFuncCoreTypeAttr::get(vecFunc->getContext(),
                                                hivm::TFuncCoreType::AIV));

  // filter vector ops from AIC kernel, and cube ops from AIV kernel
  filterMixFunc(builder, func, TCoreType::VECTOR);
  LDBG("New aic:\n" << func);
  postProcessCubeFunc(func);
  filterMixFunc(builder, vecFunc, TCoreType::CUBE);
  LDBG("New aiv:\n" << vecFunc);
  postProcessVectorFunc(vecFunc);
}

void SplitMixKernelPass::runOnOperation() {
  /// For DebugOp, the core type should be computed before doing
  /// "filterMixFunc", because "filterMixFunc" may remove OPs
  /// and thus affect the inferred core type.
  /// We just call inferCoreType on every DebugOp
  /// to ensure that the core type has already been marked in attributes.
  /// See DebugOp::inferCoreType for details.
  getOperation()->walk([](hivm::DebugOp debugOp) {
    debugOp.inferCoreType();
    return WalkResult::advance();
  });

  SmallVector<func::FuncOp> funcList;
  getOperation()->walk([&](func::FuncOp func) {
    if (hacc::utils::isHost(func))
      return WalkResult::advance();

    funcList.push_back(func);
    return WalkResult::advance();
  });
  for (auto &func : funcList) {
    splitMixKernel(func);
  }
}

std::unique_ptr<Pass> mlir::hivm::createSplitMixKernelPass() {
  return std::make_unique<SplitMixKernelPass>();
}
