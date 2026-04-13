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
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
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
    assert(succeeded(opCoreType));
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

  // For loop, here should consider its body
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    for (Value val : loopOp.getYieldedValues()) {
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
      .Default([&](Type t) {
        llvm::errs() << "Unsupported result type for stubbing: " << t << "\n";
        return Value();
      });
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
    assert(funcOp && "callee func not found!");

#ifndef NDEBUG
    auto funcArgAttrs = funcOp.getArgAttrsAttr();
    assert(funcArgAttrs && "called func must has arg attr");
#endif

    auto funcParamSize = funcOp.getNumArguments();
    for (size_t i = 0; i < funcParamSize; i++) {
      if (funcOp.isKernelArg(i, hacc::KernelArgType::kOutput))
        outOperands.push_back(op->getOperand(i));
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

  // TODO: should we get the last operands as out operands by default?
  llvm_unreachable("unsupported op to get out operands");
}

void replaceResultWithInitOperand(Operation *op) {
  // replace uses of op result with out operand
  auto numResults = op->getNumResults();
  if (numResults == 0 || isa<tensor::EmptyOp, memref::AllocOp>(op)) {
    return;
  }

  SmallVector<Value> outOperands = getOutOperands(op);
  assert(outOperands.size() == numResults &&
         "out operands and numResults mismatch");

  for (size_t i = 0; i < numResults; i++) {
    OpResult res = op->getResult(i);
    res.replaceAllUsesWith(outOperands[i]);
  }
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
    if (!markOp->getAttrs().empty()) {
      return failure();
    }

    Value src = markOp.getSrc();

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

template <typename OpType>
void removeOpWithAttrFromFunc(std::string attr, func::FuncOp func) {
  func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (OpType concreteOp = dyn_cast<OpType>(op)) {
      if (concreteOp.getOperation()->hasAttr(attr)) {
        op->erase();
      }
    }
  });
}

void postProcessCubeFunc(func::FuncOp func) {
  RewritePatternSet patterns(func.getOperation()->getContext());
  patterns.insert<PostCubeReplacement>(patterns.getContext());
  patterns.insert<FoldEmptyInsertSlice>(patterns.getContext());
  patterns.insert<RemoveUselessMarkOps>(patterns.getContext());
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (failed(applyPatternsGreedily(func.getOperation(), std::move(patterns)))) {
    llvm::report_fatal_error("postProcessCubeFunc failed");
  }
  removeOpWithAttrFromFunc<bufferization::ToTensorOp>(
      "DuplicateTensorExtractForCube::cubeErasureLabel", func);
  removeOpWithAttrFromFunc<hivm::LoadOp>(
      "DuplicateTensorExtractForCube::cubeErasureLabel", func);
  removeOpWithAttrFromFunc<memref::AllocOp>(
      "DuplicateTensorExtractForCube::cubeErasureLabel", func);
}

void postProcessVectorFunc(func::FuncOp func) {
  RewritePatternSet patterns(func.getOperation()->getContext());
  patterns.insert<RemoveUselessMarkOps>(patterns.getContext());
  if (failed(applyPatternsGreedily(func.getOperation(), std::move(patterns)))) {
    llvm::report_fatal_error("postProcessVectorFunc failed");
  }
  removeOpWithAttrFromFunc<annotation::MarkOp>(
      "DuplicateTensorExtractForCube::replacementLabel", func);
  removeOpWithAttrFromFunc<tensor::ExtractOp>(
      "DuplicateTensorExtractForCube::newExtractLabel", func);
}

} // namespace

static bool isLoopOfCoreType(scf::ForOp forOp, TCoreType coreType) {
  FailureOr<TCoreType> inferredCoreType = getCoreType(forOp);
  return llvm::succeeded(inferredCoreType) &&
         coreType == inferredCoreType.value();
}

// erase ops of given core type from function
void SplitMixKernelPass::filterMixFunc(OpBuilder &builder,
                                       func::FuncOp mixedFunc,
                                       enum TCoreType filterCoreType) {
  const enum TCoreType coreType =
      filterCoreType == TCoreType::CUBE ? TCoreType::VECTOR : TCoreType::CUBE;

  SmallVector<Operation *> toSinkOutOfLoop;
  mixedFunc.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (isLoopOfCoreType(forOp, filterCoreType)) {
        forOp.setUpperBound(forOp.getLowerBound());
        return;
      }
    }

    FailureOr<bool> res = isCoreTypeOp(op, filterCoreType);
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      if (auto coreTypeAttr = op->getAttrOfType<hivm::TCoreTypeAttr>(
              hivm::kPipelinedLoopCoreTypeAttrName)) {
        auto scopeCoreType = coreTypeAttr.getTcoretype();
        res = scopeCoreType == filterCoreType;
      }
    }

    if (failed(res)) {
      signalPassFailure();
      return;
    }
    // If core type does not match, erase the operation
    if (res.value()) {
      annotateOpOperand(builder, op, coreType);
      replaceResultWithInitOperand(op);
      op->erase();
    }
  });

  for (Operation *op : toSinkOutOfLoop)
    op->moveAfter(op->getParentOfType<scf::ForOp>());
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
  if (!func->hasAttr(hivm::TFuncCoreTypeAttr::name))
    return;

  hivm::TFuncCoreTypeAttr funcCoreTypeAttr = cast<hivm::TFuncCoreTypeAttr>(
      func.getOperation()->getAttr(hivm::TFuncCoreTypeAttr::name));
  // only operate on functions with CUBE_OR_VECTOR attribute
  if (!funcCoreTypeAttr ||
      funcCoreTypeAttr.getFuncCoreType() != TFuncCoreType::MIX)
    return;

  // generate a Mix function declaration for host callers
  generateMixKernelDecl(func);

  // clone the function and add "_mix_aiv" and "_mix_aic" post-fix name
  OpBuilder builder(func);
  builder.setInsertionPointAfter(func.getOperation());
  auto vecFunc = cast<func::FuncOp>(builder.clone(*func.getOperation()));
  vecFunc.setSymNameAttr(builder.getStringAttr(funcName + "_mix_aiv"));
  func.setSymNameAttr(builder.getStringAttr(funcName + "_mix_aic"));

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
  postProcessCubeFunc(func);
  filterMixFunc(builder, vecFunc, TCoreType::CUBE);
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
