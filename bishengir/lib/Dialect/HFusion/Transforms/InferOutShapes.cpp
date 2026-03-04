//===- InferOutShapes.cpp -- create shape function of kernel outputs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate functions that infers the shape
// of kernel function's outputs.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "hfusion-infer-out-shapes"

namespace mlir {
#define GEN_PASS_DEF_INFEROUTSHAPESPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir {

namespace {

Value convertOpFoldResultToValue(OpBuilder &builder, Location loc,
                                 OpFoldResult foldResult) {
  if (auto val = llvm::dyn_cast<Value>(foldResult)) {
    return val;
  }
  auto attr = llvm::cast<Attribute>(foldResult);
  auto typedAttr = llvm::dyn_cast<TypedAttr>(attr);
  if (!typedAttr)
    llvm::report_fatal_error("internal error: attr is not TypedAttr");
  auto constOp = builder.create<arith::ConstantOp>(loc, typedAttr);
  return constOp.getResult();
}

LogicalResult reifyOperations(OpBuilder &buidler,
                              ArrayRef<Operation *> opsToReify,
                              ReifiedRankedShapedTypeDims &reifiedReturnShapes,
                              size_t &idxOfSuccessOp) {
  for (size_t i = 0; i < opsToReify.size(); i++) {
    if (succeeded(
            reifyResultShapes(buidler, opsToReify[i], reifiedReturnShapes))) {
      idxOfSuccessOp = i;
      return success();
    }
  }
  return failure();
}

void updateFuncArgs(func::FuncOp func, TypeRange args) {
  FunctionType funcType = FunctionType::get(
      func.getContext(), args, func.getFunctionType().getResults());
  func.setType(funcType);
}

void updateFuncResults(func::FuncOp func, TypeRange results) {
  FunctionType funcType = FunctionType::get(
      func.getContext(), func.getFunctionType().getInputs(), results);
  func.setType(funcType);
}

func::FuncOp copyShapeFunction(MLIRContext *context, func::FuncOp srcFunc) {
  OpBuilder builder(context);
  FunctionType funcType = srcFunc.getFunctionType();
  // Clone the arg attributes as well.
  auto newFunc = builder.create<func::FuncOp>(
      srcFunc.getLoc(),
      /*sym_name=*/
      builder.getStringAttr(hacc::constructHostFunctionName(
          srcFunc.getName().str(),
          hacc::HostFuncType::kInferOutputShapeFunction)),
      /*function_type=*/TypeAttr::get(funcType),
      /*sym_visibility=*/StringAttr(),
      /*arg_attrs=*/srcFunc.getArgAttrsAttr(),
      /*res_attrs=*/ArrayAttr(),
      /*no_inline=*/UnitAttr());
  IRMapping mapper;
  srcFunc.cloneInto(newFunc, mapper);
  return newFunc;
}

struct InferOutShapesPass
    : public impl::InferOutShapesPassBase<InferOutShapesPass> {
public:
  void runOnOperation() override;

private:
  ModuleOp moduleOp;
  void populateShapeFunc();
  void updateShapeFuncArgs();
  LogicalResult simplifyShapeFunc();
};

void InferOutShapesPass::runOnOperation() {
  moduleOp = getOperation();

  populateShapeFunc();

  if (failed(simplifyShapeFunc())) {
    return signalPassFailure();
  }

  updateShapeFuncArgs();
}

void InferOutShapesPass::populateShapeFunc() {
  SymbolTable symbolTable(moduleOp);
  auto context = moduleOp.getContext();

  moduleOp.walk([&](func::FuncOp funcOp) {
    if (!hacc::utils::isDevice(funcOp))
      return;

    func::FuncOp shapeFunc = copyShapeFunction(context, funcOp);
    auto returnOp =
        llvm::cast<func::ReturnOp>(shapeFunc.getBody().front().getTerminator());

    SmallVector<Value> shapeValues;
    SmallVector<Type> shapeTypes;
    for (auto operand : returnOp.getOperands()) {
      auto rootOp = operand.getDefiningOp();
      if (rootOp == nullptr) {
        returnOp->emitError("The parameter is returned directly.");
        return;
      }
      OpBuilder builder{rootOp};

      SmallVector<Operation *> opsToReify{rootOp};
      SmallVector<Value> opResults{operand};

      // Handle the op that implements SameOperandsAndResultShape trait.
      // Only one layer is handled there, no recursive code is perfomed.
      if (rootOp->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
        for (auto rootOperand : rootOp->getOperands()) {
          auto op = rootOperand.getDefiningOp();
          if (op != nullptr) {
            opsToReify.push_back(op);
            opResults.push_back(rootOperand);
          }
        }
      }

      ReifiedRankedShapedTypeDims shapes;
      size_t currentIdx{};
      // if the current kernel func is invalid, continue to process the next one
      if (failed(reifyOperations(builder, opsToReify, shapes, currentIdx))) {
        rootOp->emitError("reifyResultShapes failed");
        return;
      }

      auto reifiedOp = opsToReify[currentIdx];
      auto result = llvm::cast<OpResult>(opResults[currentIdx]);

      auto loc = reifiedOp->getLoc();
      unsigned int resultIdx = result.getResultNumber();
      SmallVector<Value, 10> dimValues;
      for (auto foldResult : shapes[resultIdx]) {
        dimValues.push_back(
            convertOpFoldResultToValue(builder, loc, foldResult));
      }
      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(dimValues.size())}, builder.getIndexType());
      auto fromElementOp =
          builder.create<tensor::FromElementsOp>(loc, shapeType, dimValues);
      shapeValues.push_back(fromElementOp.getResult());
      shapeTypes.push_back(shapeType);
    }

    // Update kernel func attribute
    funcOp->setAttr(hacc::InferOutputShapeFunctionAttr::name,
                    hacc::InferOutputShapeFunctionAttr::get(
                        context, shapeFunc.getNameAttr()));

    // Update shape func type
    updateFuncResults(shapeFunc, shapeTypes);

    // Update return value
    for (size_t i = 0; i < shapeValues.size(); i++) {
      returnOp.setOperand(i, shapeValues[i]);
    }

    // Update shape func attribute
    hacc::utils::setHost(shapeFunc);
    hacc::utils::setHostFuncType(shapeFunc,
                                 hacc::HostFuncType::kInferOutputShapeFunction);

    symbolTable.insert(shapeFunc);
  });
}

LogicalResult InferOutShapesPass::simplifyShapeFunc() {
  auto context = this->moduleOp.getContext();
  RewritePatternSet patterns(context);
  tensor::DimOp::getCanonicalizationPatterns(patterns, context);
  // Currently, the effect of this pattern can be replaced by DimOfDestStyleOp
  // and FoldEmptyTensorWithDimOp. If uncomment this line, you need to modify
  // the MLIR repo to expose DimOfReifyRankedShapedTypeOpInterface.
  // patterns.add<DimOfReifyRankedShapedTypeOpInterface>(context);
  return applyPatternsGreedily(moduleOp, std::move(patterns));
}

void InferOutShapesPass::updateShapeFuncArgs() {
  moduleOp.walk([&](func::FuncOp shapeFunc) {
    auto hostFuncType = hacc::utils::getHostFuncType(shapeFunc);
    if (!hostFuncType ||
        *hostFuncType != hacc::HostFuncType::kInferOutputShapeFunction) {
      return;
    }

    // Filter out output arguments from shape function arguments
    auto inputs = shapeFunc.getFunctionType().getInputs();
    SmallVector<Type> newInputs;
    SmallVector<int> argIndicesToErase;
    for (size_t i = 0; i < inputs.size(); i++) {
      Type ty = inputs[i];
      if (hacc::utils::isKernelArg(shapeFunc, i,
                                   hacc::KernelArgType::kOutput)) {
        argIndicesToErase.push_back(i);
      } else {
        newInputs.push_back(ty);
      }
    }

    // Update shape func type
    updateFuncArgs(shapeFunc, newInputs);

    // Update shape func arguments
    Block &entryBlock = shapeFunc.front();
    std::for_each(argIndicesToErase.begin(), argIndicesToErase.end(),
                  [&entryBlock](int idx) { entryBlock.eraseArgument(idx); });
  });
}

} // namespace
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::hfusion::createInferOutShapesPass() {
  return std::make_unique<InferOutShapesPass>();
}
