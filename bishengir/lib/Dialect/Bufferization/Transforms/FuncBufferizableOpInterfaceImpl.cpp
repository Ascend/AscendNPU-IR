//===- FuncBufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::bufferization::func_ext;
using namespace mlir::func;

static constexpr llvm::StringLiteral KFoldOffsetMarker = "fold_offset_into_ptr";

static FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

namespace CallOpInterfaceOverwriter {
static bool isNotConflicting(Operation *op, OpOperand *uRead, OpOperand *uWrite,
                             const AnalysisState &state) {
  /// Same callOp and different value are the preconditions to have no
  /// conflict. Because OpOperand can be read and written both.
  ///
  /// case:
  ///
  /// @test_vf(%arg: tensor<16xf32>) -> tensor<16xf32> {
  ///   %0 = read %arg : tensor<16xf32>
  ///   %1 = vadd %0, %cst
  ///   %2 = write %1 to %arg : tensor<16xf32>
  ///   return %2 : tensor<16xf32>
  ///  }

  /// @AIV (%arg0 = memref<16xf32>) {
  ///   %1 = vbrc %0 : tensor<16xf32>
  ///   for
  ///     %2 = call @test_vf(%1) {hivm.vector_function, no_inline} :
  ///     store %2 to %arg0 : memref<16xf32>
  ///   endfor
  ///  }

  /// def:vbrcOp, uRead:callOp OpOperand #0(%1), uWrite:callOp OpOperand #0(%1).
  /// In this case, we expect a RaW conflict because def is out of loop.
  /// But wouldCreateReadAfterWriteInterference will return noConflict. So we
  /// add extra check to make sure the uRead and uWrite are not the same value.

  if (uRead->getOwner() != uWrite->getOwner() || uRead->get() == uWrite->get())
    return false;

  func::CallOp callOp = cast<func::CallOp>(op);
  FuncOp funcOp = getCalledFunction(callOp);
  if (!funcOp)
    return false;

  if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
    return false;

  auto readArg = funcOp.getArgument(uRead->getOperandNumber());
  auto writeArg = funcOp.getArgument(uWrite->getOperandNumber());
  DominanceInfo domInfo(funcOp);
  return !wouldCreateReadAfterWriteInterference(
      readArg, writeArg, domInfo,
      const_cast<OneShotAnalysisState &>(
          static_cast<const OneShotAnalysisState &>(state)));
}

static LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                               const BufferizationOptions &options) {
  func::CallOp callOp = cast<func::CallOp>(op);

  SmallVector<Type> resultTypes;
  for (Value result : callOp.getResults()) {
    Type returnType = result.getType();
    if (!isa<TensorType>(returnType)) {
      resultTypes.push_back(returnType);
      continue;
    }

    FailureOr<BaseMemRefType> resultType =
        bufferization::getBufferType(result, options);
    if (failed(resultType))
      return failure();
    resultTypes.push_back(*resultType);
  }

  SmallVector<Value> newOperands;
  FuncOp funcOp = getCalledFunction(callOp);
  assert(funcOp && "expected CallOp to a FuncOp");
  FunctionType funcType = funcOp.getFunctionType();

  for (OpOperand &opOperand : callOp->getOpOperands()) {
    if (!isa<TensorType>(opOperand.get().getType())) {
      newOperands.push_back(opOperand.get());
      continue;
    }

    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, opOperand.get(), options);
    if (failed(maybeBuffer))
      return failure();
    Value buffer = *maybeBuffer;

    auto memRefType = funcType.getInput(opOperand.getOperandNumber());
    if (buffer.getType() != memRefType) {
      assert(memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
             "CallOp::bufferize: cast incompatible");

      auto castOp =
          rewriter.create<memref::CastOp>(callOp.getLoc(), memRefType, buffer);

      /// Add KFoldOffsetMarker start
      auto srcType = dyn_cast<MemRefType>(buffer.getType());
      auto dstType = dyn_cast<MemRefType>(memRefType);
      SmallVector<int64_t> srcStrides;
      SmallVector<int64_t> dstStrides;
      int64_t srcOffset;
      int64_t dstOffset;
      if (succeeded(getStridesAndOffset(srcType, srcStrides, srcOffset)) &&
          succeeded(getStridesAndOffset(dstType, dstStrides, dstOffset))) {
        bool srcStridesStatic = llvm::all_of(
            srcStrides, [](int64_t s) { return !ShapedType::isDynamic(s); });
        bool dstStridesStatic = llvm::all_of(
            dstStrides, [](int64_t s) { return !ShapedType::isDynamic(s); });
        if (srcStridesStatic && srcOffset == ShapedType::kDynamic &&
            dstStridesStatic && dstOffset == 0)
          castOp->setAttr(KFoldOffsetMarker, rewriter.getUnitAttr());
      }
      /// Add KFoldOffsetMarker end

      buffer = castOp.getResult();
    }
    newOperands.push_back(buffer);
  }

  Operation *newCallOp = rewriter.create<func::CallOp>(
      callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
  newCallOp->setAttrs(callOp->getAttrs());

  replaceOpWithBufferizedValues(rewriter, callOp, newCallOp->getResults());

  return success();
}

static FailureOr<BaseMemRefType>
getBufferType(Operation *op, Value value, const BufferizationOptions &options,
              const SmallVector<Value> &invocationStack) {
  auto callOp = cast<func::CallOp>(op);
  FuncOp funcOp = getCalledFunction(callOp);
  assert(funcOp && "expected CallOp to a FuncOp");

  FunctionType funcType = funcOp.getFunctionType();
  Type resultType =
      funcType.getResult(cast<OpResult>(value).getResultNumber());
  if (auto memRefType = dyn_cast<BaseMemRefType>(resultType))
      return memRefType;

  // Outlined callees may still have tensor signatures when one-shot
  // bufferization resolves tensor copy conflicts inside enclosing loops.
  // Resolve the buffer type by following the equivalence chain through the
  // callee's body via the return operand.
  if (funcOp && !funcOp.isExternal()) {
    func::ReturnOp returnOp;
    for (Block &block : funcOp.getBody()) {
      if (auto candidate = dyn_cast<func::ReturnOp>(block.getTerminator())) {
        returnOp = candidate;
        break;
      }
    }
    if (returnOp) {
      Value returnVal =
          returnOp.getOperand(cast<OpResult>(value).getResultNumber());
      if (isa<TensorType>(returnVal.getType())) {
        SmallVector<Value> mutableStack(invocationStack);
        return bufferization::getBufferType(returnVal, options, mutableStack);
      }
    }
  }

  assert(isa<TensorType>(resultType) && "expected tensor result type");
  auto memSpace = options.defaultMemorySpaceFn(cast<TensorType>(resultType));
  if (!memSpace.has_value())
      return op->emitError("could not infer memory space");
  return getMemRefType(value, options, /*layout=*/{}, *memSpace);
}
} // namespace CallOpInterfaceOverwriter

RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/getBufferType,
    /*Impl=*/
    &CallOpInterfaceOverwriter::getBufferType);

RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/isNotConflicting,
    /*Impl=*/
    &CallOpInterfaceOverwriter::isNotConflicting);

RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferize,
    /*Impl=*/
    &CallOpInterfaceOverwriter::bufferize);

void mlir::bufferization_ext::registerFuncBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {}