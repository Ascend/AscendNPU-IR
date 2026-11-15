//===- VectorizeResult.cpp - HIVM vectorization cleanup and checks --------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/VectorizeResult.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/TransformOps/Partition.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeLinalg.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <string>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
constexpr StringLiteral kVectorFunctionAttrName = "hivm.vector_function";
constexpr StringLiteral kNoInlineAttrName = "no_inline";
constexpr StringLiteral kHIVMDialectNamespace = "hivm";

void addCanonicalizationPatterns(RewritePatternSet &patterns,
                                 MLIRContext *context) {
  for (Dialect *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);
}

LogicalResult cleanupFunction(func::FuncOp function, std::string &error) {
  MLIRContext *context = function.getContext();
  RewritePatternSet patterns(context);
  addCanonicalizationPatterns(patterns, context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

  GreedyRewriteConfig config;
  if (failed(applyPatternsGreedily(function, std::move(patterns), config))) {
    error = "canonicalization failed for vector function '" +
            function.getSymName().str() + "'";
    return failure();
  }

  IRRewriter rewriter(context);
  DominanceInfo domInfo(function);
  eliminateCommonSubExpressions(rewriter, domInfo, function);
  if (failed(flattenVectorComputations(function))) {
    error = "failed to flatten cleaned vector function '" +
            function.getSymName().str() + "'";
    return failure();
  }
  if (failed(verify(function))) {
    error = "verification failed after cleaning vector function '" +
            function.getSymName().str() + "'";
    return failure();
  }
  return success();
}

LogicalResult checkVectorFunctionAttrs(func::FuncOp function,
                                       std::string &error) {
  if (!function->hasAttr(kVectorFunctionAttrName)) {
    error = "function '" + function.getSymName().str() +
            "' is not marked hivm.vector_function";
    return failure();
  }
  if (!function->hasAttr(kNoInlineAttrName)) {
    error = "function '" + function.getSymName().str() + "' is not no_inline";
    return failure();
  }
  return success();
}

bool isHIVMOp(Operation *op) {
  return op->getName().getDialectNamespace() == kHIVMDialectNamespace;
}

LogicalResult checkVectorFunctionBody(func::FuncOp function,
                                      std::string &error) {
  WalkResult walk = function.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op)) {
      error = "linalg operation left in vector function '" +
              function.getSymName().str() + "'";
      return WalkResult::interrupt();
    }
    if (isVectorizationCompute(*op)) {
      error = "vectorizable operation '" + op->getName().getStringRef().str() +
              "' left in vector function '" + function.getSymName().str() + "'";
      return WalkResult::interrupt();
    }
    if (isHIVMOp(op)) {
      error = "HIVM operation '" + op->getName().getStringRef().str() +
              "' must remain in the caller of vector function '" +
              function.getSymName().str() + "'";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walk.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult mlir::hivm::cleanupFunctions(ArrayRef<func::FuncOp> functions,
                                           std::string &error) {
  error.clear();
  for (func::FuncOp function : functions)
    if (failed(cleanupFunction(function, error)))
      return failure();
  return success();
}

FailureOr<int64_t>
mlir::hivm::checkVectorizedFunctions(ArrayRef<func::FuncOp> functions,
                                     std::string &error) {
  error.clear();
  for (func::FuncOp function : functions) {
    if (failed(checkVectorFunctionAttrs(function, error)) ||
        failed(checkVectorFunctionBody(function, error)))
      return failure();
  }
  return static_cast<int64_t>(functions.size());
}

DiagnosedSilenceableFailure HIVMCleanupOp::apply(TransformRewriter &,
                                                 TransformResults &results,
                                                 TransformState &state) {
  SmallVector<func::FuncOp> functions;
  for (Operation *payloadOp : state.getPayloadOps(getVfs())) {
    auto function = dyn_cast<func::FuncOp>(payloadOp);
    if (!function)
      return emitDefiniteFailure() << "expected only function operations";
    functions.push_back(function);
  }

  std::string error;
  if (failed(cleanupFunctions(functions, error)))
    return emitDefiniteFailure() << error;

  results.set(cast<OpResult>(getResult()),
              SmallVector<Operation *>(functions.begin(), functions.end()));
  return DiagnosedSilenceableFailure::success();
}

void HIVMCleanupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getVfsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HIVMCheckResultOp::apply(TransformRewriter &,
                                                     TransformResults &results,
                                                     TransformState &state) {
  SmallVector<func::FuncOp> functions;
  for (Operation *payloadOp : state.getPayloadOps(getVfs())) {
    auto function = dyn_cast<func::FuncOp>(payloadOp);
    if (!function)
      return emitDefiniteFailure() << "expected only function operations";
    functions.push_back(function);
  }

  std::string error;
  FailureOr<int64_t> cost = checkVectorizedFunctions(functions, error);
  if (failed(cost))
    return emitDefiniteFailure() << error;

  results.setParams(cast<OpResult>(getResult()),
                    {Builder(getContext()).getI64IntegerAttr(*cost)});
  return DiagnosedSilenceableFailure::success();
}

void HIVMCheckResultOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getVfsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}
