//===- AnnotateVF.cpp - HIVM vector function annotation -------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace {
constexpr StringLiteral kVectorFunctionAttrName = "hivm.vector_function";
constexpr StringLiteral kNoInlineAttrName = "no_inline";

bool containsOnlyFunctions(ArrayRef<Operation *> operations) {
  return llvm::all_of(operations,
                      [](Operation *op) { return isa<func::FuncOp>(op); });
}

void annotateVectorFunction(Operation *operation, UnitAttr unit) {
  operation->setAttr(kVectorFunctionAttrName, unit);
  operation->setAttr(kNoInlineAttrName, unit);
  if (operation->hasAttr(TFuncCoreTypeAttr::name)) {
    operation->setAttr(
        TFuncCoreTypeAttr::name,
        TFuncCoreTypeAttr::get(operation->getContext(), TFuncCoreType::AIV));
    operation->removeAttr(TCoreRatioAttr::name);
    operation->setAttr("mix_mode",
                       StringAttr::get(operation->getContext(), "aiv"));
  }
  auto function = cast<func::FuncOp>(operation);
  auto module = function->getParentOfType<ModuleOp>();
  if (!module)
    return;
  module.walk([&](func::CallOp call) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (callee != function)
      return;
    call->setAttr(kVectorFunctionAttrName, unit);
    call->setAttr(kNoInlineAttrName, unit);
  });
}

} // namespace

DiagnosedSilenceableFailure HIVMAnnotateVFOp::apply(TransformRewriter &,
                                                    TransformResults &results,
                                                    TransformState &state) {
  auto payloadOps = state.getPayloadOps(getFuncs());
  SmallVector<Operation *> functions(payloadOps.begin(), payloadOps.end());
  if (!containsOnlyFunctions(functions))
    return emitDefiniteFailure() << "expected only function operations";

  UnitAttr unit = UnitAttr::get(getContext());
  for (Operation *function : functions)
    annotateVectorFunction(function, unit);

  results.set(cast<OpResult>(getResult()), functions);
  return DiagnosedSilenceableFailure::success();
}

void HIVMAnnotateVFOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getFuncsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}
