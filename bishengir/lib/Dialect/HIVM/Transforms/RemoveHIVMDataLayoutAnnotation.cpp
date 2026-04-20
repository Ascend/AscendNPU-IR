//===- RemoveHIVMDataLayoutAnnotation.cpp - Remove layout annotation ------===//
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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "hivm-remove-layout-annotation"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_REMOVEHIVMDATALAYOUTANNOTATION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct RemoveHIVMDataLayoutAnnotationPass
    : public impl::RemoveHIVMDataLayoutAnnotationBase<
          RemoveHIVMDataLayoutAnnotationPass> {
  using Base::Base;
  void runOnOperation() override;
};

void RemoveHIVMDataLayoutAnnotationPass::runOnOperation() {
  auto funcOp = getOperation();

  funcOp.walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttr(hivm::kHIVMDataLayoutAttrName)) {
      LDBG("Removing hivm_data_layout attr from: " << markOp);
      markOp->removeAttr(hivm::kHIVMDataLayoutAttrName);
    }
  });
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createRemoveHIVMDataLayoutAnnotationPass() {
  return std::make_unique<RemoveHIVMDataLayoutAnnotationPass>();
}