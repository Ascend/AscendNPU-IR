//===- SelectHIVMVectorize.cpp - Select HIVM vectorization work -----------===//
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

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/HIVMVectorizeLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_SELECTHIVMVECTORIZE
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isGlueOp(Operation *op) {
  return isa<func::ReturnOp, arith::ConstantOp, tensor::EmptyOp>(op);
}

bool hasStaticShapes(Operation *op) {
  auto isShapeStatic = [](Type type) {
    auto shapedType = dyn_cast<ShapedType>(type);
    return !shapedType || shapedType.hasStaticShape();
  };
  return llvm::all_of(op->getOperandTypes(), isShapeStatic) &&
         llvm::all_of(op->getResultTypes(), isShapeStatic);
}

void selectBlock(Block &block) {
  SmallVector<Operation *> candidates;
  for (auto &op : block) {
    if (isGlueOp(&op))
      continue;
    if (!hivm::vectorize::hasVectorizableLowering(&op))
      return;
    if (!hasStaticShapes(&op))
      return;
    candidates.push_back(&op);
  }

  if (candidates.empty())
    return;

  auto *context = block.getParentOp()->getContext();
  for (auto *op : candidates)
    op->setAttr("hivm.defer_vectorize", UnitAttr::get(context));
}

void selectHIVMVectorizePayload(func::FuncOp func) {
  func.walk([](Block *block) { selectBlock(*block); });
}

struct SelectHIVMVectorizePass
    : public mlir::impl::SelectHIVMVectorizeBase<SelectHIVMVectorizePass> {
  void runOnOperation() override {
    getOperation()->walk([](func::FuncOp func) {
      if (hfusion::isAIVFunc(func))
        selectHIVMVectorizePayload(func);
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::hfusion::createSelectHIVMVectorizePass() {
  return std::make_unique<SelectHIVMVectorizePass>();
}
