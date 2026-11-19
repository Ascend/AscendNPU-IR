//===- MarkMatmulRowCollapses.cpp - Mark ND row merges ----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
#define GEN_PASS_DEF_MARKMATMULROWCOLLAPSES
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct MarkMatmulRowCollapsesPass
    : impl::MarkMatmulRowCollapsesBase<MarkMatmulRowCollapsesPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto module = funcOp->getParentOfType<ModuleOp>();
    if (!module)
      return;
    auto target = hacc::utils::getTargetDevice(module);
    if (!target || !hacc::utils::isMemBasedArch(*target))
      return;

    constexpr llvm::StringLiteral hint = "maybeUnCollapsibleReshape";
    OpBuilder builder(&getContext());
    funcOp.walk([&](MmadL1Op matmul) {
      for (Value operand : {matmul.getA(), matmul.getB()}) {
        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandType || operandType.getRank() != 2 ||
            !operandType.hasStaticShape())
          continue;
        for (Operation *producer :
             traceDefOps<tensor::CollapseShapeOp>(operand)) {
          auto collapse = cast<tensor::CollapseShapeOp>(producer);
          auto srcType = collapse.getSrcType();
          if (srcType.getRank() != 3 ||
              collapse.getResultType().getRank() != 2 ||
              !srcType.hasStaticShape() || srcType.getDimSize(0) == 1 ||
              srcType.getDimSize(1) == 1 ||
              collapse.getReassociationIndices().front() !=
                  ReassociationIndices{0, 1})
            continue;
          if (utils::getAnnotateOpWithAttr(collapse.getResult(), hint))
            continue;

          // Merge rows in ND before converting the complete matrix to NZ.
          builder.setInsertionPointAfter(collapse);
          auto mark = builder.create<annotation::MarkOp>(collapse.getLoc(),
                                                         collapse.getResult());
          mark->setAttr(hint, builder.getUnitAttr());
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createMarkMatmulRowCollapsesPass() {
  return std::make_unique<MarkMatmulRowCollapsesPass>();
}
