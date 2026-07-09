//===- HFusionGeneralize.cpp ---- convert hfusionOp To linalg.generic -----===//
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

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
namespace mlir {
#define GEN_PASS_DEF_HFUSIONGENERALIZEPASS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hfusion-generalize"

using namespace mlir;

namespace {
struct HFusionGeneralizePass
    : public impl::HFusionGeneralizePassBase<HFusionGeneralizePass> {
public:
  void runOnOperation() override;
};

void HFusionGeneralizePass::runOnOperation() {
  auto module = getOperation();
  // Cumsum cancellation detector (runs in the active RegBase HFusion pipeline, on the
  // clean pre-fusion IR where cumsum's result is still consumed by a bare
  // linalg.elemwise_binary<sub>). Tag a float 2D dim0 cumsum whose result feeds a
  // subtraction (X - cumsum(X), e.g. dg): fp32 cancellation amplifies the scan's
  // rounding error -> must use the TwoSum-compensated template. The "needs_compensation"
  // attr is copied to the VCumsumOp by HFusionToHIVMCumOp and read by
  // VCumsumOp::getOpLibraryCallName (appends "_comp").
  module.walk([&](hfusion::CumsumOp c) {
    auto resTy = dyn_cast<ShapedType>(c->getResult(0).getType());
    llvm::ArrayRef<int64_t> cumDims = c.getCumDims();
    // Compensation scope (must mirror the available "_comp" template symbols):
    // f32, 2D dim0, 3D dim0/dim1.
    if (!resTy || !resTy.getElementType().isF32() || cumDims.size() != 1)
      return;
    int64_t rank = resTy.getRank(), dim = cumDims[0];
    if (!((rank == 2 && dim == 0) || (rank == 3 && (dim == 0 || dim == 1))))
      return;
    auto isSubOp = [](Operation *op) {
      if (!op)
        return false;
      if (isa<arith::SubFOp>(op))
        return true;
      if (auto eb = dyn_cast<linalg::ElemwiseBinaryOp>(op))
        return eb.getFun() == linalg::BinaryFn::sub;
      return false;
    };

    bool cancel = isSubOp(c.getInput().getDefiningOp());
    for (Operation *user : c->getResult(0).getUsers()) {
      if (cancel)
        break;
      cancel = isSubOp(user);
    }
    if (cancel)
      c->setAttr("needs_compensation", UnitAttr::get(c->getContext()));
  });
  module.walk([&](hfusion::GatherOp op) {
    IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    (void)generalizeNamedOp(rewriter, op);
  });
}
} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionGeneralizePass() {
  return std::make_unique<HFusionGeneralizePass>();
}
