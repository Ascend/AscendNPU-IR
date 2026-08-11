//===- HFusionGeneralize.cpp ---- convert hfusionOp To linalg.generic ------------------===//
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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
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

    auto isBufferizationOp = [](hfusion::CumsumOp cumsumOp) {
      // 1. Trace Backward to find the input subview and copy
      auto toTensorOp = cumsumOp.getOperand().getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensorOp) return false;

      Value allocMemref = toTensorOp.getMemref();
      memref::CopyOp copyOp = nullptr;
      memref::SubViewOp srcSubView = nullptr;

      // Find the copy operation populating the alloc
      for (auto user : allocMemref.getUsers()) {
        if (auto subViewDst = dyn_cast<memref::SubViewOp>(user)) {
          for (auto subViewUser : subViewDst.getResult().getUsers()) {
            if (auto copy = dyn_cast<mlir::memref::CopyOp>(subViewUser)) {
              if (copy.getTarget() == subViewDst.getResult()) {
                copyOp = copy;
                srcSubView = copy.getSource().getDefiningOp<memref::SubViewOp>();
                break;
              }
            }
          }
        }
        if (copyOp) break;
      }
      if (!copyOp || !srcSubView) return false;

      // 2. Trace Forward through users to find the valid destination chain
      // This explicitly handles the "multiple users of cumsum" condition
      bufferization::MaterializeInDestinationOp matchingMatOp = nullptr;

      for (auto user : cumsumOp.getResult().getUsers()) {
        if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
          // Check if this slice feeds into a materialize operation
          for (auto sliceUser : extractSliceOp.getResult().getUsers()) {
            if (auto matOp = dyn_cast<bufferization::MaterializeInDestinationOp>(sliceUser)) {
              if (matOp.getSource() == extractSliceOp.getResult()) {
                if (auto dstSubView = matOp.getDest().getDefiningOp<memref::SubViewOp>()) {
                  // 1. Get the subview representing the local destination slot of the copy operation
                  auto localAllocDstSubview = copyOp.getTarget().getDefiningOp<memref::SubViewOp>();
                  if (!localAllocDstSubview) continue;

                  // 2. Strict size & stride equality across all domains
                  // (They must all slice a block of the same dimensions)
                  const bool sizesMatch = (srcSubView.getMixedSizes() == extractSliceOp.getMixedSizes() &&
                                           dstSubView.getMixedSizes() == extractSliceOp.getMixedSizes());

                  const bool stridesMatch = (srcSubView.getMixedStrides() == extractSliceOp.getMixedStrides() &&
                                             dstSubView.getMixedStrides() == extractSliceOp.getMixedStrides());

                  // 3. Offset cross-verification:
                  // The offset into the scratchpad %alloc during the copy step MUST match the 
                  // offset extracted from the tensor domain after the cumsum step.
                  const bool offsetsMatch = (localAllocDstSubview.getMixedOffsets() == extractSliceOp.getMixedOffsets());

                  if (sizesMatch && stridesMatch && offsetsMatch) {
                      // Found a valid sub-graph matching the pattern
                      matchingMatOp = matOp;
                      break;
                  }
                }
              }
            }
          }
        }
        if (matchingMatOp) break;
      }

      return matchingMatOp ? true : false;
    };

    bool cancel = isSubOp(c.getInput().getDefiningOp());
    for (Operation *user : c->getResult(0).getUsers()) {
      if (cancel)
        break;
      cancel = isSubOp(user);
    }

    if (!cancel) {
      cancel = isBufferizationOp(c);
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
