//===- InlineOTFBroadcast.cpp ----- inline OTF broadcast ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace mlir {
#define GEN_PASS_DEF_INLINEOTFBROADCAST
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-otf-broadcast-ops"

using namespace mlir;
using namespace mlir::hivm;

namespace {

bool isValidUser(Operation *user, AxisKind axisKind) {
  // TODO: replace whitelist with trait
  bool isInLastAxisWhitelist =
      llvm::isa<VAddOp, VMulOp, VMaxOp, VMinOp, VSubOp, VDivOp, VAndOp, VOrOp,
                VShLOp, VShROp, VNotOp, VAbsOp, VLnOp, VReluOp, VExpOp,
                VRsqrtOp, VSqrtOp>(user);
  if (llvm::isa<VAbsOp>(user)) {
    auto vabsOp = dyn_cast<VAbsOp>(user);
    Value src = vabsOp.getSrc()[0];
    Type elemType = getElementTypeOrSelf(src.getType());
    // In the vabs int16/int32 scenario, the decompose composite
    // implementation is used and the last axis inline brc is not
    // currently supported.
    if (elemType.isInteger(16) || elemType.isInteger(32)) {
      isInLastAxisWhitelist = false;
    }
  }
  bool isBroadcastable = user->hasTrait<mlir::OpTrait::BroadcastableOTF>();
  bool isBinary = user->hasTrait<OpTrait::ElementwiseNaryOpTrait<2>::Impl>();
  bool hivmStructured = dyn_cast_or_null<HIVMStructuredOp>(user);
  return (axisKind == AxisKind::LAST)
             ? isInLastAxisWhitelist
             : (isBroadcastable && isBinary && hivmStructured);
}

void updateBroadcastAttr(Operation *user, int dim) {
  std::set<int64_t> userBroadcastDims = {dim};
  auto hivmStructed = dyn_cast_or_null<HIVMStructuredOp>(user);
  assert(hivmStructed);
  if (auto originalAttr = user->getAttrOfType<DenseI64ArrayAttr>(
          hivmStructed.getBroadcastAttrString()))
    for (auto dim : originalAttr.asArrayRef())
      userBroadcastDims.insert(dim);
  llvm::SmallVector<int64_t, 4> broadcastVector(userBroadcastDims.begin(),
                                                userBroadcastDims.end());
  user->setAttr(
      hivmStructed.getBroadcastAttrString(),
      mlir::DenseI64ArrayAttr::get(user->getContext(), broadcastVector));
}

struct VBrcInlinePattern : public OpRewritePattern<hivm::VBrcOp> {
  using OpRewritePattern<hivm::VBrcOp>::OpRewritePattern;

  /// This function inline broadcastOTF. e.g.
  ///   %brc = hivm.hir.vbrc
  ///      ins(%arg0 : tensor<1x128xf32>)
  ///      outs(%empty0 : tensor<5x128xf32>)
  ///      broadcast_dims = [0] -> tensor<5x128xf32>
  ///   %ret = hivm.hir.vmul
  ///      ins(%brc, %arg1 : tensor<5x128xf32>, tensor<5x128xf32>)
  ///      outs(%empty1 : tensor<5x128xf32>) -> tensor<5x128xf32>
  /// converts to
  ///   %ret = hivm.hir.vmul
  ///      ins(%arg0, %arg1 : tensor<1x128xf32>, tensor<5x128xf32>)
  ///      outs(%0 : tensor<5x128xf32>)
  ///      broadcast = [0] -> tensor<5x128xf32>
  LogicalResult matchAndRewrite(hivm::VBrcOp vbrcOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "operation: " << *vbrcOp << "\n");
    Value src = vbrcOp.getSrc();
    Value dst = vbrcOp.getResult()[0];
    if (!isa<TensorType>(src.getType()))
      return failure();
    // Initialize the variables needed
    ArrayRef<int64_t> broadDims = vbrcOp.getBroadcastDims();
    TensorType dstType = cast<TensorType>(dst.getType());
    int64_t rank = dstType.getRank();
    AxisKind axisKind = utils::getAxisKind(broadDims[0], rank);
    Type elementType = dstType.getElementType();
    // Only non-i64/bool can be inlined
    if (elementType == rewriter.getI64Type() ||
        elementType == rewriter.getI1Type())
      return failure();
    // Only broadcast with 1 broadcast axis can be inlined
    // TODO: need to confirm multi-axis broadcast condition
    if (broadDims.size() != 1)
      return failure();
    bool isBrcInlined = false;
    for (Operation *user : dst.getUsers()) {
      if (!isValidUser(user, axisKind))
        continue;
      auto hivmOp = dyn_cast<HIVMStructuredOp>(user);
      LLVM_DEBUG(llvm::dbgs() << "validUser: " << *user << "\n");
      // IR modification
      isBrcInlined = true;
      updateBroadcastAttr(hivmOp, broadDims[0]);
      for (auto &opOperand : hivmOp.getDpsInputOperands())
        if (opOperand->get() == dst) {
          // Inline the broadcast src in the user's operands
          rewriter.modifyOpInPlace(opOperand->getOwner(),
                                   [&]() { opOperand->set(src); });
        }
    }
    return success(isBrcInlined);
  }
};

struct InlineOTFBroadcastPass
    : public impl::InlineOTFBroadcastBase<InlineOTFBroadcastPass> {
public:
  void runOnOperation() override;
};
} // namespace

void InlineOTFBroadcastPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<VBrcInlinePattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> mlir::hivm::createInlineOTFBroadcastPass() {
  return std::make_unique<InlineOTFBroadcastPass>();
}
