//===-------------------- ConvertLayoutUtils.cpp --------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#define DEBUG_TYPE "convert-layout-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

namespace {

/// Users of `tensor` are only `convertOp` or `annotation.mark`.
bool tensorFeedsOnlyConvert(Value tensor, ConvertLayoutOp convertOp) {
  return llvm::all_of(tensor.getUsers(), [&](Operation *user) {
    return user == convertOp.getOperation() || isa<annotation::MarkOp>(user);
  });
}

/// Walk convert-src toward alloc/empty; erase marks then unused ops.
void eraseDeadDummyChain(PatternRewriter &rewriter, Value root) {
  SmallVector<Operation *, 8> chain;
  Value v = root;
  while (Operation *def = v.getDefiningOp()) {
    if (isa<bufferization::ToTensorOp, ViewLikeOpInterface>(def)) {
      chain.push_back(def);
      if (def->getNumOperands() == 0)
        break;
      v = def->getOperand(0);
      continue;
    }
    if (isa<memref::AllocOp, tensor::EmptyOp>(def))
      chain.push_back(def);
    break;
  }
  for (Operation *dead : chain) {
    for (Operation *user : llvm::make_early_inc_range(dead->getUsers())) {
      if (isa<annotation::MarkOp>(user))
        rewriter.eraseOp(user);
    }
    if (dead->use_empty())
      rewriter.eraseOp(dead);
  }
}

} // namespace

constexpr llvm::StringLiteral convertLayoutNotToPropagateUp =
    "not_to_propagate_up";

namespace mlir::hivm {

//===----------------------------------------------------------------------===//
// Common Helpers
//===----------------------------------------------------------------------===//

/// Compute batch index bias from rank
int computeBatchIndexBias(size_t rank) { return (rank == 3) ? 1 : 0; }
//===----------------------------------------------------------------------===//
// Public API - Unified Target Shape Computation
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<OpFoldResult>> computeMixedTargetLayoutShape(
    ArrayRef<OpFoldResult> currentShape, DataLayoutAttr srcLayout,
    DataLayoutAttr dstLayout, OpBuilder &builder, Location loc) {

  LDBG("=== computeMixedTargetLayoutShape ===");

  bool srcIsND = srcLayout.isNDLayout();
  bool dstIsND = dstLayout.isNDLayout();

  // ND -> Fractal (matrix nZ/zN or scale zZ/nN). Scale tile ordering is handled
  // inside computeMixedNDToFractalShape via dstLayout.isScaleFractalLayout().
  if (srcIsND && !dstIsND) {
    return computeMixedNDToFractalShape(currentShape, srcLayout, dstLayout,
                                        builder, loc);
  }

  // Fractal -> ND conversion
  if (!srcIsND && dstIsND) {
    return computeMixedFractalToNDShape(currentShape, srcLayout, dstLayout,
                                        builder, loc);
  }

  return failure();
}

void markAsNotPropagatingUp(PatternRewriter &rewriter, ConvertLayoutOp op) {
  op->setAttr(convertLayoutNotToPropagateUp, rewriter.getBoolAttr(true));
}

bool isPropagatingUp(ConvertLayoutOp op) {
  auto dstLayout = op.getDstLayout().getDataLayout();
  bool propagatesUp = (dstLayout == DataLayout::Fractal) ||
                      (dstLayout == DataLayout::SCALEA_zZ) ||
                      (dstLayout == DataLayout::SCALEB_nN);
  return propagatesUp && !(op->getAttr(convertLayoutNotToPropagateUp));
}

bool isPropagatingDown(ConvertLayoutOp op) { return !isPropagatingUp(op); }

bool isLayoutAgnosticOp(Operation *op) {
  // TODO: When propagating is fixed, can remove this following line
  if (!op)
    return false;
  if (auto vbrcOp = dyn_cast<VBrcOp>(op)) {
    return isScalarLike(vbrcOp.getSrc().getType());
  }
  bool isAllowed = mlir::hivm::detail::isElemwiseNaryOpImpl(op);
  return isAllowed;
}

/// Check if operation is a fixpipe operation
bool isFixpipeOp(Operation *op) { return isa_and_present<hivm::FixpipeOp>(op); }

/// Create a ConvertLayoutOp with the same direction attribute
Value createConvertLayoutLike(PatternRewriter &rewriter,
                              ConvertLayoutOp templateOp, Value input) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  auto converted = cast<ConvertLayoutOp>(rewriter.clone(*templateOp));
  converted->setLoc(input.getLoc());
  converted.getSourceMutable().assign(input);
  auto newReplacedElementType =
      cast<ShapedType>(converted.getResult().getType())
          .clone(getElementTypeOrSelf(input));
  converted.getResult().setType(newReplacedElementType);
  return converted.getResult();
}

Value createInverseConvertLayout(PatternRewriter &rewriter,
                                 ConvertLayoutOp templateOp, Value input) {
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  auto newReplacedElementType =
      cast<ShapedType>(templateOp.getSource().getType())
          .clone(getElementTypeOrSelf(input));
  auto converted = rewriter.create<ConvertLayoutOp>(
      input.getLoc(), newReplacedElementType, input,
      templateOp.getDstLayoutAttr(), templateOp.getSrcLayoutAttr());
  return converted.getResult();
}

bool isUninitL1NDConvertLayout(ConvertLayoutOp op) {
  Value src = op.getSource();

  if (auto emptyOp = src.getDefiningOp<tensor::EmptyOp>()) {
    auto spaceAttr =
        emptyOp->getAttrOfType<AddressSpaceAttr>(AddressSpaceAttr::name);
    if (!spaceAttr || spaceAttr.getAddressSpace() != AddressSpace::L1)
      return false;
    return tensorFeedsOnlyConvert(emptyOp.getResult(), op);
  }

  auto toTensor = src.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return false;

  auto alloc = getMemRefAlloc(toTensor.getMemref());
  if (failed(alloc))
    return false;
  auto space = getOptionalHIVMAddressSpace(alloc->getType());
  if (!space || *space != AddressSpace::L1)
    return false;

  SmallVector<Value, 8> worklist{alloc->getResult()};
  SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (Operation *user : cur.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;
      if (isa<ViewLikeOpInterface>(user)) {
        llvm::append_range(worklist, user->getResults());
        continue;
      }
      auto toTensorUser = dyn_cast<bufferization::ToTensorOp>(user);
      if (!toTensorUser)
        return false;
      if (!tensorFeedsOnlyConvert(toTensorUser.getResult(), op))
        return false;
    }
  }
  return true;
}

void replaceUninitL1NDConvertWithEmpty(PatternRewriter &rewriter,
                                       ConvertLayoutOp op) {
  Value src = op.getSource();
  auto resultTy = cast<RankedTensorType>(op.getType());
  rewriter.setInsertionPoint(op);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getMixedOutputShape(), resultTy.getElementType());
  emptyOp->setAttr(AddressSpaceAttr::name,
                   rewriter.getAttr<AddressSpaceAttr>(AddressSpace::L1));
  rewriter.replaceOp(op, emptyOp.getResult());
  eraseDeadDummyChain(rewriter, src);
}
} // namespace mlir::hivm
