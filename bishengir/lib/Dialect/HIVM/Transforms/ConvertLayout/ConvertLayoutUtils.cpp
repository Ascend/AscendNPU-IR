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
#include "bishengir/Conversion/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

#define DEBUG_TYPE "convert-layout-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

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

  // ND -> Fractal conversion
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
  return (op.getDstLayout().getDataLayout() == DataLayout::Fractal) &&
          !(op->getAttr(convertLayoutNotToPropagateUp));
}

bool isPropagatingDown(ConvertLayoutOp op) { return !isPropagatingUp(op); }

bool isLayoutAgnosticOp(Operation *op) {
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
} // namespace mlir::hivm