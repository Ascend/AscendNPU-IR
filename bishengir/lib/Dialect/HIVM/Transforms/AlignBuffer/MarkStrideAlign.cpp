//===--- MarkStrideAlign.cpp ---- Annotate stride_align marks -------------===//
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
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hivm-mark-stride-align"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_MARKSTRIDEALIGN
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct MarkStrideAlignPass
    : public impl::MarkStrideAlignBase<MarkStrideAlignPass> {
public:
  void runOnOperation() override;
};

} // namespace

LogicalResult markAlignedDim(OpBuilder &builder, Operation *markedOp, Value arg,
                             std::optional<int> alignedDim) {
  if (!alignedDim.has_value()) {
    return success();
  }

  auto alignedDimValue = alignedDim.value();
  LDBG("try to mark align " << alignedDimValue << " for " << arg);

  if (isa<TensorType>(arg.getType())) {
    return markedOp->emitError("Not bufferized.");
  }

  if (isa<MemRefType>(arg.getType())) {
    auto memrefTy = cast<MemRefType>(arg.getType());
    if (!memrefTy.hasRank())
      return markedOp->emitError("UnrankedMemRef not supported.");
    if (!memrefTy.getLayout().isIdentity() &&
        !isa<StridedLayoutAttr>(memrefTy.getLayout()))
      return markedOp->emitError("Non-strided-memref not supported.");

    int rank = memrefTy.getRank();
    if (alignedDimValue > rank - 1) {
      return markedOp->emitError("align dim is too large.");
    }

    builder.setInsertionPoint(markedOp);
    auto hwAlignBytes = getHWAlignBytes(memrefTy.getMemorySpace());
    createAlignMarkOp(builder, markedOp->getLoc(), arg, {alignedDimValue},
                      {static_cast<int>(hwAlignBytes)});
  }
  return success();
}

/// get last un-continuous dim
std::optional<int> getLastUnContinuousDim(
    const SmallVectorImpl<MemRefType> &memRefTypes,
    const SmallVectorImpl<MemRefType> &origMemRefTypes,
    const SmallVector<ReassociationIndices> &continuousReassociations) {
  assert(!continuousReassociations.empty());
  if (llvm::any_of(memRefTypes, [&](MemRefType memRefType) {
        return !isLastMemrefDimUnitStride(memRefType);
      })) {
    LDBG("last dim stride is not 1\n");
    return getLastNotUnitDim(origMemRefTypes, continuousReassociations,
                             continuousReassociations.size() - 1);
  }

  if (continuousReassociations.size() == 1) {
    LDBG("last un-continuous dim does not exist");
    return std::nullopt;
  }

  assert(continuousReassociations.size() > 1);
  return getLastNotUnitDim(origMemRefTypes, continuousReassociations,
                           continuousReassociations.size() - 2);
}

bool isAllRank0(const SmallVectorImpl<MemRefType> &memrefTypes) {
  bool isAllRank0 = llvm::all_of(memrefTypes, [](MemRefType mtype) {
    return mtype.hasRank() && mtype.getRank() == 0;
  });
  return isAllRank0;
}

bool isAnyOfLocalBuffer(const SmallVectorImpl<MemRefType> &memrefTypes) {
  bool anyOfLocalBuffer = llvm::any_of(memrefTypes, [](MemRefType mtype) {
    return isLocalBuffer(getHIVMAddressSpaceAttr(mtype));
  });
  return anyOfLocalBuffer;
}

void MarkStrideAlignPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  WalkResult result = funcOp->walk([&builder](Operation *op) {
    LDBG("Walk operation : " << *op);
    if (!isa<HIVMStructuredOp>(op)) {
      return WalkResult::advance();
    }

    auto hivmOp = cast<HIVMStructuredOp>(op);
    if (!hivmOp.hasPureBufferSemantics()) {
      hivmOp->emitError("Not bufferized.");
      return WalkResult::interrupt();
    }

    if (isa<hivm::VTransposeOp>(op) &&
        isLastDimTranspose(cast<hivm::VTransposeOp>(op))) {
      // already alloc size aligned, no need to do storage align
      return WalkResult::skip();
    }

    auto types = hivmOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
    auto memrefTypes = util::getMemRefTypes(types);
    if (isAllRank0(memrefTypes)) {
      return WalkResult::advance();
    }
    if (!isAnyOfLocalBuffer(memrefTypes)) {
      return WalkResult::advance();
    }

    auto hivmFlattenInterfaceOp = dyn_cast<hivm::FlattenInterface>(op);
    if (hivmFlattenInterfaceOp == nullptr) {
      return WalkResult::skip();
    }
    FlattenOptions flattenOptions;
    flattenOptions.checkMarkStride = true;
    auto flattenResult = hivmFlattenInterfaceOp.getFlattened(flattenOptions);
    if (failed(flattenResult)) {
      op->emitError("unsupport flatten op");
      return WalkResult::skip();
    }
    auto flattenedAssociations = flattenResult->reassociation[0];
    auto flattenedTypes = flattenResult->getOperandTypes(DpsKind::kDpsAll);
    auto flattenedMemrefTypes = util::getMemRefTypes(flattenedTypes);
    auto alignDim = getLastUnContinuousDim(flattenedMemrefTypes, memrefTypes,
                                           flattenedAssociations);
    LDBG("getLastUnContinuousDim " << alignDim.value_or(-1) << "\n");
    for (const auto &oper : hivmOp.getTargetSpaceOperands(
             hivm::AddressSpace::UB, false /*includeTmpBuffer*/)) {
      auto adjustedAlignDim = adjustAlignDim(op, oper, alignDim);
      LDBG("adjustedAlignDim " << adjustedAlignDim.value_or(-1) << "\n");
      if (failed(markAlignedDim(builder, op, oper, adjustedAlignDim)))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createMarkStrideAlignPass() {
  return std::make_unique<MarkStrideAlignPass>();
}
