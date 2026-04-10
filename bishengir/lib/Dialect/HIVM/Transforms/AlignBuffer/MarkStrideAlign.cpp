//===--- MarkStrideAlign.cpp ---- Annotate stride_align marks -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Return "true", if the size of lowest dimension equal with the stride of
// sub-low dimension or the stride of sub-low dimension is UB align.
// Such as memref<3x13xi8, strided<[16, 1]>>
static bool isNotTailJumpOrStrideAlign(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  int64_t dim = type.getRank();
  int64_t hwAlignBits = static_cast<int64_t>(getHWAlignBytes(type.getMemorySpace()) * 8);
  int64_t dataWidth = type.getElementTypeBitWidth();
  if (dim == 1)
    return true;
  if (succeeded(successStrides) && !strides.empty() &&
      !type.isDynamicDim(dim - 1))
    return (type.getDimSize(dim - 1) == strides[dim - 2]) ||
           (strides[dim - 2] * dataWidth % hwAlignBits == 0);
  return false;
}

// Return "true" if the dim of the given type greater than 2 and the stride of
// sub-low dimension is UB align.
// Such as memref<3x13x15xi8>
static bool isSubLowDimStrideAlign(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  int64_t dim = type.getRank();
  int64_t hwAlignBits = static_cast<int64_t>(getHWAlignBytes(type.getMemorySpace()) * 8);
  int64_t dataWidth = type.getElementTypeBitWidth();
  if (dim <= 2)
    return true;
  if (succeeded(successStrides) && !strides.empty())
    return strides[dim - 3] * dataWidth % hwAlignBits == 0;
  return false;
}

/// get last discontinuous dim
std::optional<int> getLastDiscontinuousDim(
    const SmallVectorImpl<MemRefType> &memRefTypes,
    const SmallVectorImpl<MemRefType> &origMemRefTypes,
    const SmallVector<ReassociationIndices> &continuousReassociations,
    bool isUBDMAOp, bool archIsRegbased, bool archIs950) {
  LLVM_DEBUG(llvm::dbgs() << "memRefTypes " << memRefTypes << "\n";
             llvm::dbgs() << "origMemRefTypes " << origMemRefTypes << "\n";
             utils::dumpReassociationIndicesVector(continuousReassociations););
  assert(!continuousReassociations.empty());
  if (llvm::any_of(memRefTypes, [&](MemRefType memRefType) {
        return !isLastMemrefDimUnitStride(memRefType);
      })) {
    LDBG("last dim stride is not 1\n");
    return getLastNotUnitDim(origMemRefTypes, continuousReassociations.back());
  }

  if (archIs950 && isUBDMAOp) {
    // If the op is copy_gm_to_ub or copy_ub_to_gm or copy_ub_to_ub,
    // only consider the stride of last dim.
    if (llvm::any_of(origMemRefTypes, [&](MemRefType memRefType) {
          auto hivmSpace =
              dyn_cast<hivm::AddressSpaceAttr>(memRefType.getMemorySpace());
          if (hivmSpace.getAddressSpace() == hivm::AddressSpace::UB)
            return isSubLowDimStrideAlign(memRefType) &&
                   isNotTailJumpOrStrideAlign(memRefType);
          else
            return false;
        })) {
      return std::nullopt;
    }
  }

  if (continuousReassociations.size() == 1) {
    if (!archIsRegbased || continuousReassociations[0].size() == 1) {
      LDBG("last un-continuous dim does not exist");
      return std::nullopt;
    }
    // At this step, we can not do flatten for VF(flatten should be done
    // before auto-vectorize pass).
    return continuousReassociations.back().back();
  }

  assert(continuousReassociations.size() > 1);
  if (!archIsRegbased)
    return getLastNotUnitDim(
        origMemRefTypes,
        continuousReassociations[continuousReassociations.size() - 2]);

  return getLastNotUnitDim(
      origMemRefTypes,
      continuousReassociations[continuousReassociations.size() - 1]);
}

/// get last discontinuous dim
std::optional<int> getLastDiscontinuousDimRegBased(
    const SmallVectorImpl<MemRefType> &memRefTypes,
    const ReassociationIndices &continuousReassociations, bool isUBDMAOp) {
  LLVM_DEBUG(llvm::dbgs() << "memRefTypes " << memRefTypes << "\n";);
  // 1. if any memref contains non-unit stride at the last dim,
  //    the last axis of size > 1 should be aligned.
  if (llvm::any_of(memRefTypes, [&](MemRefType memRefType) {
        return !isLastMemrefDimUnitStride(memRefType);
      })) {
    LDBG("last dim stride is not 1\n");
    return getLastNotUnitDim(memRefTypes, continuousReassociations);
  }
  // Now the last dim's stride of all memrefs are unit-stride.
  // 2. short cut if template and ISA could handle these unaligned cases
  if (isUBDMAOp) {
    // If the op is copy_gm_to_ub or copy_ub_to_gm or copy_ub_to_ub,
    // only consider the stride of last dim.
    if (llvm::any_of(memRefTypes, [&](MemRefType memRefType) {
          auto hivmSpace =
              dyn_cast<hivm::AddressSpaceAttr>(memRefType.getMemorySpace());
          if (hivmSpace.getAddressSpace() == hivm::AddressSpace::UB)
            return isSubLowDimStrideAlign(memRefType) &&
                   isNotTailJumpOrStrideAlign(memRefType);
          else
            return false;
        })) {
      return std::nullopt;
    }
  }

  if (continuousReassociations.size() == 1) {
    LDBG("1D memref does not contain last non-contiguous axis");
    return std::nullopt;
  }
  // multi-dim memref
  // At this step, we can not do flatten for VF(flatten should be done
  // before auto-vectorize pass).
  return getLastNotUnitDim(memRefTypes, continuousReassociations);
}

void getDimAxisInfo(Operation *op, SmallVectorImpl<int64_t> &reshapeDims,
                    SmallVectorImpl<int64_t> &permutations) {
  if (auto brcOp = dyn_cast<hivm::VBrcOp>(op)) {
    auto brcDims = brcOp.getBroadcastDims();
    auto dstRank = cast<ShapedType>(brcOp.getDst().getType()).getRank();
    reshapeDims =
        brcDims.empty()
            ? SmallVector<int64_t>(llvm::to_vector(llvm::seq<int64_t>(dstRank)))
            : SmallVector<int64_t>(brcDims);
  } else if (auto reduceOp = dyn_cast<hivm::VReduceOp>(op)) {
    reshapeDims = SmallVector<int64_t>(reduceOp.getReduceDims());
  } else if (auto transOp = dyn_cast<hivm::VTransposeOp>(op)) {
    auto perms = transOp.getPermutation();
    permutations = SmallVector<int64_t>(perms);
  } else if (auto concatOp = dyn_cast<hivm::VConcatOp>(op)) {
    concatOp.getConcatLoopDims(reshapeDims);
  } else if (auto padOp = dyn_cast<hivm::VPadOp>(op)) {
    padOp.getPadLoopDims(reshapeDims);
  } else if (auto gatherOp = dyn_cast<hivm::VGatherOp>(op)) {
    gatherOp.getGatherLoopDims(reshapeDims);
  }
}

bool isAllRank0(const SmallVectorImpl<MemRefType> &memrefTypes) {
  bool isAllRank0 = llvm::all_of(memrefTypes, [](MemRefType mtype) {
    return mtype.hasRank() && mtype.getRank() == 0;
  });
  return isAllRank0;
}

static bool isAllOnesShape(const SmallVectorImpl<MemRefType> &types) {
  return llvm::all_of(types, [](MemRefType t) {
    if (!t.hasRank())
      return false;
    auto shape = t.getShape();
    return llvm::all_of(shape, [](int64_t d) { return d == 1; });
  });
}

bool isAnyOfLocalBuffer(const SmallVectorImpl<MemRefType> &memrefTypes) {
  bool anyOfLocalBuffer = llvm::any_of(memrefTypes, [](MemRefType mtype) {
    return isLocalBuffer(getHIVMAddressSpaceAttr(mtype));
  });
  return anyOfLocalBuffer;
}

// Whether the op is already marked stride_align_dims attr.
static bool isMarked(Value op) {
  auto users = op.getUsers();
  for (auto user : users) {
    if (utils::isAnnotationWithAttr(user, hivm::StrideAlignDimsAttr::name))
      return true;
  }

  return false;
}

// Filter memref without UB/GM target space
static SmallVector<MemRefType>
filterNonHivmSpace(const SmallVectorImpl<MemRefType> &memRefTypes) {
  SmallVector<MemRefType> resMemRefTypes;
  for (auto memRefType : memRefTypes) {
    auto hivmSpace =
        dyn_cast_or_null<hivm::AddressSpaceAttr>(memRefType.getMemorySpace());
    if (hivmSpace && (hivmSpace.getAddressSpace() == hivm::AddressSpace::UB)) {
      resMemRefTypes.push_back(memRefType);
    }
  }
  return resMemRefTypes;
}

// find the src memref arg used in transferwrite op
static int findArgNeedMark(vector::TransferWriteOp writeOp) {
  int res = -1;
  Value memrefVal = writeOp.getSource();
  Operation *defOp = memrefVal.getDefiningOp();
  while (defOp) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
      memrefVal = subviewOp.getSource();
      defOp = memrefVal.getDefiningOp();
    } else if (auto viewOp = dyn_cast<memref::ViewOp>(defOp)) {
      memrefVal = viewOp.getSource();
      defOp = memrefVal.getDefiningOp();
    } else {
      break;
    }
  }
  if (auto blockArg = dyn_cast<BlockArgument>(memrefVal)) {
    defOp = blockArg.getOwner()->getParentOp();
    if (isa<func::FuncOp>(defOp)) {
      res = (int)blockArg.getArgNumber();
    }
  }
  return res;
}

// calculate align size to avoid bank conflict
// Davinci recommend the vsstb stride align to (512*N + 32) Byte
// In isTransferWriteSuitForStoreWithStride, the memref last dim has been
// guaranteed always 32Byte.
// So the align size of sub tail dim is (16*N + 1)
static int32_t alignSizeForBankConflict(MemRefType memrefTy) {
  ArrayRef<int64_t> shape = memrefTy.getShape();
  int32_t alignDim = (int32_t)memrefTy.getRank() - 2;
  int32_t alignSize = shape[alignDim] * 16 / std::gcd(shape[alignDim], 16) + 1;
  // In enable-stride-align, the alignSize will be divided by type width.
  // So the marked align size will be multiplied by type width.
  int64_t dataWidth = memrefTy.getElementTypeBitWidth() / utils::kBitsToByte;
  return alignSize * dataWidth;
}

// if vf function has vsstb, mark the related input arg with a higher size
// to avoid memory access bank
static void markForVsstb(OpBuilder &builder, func::CallOp &callOp,
                         ModuleOp moduleOp) {
  StringRef callee = callOp.getCallee();
  if (auto func = moduleOp.lookupSymbol<func::FuncOp>(callee)) {
    func->walk([&builder, &callOp](vector::TransferWriteOp writeOp) {
      if (!utils::isTransferWriteSuitForStoreWithStride(writeOp))
        return WalkResult::skip();
      int argIdx = findArgNeedMark(writeOp);
      if (argIdx == -1)
        return WalkResult::skip();
      Value operand = callOp.getOperand(argIdx);
      if (isMarked(operand))
        return WalkResult::skip();
      auto memrefTy = dyn_cast<MemRefType>(operand.getType());
      if (!memrefTy)
        return WalkResult::skip();
      // TODO: Currently vsstb can make continuous shape into discontinous
      // shape, which may cause reshape & collaspse operation users that not
      // correct, because these operation relys on continous shape feature. So
      // this may cause FlattenOps afterwards facing 2 different shapes, which
      // would bring compilation errors. This conflict may be solved afterwards.
      for (auto use : operand.getUsers()) {
        if (llvm::isa_and_nonnull<memref::CollapseShapeOp>(use) ||
            llvm::isa_and_nonnull<memref::ReshapeOp>(use))
          return WalkResult::skip();
      }
      builder.setInsertionPoint(callOp);
      int32_t alignDim = (int32_t)memrefTy.getRank() - 2;
      createAlignMarkOp(builder, callOp->getLoc(), operand, {alignDim},
                        {alignSizeForBankConflict(memrefTy)});
      return WalkResult::advance();
    });
  }
}

void MarkStrideAlignPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  bool archIsRegbased = hacc::utils::isRegBasedArch(moduleOp);
  bool archIs950 = hacc::utils::isAscend950(moduleOp);
  WalkResult result = funcOp->walk([&builder, archIsRegbased,
                                    archIs950](Operation *op) {
    LDBG("Walk operation : " << *op);
    if (!isa<HIVMStructuredOp>(op)) {
      return WalkResult::advance();
    }

    // TODO: Relax when enable user provided optimization hints
    if (isa<CustomOp>(op) || isa<CustomMacroOp>(op)) {
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
    if (isa<hivm::VReduceOp>(op)) {
      auto vReduceOp = dyn_cast<hivm::VReduceOp>(op);
      hivm::ReduceOperation arithOp = vReduceOp.getArith().getReduceOp();
      if (utils::isReduceWithIndex(arithOp))
        return WalkResult::skip();
    }
    auto types = hivmOp.getHIVMOperandTypes(/*includeExtraBuffer=*/false);
    auto memrefTypes = util::getMemRefTypes(types);
    if (isAllRank0(memrefTypes) || isAllOnesShape(memrefTypes)) {
      return WalkResult::advance();
    }
    if (!isAnyOfLocalBuffer(memrefTypes)) {
      return WalkResult::advance();
    }

    bool isUBDMAOp = isa<hivm::LoadOp>(op) || isa<hivm::StoreOp>(op) ||
                     isa<hivm::CopyOp>(op);
    std::optional<int> alignDim;
    if (archIsRegbased) {
      // Filter memrefs not in ubspace
      auto filterMemrefTypes = filterNonHivmSpace(memrefTypes);
      if (filterMemrefTypes.empty())
        return WalkResult::skip();
      // In A5, the previous hfusion-flatten pass merges axes for both tensors
      // and memrefs, so it no longer calls the getFlattened method of the Op
      // for analysis. Instead, it directly treats the Op as already having
      // undergone the axis merging operation and proceeds with analysis.
      ReassociationIndices identityReassoc(filterMemrefTypes[0].getRank());
      for (auto i = 0; i < filterMemrefTypes[0].getRank(); ++i) {
        identityReassoc[i] = i;
      }
      // In A5, memrefTypes is already the result of flattening.
      alignDim = getLastDiscontinuousDimRegBased(filterMemrefTypes,
                                                 identityReassoc, isUBDMAOp);
    } else {
      // For A2/3
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
      alignDim = getLastDiscontinuousDim(flattenedMemrefTypes, memrefTypes,
                                         flattenedAssociations, isUBDMAOp,
                                         archIsRegbased, archIs950);
    }
    LDBG("getLastDiscontinuousDim " << alignDim.value_or(-1) << "\n");
    for (const auto &oper : hivmOp.getTargetSpaceOperands(
             hivm::AddressSpace::UB, false /*includeTmpBuffer*/)) {
      auto adjustedAlignDim = adjustAlignDim(op, oper, alignDim);
      LDBG("adjustedAlignDim " << adjustedAlignDim.value_or(-1) << "\n");
      if (failed(markAlignedDim(builder, op, oper, adjustedAlignDim)))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (archIsRegbased) {
    WalkResult resultVF = funcOp->walk([&builder, &moduleOp, archIsRegbased,
                                        archIs950](func::CallOp callOp) {
      for (Value operand : callOp.getArgOperands()) {
        if (isMarked(operand))
          continue;
        auto type = operand.getType();
        auto memrefTypes = util::getMemRefTypes({type});
        if (isAllRank0(memrefTypes))
          continue;
        if (!isAnyOfLocalBuffer(memrefTypes))
          continue;
        if (archIs950) {
          if (llvm::any_of(memrefTypes, [](MemRefType mtype) {
                auto elemWidth = mtype.getElementType().getIntOrFloatBitWidth();
                return elemWidth != 1;
              }))
            continue;
        }

        SmallVector<int64_t> reshapeDims{};
        SmallVector<int64_t> permutations{};
        getDimAxisInfo(callOp, reshapeDims, permutations);
        auto continuousAssociations = util::getContinuousReassociation(
            memrefTypes, ArrayRef<int64_t>(reshapeDims),
            ArrayRef<int64_t>(permutations));
        auto alignDim = getLastDiscontinuousDim(memrefTypes, memrefTypes,
                                                continuousAssociations, false,
                                                archIsRegbased, archIs950);
        LDBG("getLastDiscontinuousDim " << alignDim.value_or(-1) << "\n");

        auto adjustedAlignDim = adjustAlignDim(callOp, operand, alignDim);
        LDBG("adjustedAlignDim " << adjustedAlignDim.value_or(-1) << "\n");
        if (failed(markAlignedDim(builder, callOp, operand, adjustedAlignDim)))
          return WalkResult::interrupt();
      }
      markForVsstb(builder, callOp, moduleOp);
      return WalkResult::advance();
    });
    if (resultVF.wasInterrupted()) {
      return signalPassFailure();
    }
  }
  if (result.wasInterrupted()) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createMarkStrideAlignPass() {
  return std::make_unique<MarkStrideAlignPass>();
}
