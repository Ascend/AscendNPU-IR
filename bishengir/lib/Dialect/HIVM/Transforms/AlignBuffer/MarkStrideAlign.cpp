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
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

struct TrailingUnitRankReducedSubviewInfo {
  Value source;
  MemRefType sourceType;
  MemRefType resultType;
  llvm::SmallBitVector droppedDims;
};

struct AlignMarkDecision {
  Value markTarget;
  std::optional<int> alignDim;
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
    auto hwAlignBytes = util::getHWAlignBytes(memrefTy.getMemorySpace());
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
// Return "true", if the size of lowest dimension equal with the stride of
// sub-low dimension or the stride of sub-low dimension is UB align.
// Such as memref<3x13xi8, strided<[16, 1]>>
static bool isNotTailJumpOrStrideAlign(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  int64_t dim = type.getRank();
  int64_t hwAlignBits =
      static_cast<int64_t>(getHWAlignBytes(type.getMemorySpace()) * 8);
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
  int64_t hwAlignBits =
      static_cast<int64_t>(getHWAlignBytes(type.getMemorySpace()) * 8);
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

static bool noNeedAlign(Operation *op) {
  // Check if at least one operand is non-GM memscope
  bool hasNonGMMemref = false;
  for (Value operand : op->getOperands()) {
    if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
      auto memSpace = memrefType.getMemorySpace();
      assert(memSpace && "memSpace must not be null");
      if (auto spaceAttr = dyn_cast<AddressSpaceAttr>(memSpace)) {
        if (spaceAttr.getAddressSpace() != hivm::AddressSpace::GM) {
          hasNonGMMemref = true;
          break;
        }
      }
    }
  }

  return !hasNonGMMemref;
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

static std::optional<TrailingUnitRankReducedSubviewInfo>
getTrailingUnitRankReducedSubviewInfo(Value operand) {
  auto subviewOp = operand.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    LDBG("subview-recover: operand is not defined by memref.subview, operand="
         << operand);
    return std::nullopt;
  }

  auto sourceType = dyn_cast<MemRefType>(subviewOp.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(subviewOp.getResult().getType());
  if (!sourceType || !resultType || !sourceType.hasRank() ||
      !resultType.hasRank() || sourceType.getRank() <= resultType.getRank()) {
    LDBG("subview-recover: rank-reduction precheck failed, sourceType="
         << sourceType << ", resultType=" << resultType
         << ", subview=" << subviewOp);
    return std::nullopt;
  }

  auto droppedDims = subviewOp.getDroppedDims();
  if (droppedDims.none()) {
    LDBG("subview-recover: no dropped dims, subview=" << subviewOp);
    return std::nullopt;
  }

  int64_t firstDroppedDim = droppedDims.find_first();
  if (firstDroppedDim < 0) {
    LDBG("subview-recover: failed to locate first dropped dim, subview="
         << subviewOp);
    return std::nullopt;
  }

  for (int64_t dim = firstDroppedDim; dim < sourceType.getRank(); ++dim) {
    if (!droppedDims.test(dim)) {
      LDBG("subview-recover: dropped dims are not trailing, dim="
           << dim << ", subview=" << subviewOp);
      return std::nullopt;
    }
  }

  auto staticSizes = subviewOp.getStaticSizes();
  for (int64_t dim = firstDroppedDim; dim < sourceType.getRank(); ++dim) {
    if (staticSizes[dim] != 1) {
      LDBG("subview-recover: dropped dim slice size is not static-1, dim="
           << dim << ", staticSize=" << staticSizes[dim]
           << ", sourceType=" << sourceType << ", subview=" << subviewOp);
      return std::nullopt;
    }
  }

  std::string droppedDimsStr;
  llvm::raw_string_ostream droppedDimsOs(droppedDimsStr);
  droppedDimsOs << '[';
  bool first = true;
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    if (!droppedDims.test(dim))
      continue;
    if (!first)
      droppedDimsOs << ", ";
    droppedDimsOs << dim;
    first = false;
  }
  droppedDimsOs << ']';
  droppedDimsOs.flush();
  LDBG("subview-recover: recognized trailing-unit rank-reduced subview, "
       << "sourceType=" << sourceType << ", resultType=" << resultType
       << ", droppedDims=" << droppedDimsStr);
  return TrailingUnitRankReducedSubviewInfo{subviewOp.getSource(), sourceType,
                                            resultType, droppedDims};
}

static std::optional<int>
getLastNotUnitDimForMemRefType(MemRefType memRefType) {
  if (!memRefType.hasRank())
    return std::nullopt;

  ReassociationIndices identityReassoc(memRefType.getRank());
  for (int64_t i = 0; i < memRefType.getRank(); ++i)
    identityReassoc[i] = i;
  SmallVector<MemRefType> memRefTypes{memRefType};
  return getLastNotUnitDim(memRefTypes, identityReassoc);
}

static int
getPostAlignDimAfterDropLocal(int prevDim,
                              const llvm::SmallBitVector &droppedDims) {
  int postDim = -1;
  int curDim = droppedDims.find_first_unset();
  while (curDim <= prevDim && curDim != -1) {
    postDim++;
    curDim = droppedDims.find_next_unset(curDim);
  }
  return postDim;
}

static AlignMarkDecision fixAlignDimForSingleUBStoreSubview(
    Value operand, std::optional<int> alignDim,
    const TrailingUnitRankReducedSubviewInfo &subviewInfo) {
  auto expectedSourceAlignDim =
      getLastNotUnitDimForMemRefType(subviewInfo.sourceType);
  LDBG("subview-fix: sourceType="
       << subviewInfo.sourceType << ", resultType=" << subviewInfo.resultType
       << ", originalAlignDim=" << alignDim.value_or(-1)
       << ", expectedSourceAlignDim=" << expectedSourceAlignDim.value_or(-1));

  if (!expectedSourceAlignDim.has_value())
    return {operand, alignDim};

  // If the true source-side align axis is one of the dropped dims, the current
  // rank-reduced operand cannot represent that axis anymore. Mark the source
  // directly so propagation sees the correct source semantic axis.
  if (subviewInfo.droppedDims.test(expectedSourceAlignDim.value())) {
    LDBG("subview-fix: expected source align dim "
         << expectedSourceAlignDim.value()
         << " is dropped in the rank-reduced result, mark source directly");
    return {subviewInfo.source, expectedSourceAlignDim};
  }

  int mappedResultAlignDim = getPostAlignDimAfterDropLocal(
      expectedSourceAlignDim.value(), subviewInfo.droppedDims);
  if (mappedResultAlignDim < 0) {
    LDBG("subview-fix: failed to map source align dim "
         << expectedSourceAlignDim.value()
         << " back to result rank, mark source directly");
    return {subviewInfo.source, expectedSourceAlignDim};
  }

  LDBG("subview-fix: mapped source align dim "
       << expectedSourceAlignDim.value() << " to result align dim "
       << mappedResultAlignDim << ", keep marking current operand");
  return {operand, mappedResultAlignDim};
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

// Davinci recommends the vsstb stride align to (512*N + 32) Byte.
// In isTransferWriteSuitForStoreWithStride, the memref last dim has been
// guaranteed always 32 Byte (256 bit / element bit width).
// So the alignment of the sub-tail dim is derived as (16*N + 1) elements,
// where 16 = 512 Byte / 32 Byte (the ratio of alignment to last-dim size).
static constexpr int32_t kVsstbBankConflictAlignUnit = 16;
static constexpr int32_t kVsstbSubTailDimOffset = 2;

// calculate align size to avoid bank conflict
static int32_t alignSizeForBankConflict(MemRefType memrefTy) {
  ArrayRef<int64_t> shape = memrefTy.getShape();
  int32_t alignDim =
      static_cast<int32_t>(memrefTy.getRank()) - kVsstbSubTailDimOffset;
  int32_t alignSize = shape[alignDim] * kVsstbBankConflictAlignUnit /
                          std::gcd(shape[alignDim],
                                   kVsstbBankConflictAlignUnit) +
                      1;
  // In enable-stride-align, the alignSize will be divided by type width.
  // So the marked align size will be multiplied by type width.
  int64_t dataWidth = memrefTy.getElementTypeBitWidth() / utils::kBitsToByte;
  return alignSize * static_cast<int32_t>(dataWidth);
}

// Check if the operand's root alloc has a SkipStrideAlignForVLoad
// annotation (placed by PreMarkStrideAlign).
static bool shouldSkipStrideAlignForVLoad(Value oper) {
  Value root = traceToRoot(oper);
  return utils::getAnnotateOpWithAttr(
      root, hivm::SkipStrideAlignForVLoadAttr::name).has_value();
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
      static constexpr int kNotFoundArgIdx = -1;
      if (argIdx == kNotFoundArgIdx)
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
            llvm::isa_and_nonnull<memref::ReshapeOp>(use)) {
          return WalkResult::skip();
        }
      }
      builder.setInsertionPoint(callOp);
      int32_t alignDim =
          static_cast<int32_t>(memrefTy.getRank()) - kVsstbSubTailDimOffset;
      createAlignMarkOp(builder, callOp->getLoc(), operand, {alignDim},
                        {alignSizeForBankConflict(memrefTy)});
      return WalkResult::advance();
    });
  }
}

LogicalResult markAlignedDimForFixcctoub(OpBuilder &builder,
                                         Operation *markedOp, Value arg,
                                         std::optional<int> alignedDim,
                                         int32_t alignByte) {
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
    createAlignMarkOp(builder, markedOp->getLoc(), arg, {alignedDimValue},
                      {alignByte});
  }
  return success();
}

enum class DataWidthType { B4, B8, B16, B32 };
enum class ChannelSplit { CS_N, CS_Y };
enum class NZ2ND { ND_N, ND_Y };
enum class NZ2DN { DN_N, DN_Y };
enum class LoopEnhance { LE_N, LE_Y };
enum class DualDstMode { N, SplitN, SplitM };
struct TupleHashForN {
  template <typename... Args>
  size_t operator()(const std::tuple<Args...> &t) const {
    return std::hash<size_t>{}((static_cast<size_t>(std::get<0>(t)) << 24) |
                               (static_cast<size_t>(std::get<1>(t)) << 16) |
                               (static_cast<size_t>(std::get<2>(t)) << 8) |
                               (static_cast<size_t>(std::get<3>(t)) << 4) |
                               static_cast<size_t>(std::get<4>(t)));
  }
};
struct TupleHashForM {
  template <typename... Args>
  size_t operator()(const std::tuple<Args...> &t) const {
    return std::hash<size_t>{}((static_cast<size_t>(std::get<0>(t)) << 8) |
                               (static_cast<size_t>(std::get<1>(t)) << 4) |
                               static_cast<size_t>(std::get<2>(t)));
  }
};
static int64_t getMsizeAlignmentRequirement(DataWidthType dtype, NZ2DN nz,
                                            DualDstMode ddm) {
  using Key = std::tuple<DataWidthType, NZ2DN, DualDstMode>;
  static const std::unordered_map<Key, int, TupleHashForM> constraints = {
      {{DataWidthType::B16, NZ2DN::DN_Y, DualDstMode::N}, 16},
      {{DataWidthType::B8, NZ2DN::DN_Y, DualDstMode::N}, 32},
      {{DataWidthType::B4, NZ2DN::DN_Y, DualDstMode::N}, 64},
      {{DataWidthType::B32, NZ2DN::DN_N, DualDstMode::SplitM}, 2},
      {{DataWidthType::B32, NZ2DN::DN_Y, DualDstMode::N}, 8},
  };
  Key query_key{dtype, nz, ddm};
  if (auto it = constraints.find(query_key); it != constraints.end()) {
    return it->second;
  }
  return -1;
}

static int64_t getNsizeAlignmentRequirement(DataWidthType dtype,
                                            ChannelSplit cs, NZ2ND nz,
                                            LoopEnhance le, DualDstMode ddm) {
  using Key =
      std::tuple<DataWidthType, ChannelSplit, NZ2ND, LoopEnhance, DualDstMode>;
  static const std::unordered_map<Key, int, TupleHashForN> constraints = {
      {{DataWidthType::B16, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::N},
       16},
      {{DataWidthType::B16, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_Y,
        DualDstMode::N},
       16},
      {{DataWidthType::B16, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_N,
        DualDstMode::N},
       16},
      {{DataWidthType::B16, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_Y,
        DualDstMode::N},
       16},
      {{DataWidthType::B8, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::N},
       16},
      {{DataWidthType::B8, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_Y,
        DualDstMode::N},
       16},
      {{DataWidthType::B8, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_N,
        DualDstMode::N},
       32},
      {{DataWidthType::B8, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_Y,
        DualDstMode::N},
       16},
      {{DataWidthType::B4, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::N},
       16},
      {{DataWidthType::B4, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_N,
        DualDstMode::N},
       64},
      {{DataWidthType::B32, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::N},
       16},
      {{DataWidthType::B32, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_N,
        DualDstMode::N},
       8},
      {{DataWidthType::B32, ChannelSplit::CS_Y, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::N},
       8},
      {{DataWidthType::B32, ChannelSplit::CS_N, NZ2ND::ND_N, LoopEnhance::LE_N,
        DualDstMode::SplitN},
       32},
      {{DataWidthType::B32, ChannelSplit::CS_N, NZ2ND::ND_Y, LoopEnhance::LE_N,
        DualDstMode::SplitN},
       32}};

  Key query_key{dtype, cs, nz, le, ddm};
  if (auto it = constraints.find(query_key); it != constraints.end()) {
    return it->second;
  }
  return -1;
}

std::optional<int>
getLastMUnitDim(const SmallVectorImpl<MemRefType> &memRefTypes,
                ReassociationIndices reassociations) {
  int foundCount = 0;
  int lastFoundIndex = -1;
  for (auto index : llvm::reverse(reassociations)) {
    foundCount++;
    lastFoundIndex = index;
    if (foundCount == 2) {
      return lastFoundIndex;
    }
  }
  // If only one non-unit dim found, return it
  if (foundCount == 1) {
    return lastFoundIndex;
  }
  return std::nullopt;
}

std::optional<int>
getLastNUnitDim(const SmallVectorImpl<MemRefType> &memRefTypes,
                  ReassociationIndices reassociations) {
  for (auto index : llvm::reverse(reassociations)) {
    return index;
  }
  return std::nullopt;
}

/// get last discontinuous dim
std::vector<std::pair<std::optional<int>, int32_t>> getLastDiscontinuousDimRegBasedForFixcctoub(
    const SmallVectorImpl<MemRefType> &memRefTypes,
    const ReassociationIndices &continuousReassociations, Operation *op) {
  Value input = op->getOperand(0);
  Value output = op->getOperand(1);
  MemRefType inputType = mlir::cast<MemRefType>(input.getType());
  MemRefType outputType = mlir::cast<MemRefType>(output.getType());
  auto inputSpace = inputType.getMemorySpace();
  auto outputSpace = outputType.getMemorySpace();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputShape.size() != 2) {
    return {};
  }
  auto hivminputSpace = mlir::cast<hivm::AddressSpaceAttr>(inputSpace);
  auto hivmoutputSpace = mlir::cast<hivm::AddressSpaceAttr>(outputSpace);
  if (!hivminputSpace || !hivmoutputSpace) {
    return {};
  }
  if (hivminputSpace.getAddressSpace() != hivm::AddressSpace::L0C ||
      hivmoutputSpace.getAddressSpace() != hivm::AddressSpace::UB) {
    return {};
  }
  auto fixpipeOp = dyn_cast<FixpipeOp>(op);
  auto dmaModeAttr = fixpipeOp.getDmaModeAttr();
  auto chsModeAttr = fixpipeOp.getChannelSplitAttr();
  int64_t dataWidth = outputType.getElementTypeBitWidth();
  DataWidthType dtype = DataWidthType::B32;
  if (dataWidth == 16) {
    dtype = DataWidthType::B16;
  } else if (dataWidth == 8) {
    dtype = DataWidthType::B8;
  } else if (dataWidth == 4) {
    dtype = DataWidthType::B4;
  }
  ChannelSplit csn = ChannelSplit::CS_N;
  if (chsModeAttr && chsModeAttr.getValue()) {
    csn = ChannelSplit::CS_Y;
  }
  NZ2ND nz2nd = NZ2ND::ND_N;
  if (dmaModeAttr && dmaModeAttr.getValue() == FixpipeDMAMode::NZ2ND) {
    nz2nd = NZ2ND::ND_Y;
  }
  LoopEnhance le = LoopEnhance::LE_N;
  DualDstMode ddm = DualDstMode::N;
  NZ2DN nz2dn = NZ2DN::DN_N;
  if (dmaModeAttr && dmaModeAttr.getValue() == FixpipeDMAMode::NZ2DN) {
    nz2dn = NZ2DN::DN_Y;
  }

  Value dst = fixpipeOp.getDst();
  FixpipeDualDstModeAttr ddmAttr = fixpipeOp.getDualDstModeAttr();
  if (!ddmAttr) {
    auto maybeAllocOp = traceDefOp<memref::AllocOp>(dst);
    if (maybeAllocOp) {
      memref::AllocOp allocOp = cast<memref::AllocOp>(*maybeAllocOp);
      mlir::Value allocVal = allocOp.getResult();
      auto maybeMarkOpRaw = utils::getAnnotateOpWithAttr(
          allocVal, hivm::HIVMTightlyCoupledBufferAttr::name.str());
      if (maybeMarkOpRaw) {
        auto markOp = dyn_cast<annotation::MarkOp>(*maybeMarkOpRaw);
        if (markOp) {
          auto tilingDimAttr =
              markOp->getAttrOfType<IntegerAttr>("hivm.tiling_dim");
          if (tilingDimAttr) {
            int64_t tilingDim = tilingDimAttr.getValue().getSExtValue();
            auto rank = allocOp.getType().getRank();
            if (!(tilingDim == -1 ||
                  (tilingDim != rank - 2 && tilingDim != rank - 1))) {
              ddm = tilingDim == rank - 2 ? DualDstMode::SplitM
                                          : DualDstMode::SplitN;
            }
          }
        }
      }
    }
  } else {
    if (ddmAttr.getDualDstMode() == FixpipeDualDstMode::ROW_SPLIT) {
      ddm = DualDstMode::SplitM;
    } else if (ddmAttr.getDualDstMode() == FixpipeDualDstMode::COLUMN_SPLIT) {
      ddm = DualDstMode::SplitN;
    }
  }
  std::vector<std::pair<std::optional<int>, int32_t>> alignDimAlignBytes;
  int32_t alignBytes = 32;
  DualDstMode mDdm = ddm == DualDstMode::SplitN ? ddm : DualDstMode::N;
  int64_t nSizeStride =
      getNsizeAlignmentRequirement(dtype, csn, nz2nd, le, mDdm);
  if (nSizeStride != -1) {
    alignBytes = nSizeStride * dataWidth / 8;
    std::optional<int> alignDim =
        getLastNUnitDim(memRefTypes, continuousReassociations);
    alignDimAlignBytes.emplace_back(alignDim, alignBytes);
  }
  DualDstMode nDdm = ddm == DualDstMode::SplitM ? ddm : DualDstMode::N;
  int64_t mSizeStride = getMsizeAlignmentRequirement(dtype, nz2dn, nDdm);
  if (mSizeStride != -1) {
    alignBytes = mSizeStride * dataWidth / 8;
    std::optional<int> alignDim =
        getLastMUnitDim(memRefTypes, continuousReassociations);
    alignDimAlignBytes.emplace_back(alignDim, alignBytes);
  }
  return alignDimAlignBytes;
}

void MarkStrideAlignPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  WalkResult result = funcOp->walk([&builder](Operation *op) {
    LDBG("Walk operation : " << *op);
    if (!isa<HIVMStructuredOp>(op) || noNeedAlign(op)) {
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
    if (isa<CustomOp>(op) || isa<CustomMacroOp>(op) ||
        isa<hivm::MmadMxL1Op>(op)) {
      return WalkResult::advance();
    }

    auto hivmOp = cast<HIVMStructuredOp>(op);
    if (!hivmOp.hasPureBufferSemantics()) {
      hivmOp->emitError("Not bufferized.");
      return WalkResult::interrupt();
    }

    if (isa<CustomOp, CustomMacroOp>(op)) {
      ArrayAttr argAttrs =
          op->getAttrOfType<ArrayAttr>(CustomOp::kArgAttrsName);
      if (!argAttrs)
        return WalkResult::advance();

      for (const auto &[idx, dictAttrs] : llvm::enumerate(argAttrs)) {
        auto dict = dyn_cast_or_null<DictionaryAttr>(dictAttrs);
        if (!dict)
          continue;

        auto intAttr =
            dyn_cast_or_null<IntegerAttr>(dict.get(CustomOp::kAlignDimName));
        if (!intAttr)
          continue;

        const int alignDim = intAttr.getInt();
        if (failed(markAlignedDim(builder, op, op->getOperand(idx), alignDim)))
          return WalkResult::interrupt();
      }

      return WalkResult::advance();
    }

    if (isa<hivm::VTransposeOp>(op) &&
        cast<hivm::VTransposeOp>(op).isLastDimTranspose()) {
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
    if (isAllRank0(memrefTypes))
      return WalkResult::advance();
    if (!isAnyOfLocalBuffer(memrefTypes))
      return WalkResult::advance();

    bool isUBDMAOp = isa<hivm::LoadOp>(op) || isa<hivm::StoreOp>(op) ||
                     isa<hivm::CopyOp>(op);
    std::optional<int> alignDim;
    if (archIsRegbased) {
      // TODO: refactor is needed — it should not depend on specific op
      // Fixpipe may write through a subview/collapse that produces an all-ones
      // memref (e.g. <1x1>), while the underlying alloc has non-unit dims
      // (e.g. <3x1x1>). Skip isAllOnesShape for fixpipe so the 3d-dot stride
      // alignment can still reach the root alloc.
      if (!isa<FixpipeOp>(op) && isAllOnesShape(memrefTypes)) {
        return WalkResult::advance();
      }
      // Filter memrefs not in ubspace
      auto filterMemrefTypes = filterNonHivmSpace(memrefTypes);
      LDBG("mark-stride-align: reg-based path, filterMemrefTypes="
           << filterMemrefTypes);
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
      if (isa<FixpipeOp>(op)) {
        auto fixpipeOp = dyn_cast<FixpipeOp>(op);
        auto alignDimAlignBytes = getLastDiscontinuousDimRegBasedForFixcctoub(
            filterMemrefTypes, identityReassoc, fixpipeOp);
        for (const auto &[alignDimUpdate, alignByte] : alignDimAlignBytes) {
          if (alignDimUpdate.has_value()) {
            for (const auto &oper : hivmOp.getTargetSpaceOperands(
                     hivm::AddressSpace::UB, false /*includeTmpBuffer*/)) {
              if (failed(markAlignedDimForFixcctoub(builder, op, oper,
                                                    alignDimUpdate, alignByte)))
                return WalkResult::interrupt();
            }
          }
        }

        // For 3d dot we need align zero dimension because we tile over it
        if (auto rootAlloc =
                traceDefOp<memref::AllocOp>(hivmOp.getTargetSpaceOperands(
                    hivm::AddressSpace::UB, false /*includeTmpBuffer*/)[0])) {
          auto allocOp = cast<memref::AllocOp>(rootAlloc.value());
          auto allocType = allocOp.getType();
          auto allocShape = allocType.getShape();
          auto allocVal = allocOp.getResult();

          if (allocShape.size() == 3 && allocShape[0] > 1) {
            auto [alignDims, alignBytes] = adjustAlignInfo(
                allocOp, allocVal, {1}, getHWAlignBytes(allocType).value());
            if (alignDims.empty())
              return WalkResult::advance();

            if (failed(markAlignedDimForFixcctoub(builder, allocOp, allocVal,
                                                  alignDims[0], alignBytes[0])))
              return WalkResult::interrupt();
          }
        }

        return WalkResult::advance();
      }
    } else {
      // For A2/3
      if (isAllOnesShape(memrefTypes)) {
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
      LDBG("mark-stride-align: non-reg-based path, flattenedMemrefTypes="
           << flattenedMemrefTypes);
      alignDim = getLastDiscontinuousDim(flattenedMemrefTypes, memrefTypes,
                                         flattenedAssociations, isUBDMAOp,
                                         archIsRegbased, archIs950);
    }
    LDBG("getLastDiscontinuousDim " << alignDim.value_or(-1) << "\n");
    auto ubOperands = hivmOp.getTargetSpaceOperands(hivm::AddressSpace::UB,
                                                    false /*includeTmpBuffer*/);
    LDBG("mark-stride-align: UB operands count=" << ubOperands.size());
    Value singleUbMarkTarget;
    std::optional<int> singleUbAlignDim = alignDim;
    bool useSingleUbMarkDecision = false;
    if (isa<hivm::StoreOp>(op) && ubOperands.size() == 1) {
      LDBG("mark-stride-align: single-UB store candidate, operand="
           << ubOperands.front());
      if (auto subviewInfo =
              getTrailingUnitRankReducedSubviewInfo(ubOperands.front())) {
        LDBG("mark-stride-align: before subview fix alignDim="
             << alignDim.value_or(-1));
        auto decision = fixAlignDimForSingleUBStoreSubview(
            ubOperands.front(), alignDim, *subviewInfo);
        singleUbMarkTarget = decision.markTarget;
        singleUbAlignDim = decision.alignDim;
        useSingleUbMarkDecision = true;
        LDBG("mark-stride-align: after subview fix alignDim="
             << singleUbAlignDim.value_or(-1)
             << ", markTarget=" << singleUbMarkTarget);
      } else {
        LDBG("mark-stride-align: single-UB store did not match trailing-unit "
             "rank-reduced subview pattern");
      }
    } else if (isa<hivm::StoreOp>(op)) {
      LDBG("mark-stride-align: store skipped subview fix because UB operand "
           "count="
           << ubOperands.size());
    }

    for (const auto &oper : ubOperands) {
      Value markTarget = useSingleUbMarkDecision ? singleUbMarkTarget : oper;
      // Skip stride-alignment for allocs pre-marked by PreMarkStrideAlign
      // (DMA-loaded buffers that will be vload'd in a vf).
      if (alignDim.has_value() && isa<hivm::LoadOp>(op) &&
          shouldSkipStrideAlignForVLoad(markTarget)) {
        continue;
      }
      auto adjustedAlignDim =
          adjustAlignDim(op, markTarget,
                         useSingleUbMarkDecision ? singleUbAlignDim : alignDim);
      LDBG("adjustedAlignDim " << adjustedAlignDim.value_or(-1) << "\n");
      if (failed(markAlignedDim(builder, op, markTarget, adjustedAlignDim)))
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
              })) {
            continue;
          }
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

        // Skip stride-alignment for allocs pre-marked by PreMarkStrideAlign
        // (DMA-loaded buffers that will be vload'd in a vf).
        if (alignDim.has_value() &&
            shouldSkipStrideAlignForVLoad(operand)) {
          continue;
        }

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
