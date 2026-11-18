//===- BroadcastLoadPattern.cpp - fused broadcast load ----------===//
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
//
// Fuse a proven GM copy -> to_tensor -> rank-2/3 broadcast. Unknown
// provenance and aliases retain the existing vector-broadcast path.
//
//===----------------------------------------------------------------------===//

#include "FuseOpIntoLoadPatterns.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpecUtils.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define DEBUG_TYPE "hivm-fuse-op-into-load"

using namespace mlir;
using namespace mlir::hivm;

namespace {

static constexpr llvm::StringLiteral kTensorKindAttrName = "tt.tensor_kind";

static bool isSupportedElementType(Type t) {
  if (!isa<IntegerType, FloatType>(t))
    return false;
  unsigned width = t.getIntOrFloatBitWidth();
  // Copies may carry types that a verified LoadOp cannot. Reuse its ODS
  // constraint before creating any IR; width alone does not prove legality.
  return (width == 8 || width == 16 || width == 32) &&
         hivm::LoadOp::isSupportedElementType(t);
}

static bool isGMSource(Value source, func::FuncOp func,
                          bool explicitLoad = false) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || !func || func.isExternal())
    return false;
  // Retain the original source/view and read position. Do not manufacture a
  // source from an unknown call, local allocation, or nested block argument.
  Value root = source;
  while (Operation *def = root.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::ExpandShapeOp, memref::CollapseShapeOp>(def))
      return false;
    root = cast<ViewLikeOpInterface>(def).getViewSource();
  }
  auto arg = dyn_cast<BlockArgument>(root);
  if (!arg || arg.getOwner() != &func.getBody().front())
    return false;
  // A classified value must be GM. This pass runs before memory-space
  // inference, so a Triton kernel argument is still unclassified; the frontend
  // marks every tensor argument with `tt.tensor_kind` (input/output/inout),
  // and those pointers are GM. Non-tensor arguments (sync locks, workspace,
  // scalars) carry no such mark and stay rejected.
  auto rootType = dyn_cast<MemRefType>(arg.getType());
  Attribute spaces[2] = {type.getMemorySpace(),
                         rootType ? rootType.getMemorySpace() : Attribute()};
  for (Attribute space : spaces) {
    if (!space)
      continue;
    auto addressSpace = dyn_cast<hivm::AddressSpaceAttr>(space);
    if (!addressSpace ||
        addressSpace.getAddressSpace() != hivm::AddressSpace::GM)
      return false;
  }
  if (type.getMemorySpace() || (rootType && rootType.getMemorySpace()))
    return true;
  return explicitLoad ||
         func.getArgAttr(arg.getArgNumber(), kTensorKindAttrName) != nullptr;
}

// Match the span-load layout contract without weakening source-layout proof.
static bool hasProvenLoadSpan(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!isSupportedElementType(type.getElementType()) || type.getRank() == 0 ||
      failed(mlir::utils::getStridesAndOffsetCompat(type, strides, offset)) ||
      strides.back() != 1 || (!ShapedType::isDynamic(offset) && offset < 0))
    return false;
  std::optional<int64_t> span = 1;
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    int64_t size = type.getDimSize(axis), stride = strides[axis];
    if ((!ShapedType::isDynamic(stride) && stride < 0) ||
        (span && !ShapedType::isDynamic(stride) &&
         !ShapedType::isDynamic(size) && axis < type.getRank() - 1 &&
         stride < *span))
      return false;
    if (!span || ShapedType::isDynamic(size) || ShapedType::isDynamic(stride)) {
      span.reset();
    } else if (size == 0) {
      span = 0;
    } else if (size > 1) {
      if (stride > (INT64_MAX - *span) / (size - 1))
        return false;
      *span += (size - 1) * stride;
    }
  }
  int64_t bytes = type.getElementType().getIntOrFloatBitWidth() / 8;
  return !span || ShapedType::isDynamic(offset) ||
         offset <= INT64_MAX / bytes - *span;
}

// Prefer module specifications; only an absent specification uses target defaults.
static FailureOr<int64_t> getUBCapacityBytes(ModuleOp module) {
  int64_t capacityBits, alignmentBits;
  if (module.getTargetSystemSpec()) {
    auto spec = hacc::utils::getNPUTargetSpec(module);
    if (!spec || !*spec)
      return failure();
    auto getBits = [&](hacc::DeviceSpec identifier) -> int64_t {
      auto entry = spec->getSpecForIdentifierEnum(identifier);
      if (!entry)
        return 0;
      auto value = dyn_cast<IntegerAttr>(entry.getValue());
      return value && value.getValue().isSignedIntN(64) ? value.getInt() : 0;
    };
    capacityBits = getBits(hacc::DeviceSpec::UB_SIZE);
    alignmentBits = getBits(hacc::DeviceSpec::UB_ALIGN_SIZE);
  } else {
    auto target = hacc::utils::getTargetDevice(module);
    if (!target)
      return failure();
    auto spec = hacc::lookupTargetSpec(*target);
    if (!spec)
      return failure();
    capacityBits = (*spec)->UbSize;
    alignmentBits = (*spec)->UbAlignSize;
  }
  if (capacityBits <= 0 || alignmentBits <= 0 || capacityBits % 8 != 0 ||
      alignmentBits % 8 != 0)
    return failure();
  int64_t capacityBytes = capacityBits / 8;
  int64_t alignmentBytes = alignmentBits / 8;
  // Rounding the budget down also bounds the aligned output allocation.
  return capacityBytes / alignmentBytes * alignmentBytes;
}

static Value asGMTypedSource(PatternRewriter &rewriter, Location loc,
                             Value source) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || type.getMemorySpace())
    return source;
  auto gm = hivm::AddressSpaceAttr::get(rewriter.getContext(),
                                        hivm::AddressSpace::GM);
  auto gmType = MemRefType::get(type.getShape(), type.getElementType(),
                                type.getLayout(), gm);
  return rewriter.create<memref::MemorySpaceCastOp>(loc, gmType, source);
}

// A one-level subview that provably covers the whole temporary is the same copy
// as a direct one: it keeps every extent, stride and the zero offset of the
// allocation. Partial, dynamic or permuted views write a strict sub-region (the
// masked frontend form) and need the pad contract this entry does not own, so
// they keep the original copy + vector broadcast chain.
static bool coversWholeAllocation(memref::SubViewOp view,
                                  memref::AllocOp alloc) {
  auto allocType = alloc.getType();
  auto viewType = dyn_cast<MemRefType>(view.getType());
  if (!viewType || view.getSource() != alloc.getResult() ||
      !allocType.hasStaticShape() || !viewType.hasStaticShape() ||
      viewType.getRank() != allocType.getRank() ||
      viewType.getElementType() != allocType.getElementType())
    return false;
  SmallVector<int64_t> allocStrides, viewStrides;
  int64_t allocOffset = 0, viewOffset = 0;
  if (failed(mlir::utils::getStridesAndOffsetCompat(allocType, allocStrides,
                                                    allocOffset)) ||
      failed(mlir::utils::getStridesAndOffsetCompat(viewType, viewStrides,
                                                    viewOffset)) ||
      allocOffset != 0 || viewOffset != 0)
    return false;
  for (int64_t i = 0; i < allocType.getRank(); ++i) {
    if (viewType.getDimSize(i) != allocType.getDimSize(i) ||
        viewStrides[i] != allocStrides[i] ||
        getConstantIntValue(view.getMixedOffsets()[i]) != 0 ||
        getConstantIntValue(view.getMixedStrides()[i]) != 1)
      return false;
  }
  return true;
}

// Values needed to validate and erase the private transfer chain.
struct BroadcastProducerChain {
  tensor::CollapseShapeOp collapse;
  bufferization::ToTensorOp toTensor;
  memref::AllocOp alloc;
  Operation *copy = nullptr;
  Value transferSource;
  bool explicitLoad = false;
  memref::SubViewOp copyDstView;
};

// Types and reshape information consumed when emitting the fused load.
struct BroadcastLoadInfo {
  MemRefType reshapeType;
  MemRefType expandedType;
  MemRefType destinationType;
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<OpFoldResult> expandedSizes;
};

static LogicalResult checkBroadcastLegality(linalg::BroadcastOp op,
                                            SmallVectorImpl<bool> &axes) {
  if (!op.hasPureTensorSemantics())
    return failure();

  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || func->hasAttr("hivm.vector_function"))
    return failure();
  ArrayRef<int64_t> broadcastDims = op.getDimensionsAttr().asArrayRef();
  auto srcType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto dstType = dyn_cast<RankedTensorType>(op.getInit().getType());
  if (!srcType || !dstType || broadcastDims.empty() ||
      (dstType.getRank() != 2 && dstType.getRank() != 3) ||
      srcType.getRank() + static_cast<int64_t>(broadcastDims.size()) !=
          dstType.getRank() ||
      srcType.getEncoding() || dstType.getEncoding())
    return failure();
  const int64_t rank = dstType.getRank();
  axes.assign(rank, false);
  for (int64_t axis : broadcastDims) {
    if (axis < 0 || axis >= rank || axes[axis] ||
        (!dstType.isDynamicDim(axis) &&
         (dstType.getDimSize(axis) <= 1 || dstType.getDimSize(axis) > 0xfffff)))
      return failure();
    axes[axis] = true;
  }
  auto empty = op.getInit().getDefiningOp<tensor::EmptyOp>();
  if (!empty || !srcType.hasStaticShape() || !dstType.hasStaticShape() ||
      srcType.getRank() < 1)
    return failure();
  SmallVector<int64_t> allocationShape(dstType.getShape());
  if (llvm::any_of(allocationShape, [](int64_t size) { return size <= 0; }))
    return failure();

  Type elementType = srcType.getElementType();
  if (!isSupportedElementType(elementType))
    return failure();

  return success();
}

static FailureOr<BroadcastProducerChain> matchBroadcastProducerChain(linalg::BroadcastOp op) {
  BroadcastProducerChain chain;
  Value input = op.getInput();
  chain.collapse = input.getDefiningOp<tensor::CollapseShapeOp>();
  if (chain.collapse) {
    if (!input.hasOneUse())
      return failure();
    input = chain.collapse.getSrc();
  }
  chain.toTensor = input.getDefiningOp<bufferization::ToTensorOp>();
  if (!chain.toTensor || !input.hasOneUse() || !chain.toTensor.getRestrict())
    return failure();
  chain.alloc = chain.toTensor.getMemref().getDefiningOp<memref::AllocOp>();
  if (!chain.alloc || !chain.alloc.getType().getLayout().isIdentity() ||
      (chain.collapse && !chain.alloc.getType().hasStaticShape()))
    return failure();

  auto space = chain.alloc.getType().getMemorySpace();
  auto ub = hivm::AddressSpaceAttr::get(op.getContext(), hivm::AddressSpace::UB);
  if (space && space != ub)
    return failure();

  // Classify every user of the private temporary. Each supported form must
  // prove its own legality before registering the unique writer; keep scanning
  // after a candidate to reject competing writers and unknown users.
  for (Operation *user : chain.alloc->getUsers()) {
    if (user == chain.toTensor.getOperation())
      continue;
    Operation *transfer = user;
    Value transferSource;
    bool explicitLoad = false;
    memref::SubViewOp dstView;
    if (auto candidate = dyn_cast<memref::CopyOp>(user)) {
      if (candidate.getTarget() != chain.alloc.getResult())
        return failure();
      transferSource = candidate.getSource();
    } else if (auto candidate = dyn_cast<hivm::LoadOp>(user)) {
      if (candidate.getDst() != chain.alloc.getResult() ||
          candidate.getPadMode() || candidate.getPadValue() ||
          candidate.getLeftPaddingNum() || candidate.getRightPaddingNum() ||
          candidate.getInitOutBuffer() || candidate.getInitCondition() ||
          candidate.getMayImplicitTransposeWithLastAxis() ||
          candidate.getBroadcastAxesAttr())
        return failure();
      transferSource = candidate.getSrc();
      explicitLoad = true;
    } else if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      if (!coversWholeAllocation(view, chain.alloc) ||
          !view.getResult().hasOneUse())
        return failure();
      auto candidate =
          dyn_cast<memref::CopyOp>(*view.getResult().getUsers().begin());
      if (!candidate || candidate.getTarget() != view.getResult())
        return failure();
      transfer = candidate;
      transferSource = candidate.getSource();
      dstView = view;
    } else {
      return failure();
    }
    if (chain.copy)
      return failure();
    chain.copy = transfer;
    chain.transferSource = transferSource;
    chain.explicitLoad = explicitLoad;
    chain.copyDstView = dstView;
  }
  if (!chain.copy)
    return failure();
  if (chain.copy->getBlock() != op->getBlock() ||
      chain.toTensor->getBlock() != op->getBlock() ||
      !chain.copy->isBeforeInBlock(chain.toTensor) ||
      !chain.toTensor->isBeforeInBlock(op))
    return failure();
  return chain;
}

static FailureOr<BroadcastLoadInfo>
computeBroadcastLoadInfo(linalg::BroadcastOp op, BroadcastProducerChain chain,
                  ArrayRef<bool> axes, PatternRewriter &rewriter) {
  auto srcType = cast<RankedTensorType>(op.getInput().getType());
  auto dstType = cast<RankedTensorType>(op.getInit().getType());
  auto dstSizes = op.getInit().getDefiningOp<tensor::EmptyOp>().getMixedSizes();
  SmallVector<int64_t> allocationShape(dstType.getShape());
  Type elementType = srcType.getElementType();
  const int64_t rank = dstType.getRank();
  auto gmType = cast<MemRefType>(chain.transferSource.getType());
  if (!isGMSource(chain.transferSource, op->getParentOfType<func::FuncOp>(),
                  chain.explicitLoad) ||
      gmType.getShape() != chain.alloc.getType().getShape() ||
      !gmType.hasStaticShape())
    return failure();
  // A reshape of the loaded buffer is valid for the GM view only when both
  // have the same packed physical order. Unknown layouts keep the old chain.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(mlir::utils::getStridesAndOffsetCompat(gmType, strides, offset)) ||
      (!ShapedType::isDynamic(offset) && offset < 0))
    return failure();
  if (chain.collapse) {
    int64_t expectedStride = 1;
    for (int64_t i = gmType.getRank() - 1; i >= 0; --i) {
      if (strides[i] != expectedStride || gmType.getDimSize(i) <= 0 ||
          gmType.getDimSize(i) > INT64_MAX / expectedStride)
        return failure();
      expectedStride *= gmType.getDimSize(i);
    }
  } else if (gmType.getRank() > 0 && strides.back() != 1) {
    return failure();
  }
  if ((!chain.collapse && gmType.getRank() != srcType.getRank()) ||
      (chain.collapse &&
       chain.collapse.getResultType().getRank() != srcType.getRank()))
    return failure();

  SmallVector<OpFoldResult> sourceSizes;
  for (int64_t size : srcType.getShape())
    sourceSizes.push_back(rewriter.getIndexAttr(size));
  if (llvm::any_of(strides,
                   [](int64_t stride) {
                     return ShapedType::isDynamic(stride) || stride <= 0;
                   }) ||
      !hasProvenLoadSpan(gmType))
    return failure();
  auto capacity = getUBCapacityBytes(op->getParentOfType<ModuleOp>());
  if (failed(capacity))
    return failure();
  int64_t ubCapacityBytes = *capacity;
  int64_t allocationBytes = elementType.getIntOrFloatBitWidth() / 8;
  for (int64_t size : allocationShape) {
    if (size > ubCapacityBytes / allocationBytes) {
      (void)rewriter.notifyMatchFailure(
          op, "single broadcast output allocation exceeds UB capacity");
      return failure();
    }
    allocationBytes *= size;
  }
  int64_t inputAxis = 0;
  for (int64_t axis = 0; axis < rank; ++axis)
    if (!axes[axis] &&
        !isEqualConstantIntOrValue(sourceSizes[inputAxis++], dstSizes[axis]))
      return failure();

  SmallVector<int64_t> shape(rank, 1);
  SmallVector<OpFoldResult> expandedSizes(rank, rewriter.getIndexAttr(1));
  SmallVector<ReassociationIndices> reassociation(srcType.getRank());
  int64_t kept = 0;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (!axes[axis]) {
      shape[axis] = srcType.getDimSize(kept);
      expandedSizes[axis] = sourceSizes[kept++];
    }
    reassociation[std::max<int64_t>(0, kept - 1)].push_back(axis);
  }
  auto reshapeType = gmType;
  if (chain.collapse) {
    // Packed source was proven above. Use the collapse type contract so an
    // identity layout stays identity and strided offsets/spaces are preserved.
    reshapeType = memref::CollapseShapeOp::computeCollapsedType(
        gmType, chain.collapse.getReassociationIndices());
  }
  auto expanded = memref::ExpandShapeOp::computeExpandedType(reshapeType, shape,
                                                             reassociation);
  if (failed(expanded) || !hasProvenLoadSpan(*expanded))
    return failure();
  auto destinationType = MemRefType::get(
      allocationShape, elementType, MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(rewriter.getContext(), hivm::AddressSpace::UB));
  return BroadcastLoadInfo{reshapeType, *expanded, destinationType,
                           std::move(reassociation), std::move(expandedSizes)};
}

// Admit only the broadcast NDDMA fast path in Copy2D/Copy3D. The preceding
// stages bound the single output allocation by UB capacity and prevent size
// calculation overflow. Whole-function memory planning remains with PlanMemory.
static LogicalResult
checkBroadcastLoadOptimization(linalg::BroadcastOp op,
                               const BroadcastLoadInfo &loadInfo,
                               PatternRewriter &rewriter) {
  auto dstType = cast<RankedTensorType>(op.getInit().getType());
  for (int64_t size : dstType.getShape())
    if (ShapedType::isDynamic(size) || size <= 0 || size > 0xfffff)
      return rewriter.notifyMatchFailure(op, "broadcast NDDMA size limit");
  int64_t elementBytes = dstType.getElementType().getIntOrFloatBitWidth() / 8;
  // Supported elements are 1/2/4 bytes; use division to avoid a byte product.
  if (dstType.getShape().back() % (128 / elementBytes) != 0)
    return rewriter.notifyMatchFailure(op, "broadcast NDDMA row alignment");

  SmallVector<int64_t> srcStrides, dstStrides;
  int64_t srcOffset, dstOffset;
  if (failed(mlir::utils::getStridesAndOffsetCompat(loadInfo.expandedType,
                                                    srcStrides, srcOffset)) ||
      failed(mlir::utils::getStridesAndOffsetCompat(loadInfo.destinationType,
                                                    dstStrides, dstOffset)))
    return rewriter.notifyMatchFailure(op, "unknown broadcast NDDMA layout");
  for (int64_t axis = 0; axis < dstType.getRank() - 1; ++axis)
    if (ShapedType::isDynamic(srcStrides[axis]) || srcStrides[axis] < 0 ||
        srcStrides[axis] > 0xffffffffLL ||
        ShapedType::isDynamic(dstStrides[axis]) || dstStrides[axis] < 0 ||
        dstStrides[axis] > 0xfffff)
      return rewriter.notifyMatchFailure(op, "broadcast NDDMA stride limit");
  return success();
}

/// Preserve the original GM read point, existing load attributes and erase
/// order.
static void rewriteBroadcastLoad(linalg::BroadcastOp op, BroadcastProducerChain chain,
                                 const BroadcastLoadInfo &loadInfo,
                                 PatternRewriter &rewriter) {
  ArrayRef<int64_t> broadcastDims = op.getDimensionsAttr().asArrayRef();
  Location loc = op.getLoc();
  Value destination;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(chain.copy);
    Value gmSource = chain.transferSource;
    if (chain.collapse)
      gmSource = rewriter.create<memref::CollapseShapeOp>(
          loc, loadInfo.reshapeType, gmSource,
          chain.collapse.getReassociationIndices());
    gmSource = rewriter.create<memref::ExpandShapeOp>(
        loc, loadInfo.expandedType, gmSource, loadInfo.reassociation,
        loadInfo.expandedSizes);
    destination =
        rewriter.create<memref::AllocOp>(loc, loadInfo.destinationType);
    gmSource = asGMTypedSource(rewriter, loc, gmSource);
    hivm::LoadOp load;
    if (chain.explicitLoad) {
      load = cast<hivm::LoadOp>(rewriter.clone(*chain.copy));
      load.getSrcMutable().assign(gmSource);
      load.getDstMutable().assign(destination);
    } else {
      load = rewriter.create<hivm::LoadOp>(loc, TypeRange{}, gmSource,
                                           destination);
    }
    load.setBroadcastAxesAttr(rewriter.getDenseI64ArrayAttr(broadcastDims));
  }
  rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
      op, destination, /*restrict=*/true, /*writable=*/true);
  if (chain.collapse)
    rewriter.eraseOp(chain.collapse);
  rewriter.eraseOp(chain.toTensor);
  rewriter.eraseOp(chain.copy);
  if (chain.copyDstView)
    rewriter.eraseOp(chain.copyDstView);
  rewriter.eraseOp(chain.alloc);
}

struct FuseBroadcastLoadPattern : public OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module || !hacc::utils::isAscend960(module))
      return failure();

    SmallVector<bool> axes;
    if (failed(checkBroadcastLegality(op, axes)))
      return failure();
    auto chain = matchBroadcastProducerChain(op);
    if (failed(chain))
      return failure();
    auto loadInfo = computeBroadcastLoadInfo(op, *chain, axes, rewriter);
    if (failed(loadInfo))
      return failure();
    if (failed(checkBroadcastLoadOptimization(op, *loadInfo, rewriter)))
      return failure();

    // All fallible analysis is complete before touching the IR.
    rewriteBroadcastLoad(op, *chain, *loadInfo, rewriter);
    return success();
  }
};

} // namespace

void mlir::hivm::detail::populateBroadcastLoadPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FuseBroadcastLoadPattern>(patterns.getContext());
}
