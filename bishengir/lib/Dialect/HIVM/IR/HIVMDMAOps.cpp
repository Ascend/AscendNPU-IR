//===- HIVMDMAOps.cpp - HIVM DMA ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/Support/FormatVariadic.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;

namespace {
// TODO: use stringifyAddressSpace after the library call names are consistent
std::map<AddressSpace, std::string> kAddressSpace2LibraryName = {
    {AddressSpace::UB, "ubuf"},
    {AddressSpace::GM, "gm"},
    {AddressSpace::L1, "cbuf"}};

std::string getLibraryCallNameForCopyLikeOp(std::string baseCallName,
                                            Type srcType, Type dstType,
                                            Location loc, int rank) {
  auto srcScope = getHIVMAddressSpace(srcType);
  assert(kAddressSpace2LibraryName.find(srcScope) !=
             kAddressSpace2LibraryName.cend() &&
         "Unsupported src address space");
  auto dstScope = getHIVMAddressSpace(dstType);
  assert(kAddressSpace2LibraryName.find(dstScope) !=
             kAddressSpace2LibraryName.cend() &&
         "Unsupported dst address space");
  std::string srcScopeName = kAddressSpace2LibraryName.at(srcScope);
  std::string dstScopeName = kAddressSpace2LibraryName.at(dstScope);
  std::string src2DstName =
      llvm::formatv("{0}_to_{1}", srcScopeName, dstScopeName);

  std::string dataTypeStr =
      hivm::detail::getTypeName(loc, getElementTypeOrSelf(srcType));
  std::string libCallDim = std::to_string(rank) + "d";

  std::string callLibraryName = llvm::formatv(
      "{0}_{1}_{2}_{3}", baseCallName, src2DstName, libCallDim, dataTypeStr);
  return callLibraryName;
}
} // namespace

#define ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(OP_NAME)                \
  Value OP_NAME::getSource() { return getSrc(); }                              \
  Value OP_NAME::getTarget() { return getDst(); }

ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(CopyOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(LoadOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(StoreOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(FixpipeOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(ND2NZOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(NZ2NDOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(L12UBOp)
#undef ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static LogicalResult checkLoadOpMemSpace(LoadOp &op) {
  auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
  auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
  auto srcMemSpaceAttr = srcMemRefType.getMemorySpace();
  auto dstMemSpaceAttr = dstMemRefType.getMemorySpace();
  if (srcMemSpaceAttr && dstMemSpaceAttr) {
    auto srcAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(srcMemSpaceAttr);
    auto dstAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr);
    if (!srcAddrSpaceAttr) {
      return op.emitOpError("cast src memory space attr failed!");
    }
    if (!dstAddrSpaceAttr) {
      return op.emitOpError("cast dst memory space attr failed!");
    }

    auto srcAddrSpace = srcAddrSpaceAttr.getAddressSpace();
    auto dstAddrSpace = dstAddrSpaceAttr.getAddressSpace();

    bool isSrcGm = srcAddrSpace == AddressSpace::GM;
    bool isDstGm = dstAddrSpace == AddressSpace::GM;

    if (!isSrcGm || isDstGm) {
      return op.emitOpError("only support src == gm and dst != gm currently!");
    }
  }

  return success();
}

static LogicalResult checkLoadOpTensor(LoadOp &op) {
  ShapedType dstOperType = op.getDstOperandType();
  auto resTensorType = cast<RankedTensorType>(op.getResultTensor().getType());
  if (dstOperType.getElementType() != resTensorType.getElementType()) {
    return op.emitOpError(
        "element types of dst src and res should be the same!");
  }

  if (!resTensorType.hasRank()) {
    return op.emitOpError("res should have a known number of dimensions!");
  }

  if (resTensorType.getRank() != dstOperType.getRank()) {
    return op.emitOpError("res and dst should have the same dimensions!");
  }

  auto resShape = resTensorType.getShape();
  if (!op.getPadMode() &&
      failed(verifyCompatibleShape(resShape, dstOperType.getShape()))) {
    return op.emitOpError(
        "if pad_mode is not set, res and dst shape should be the same!");
  }

  return success();
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst) {
  build(odsBuilder, odsState, res, src, dst, /*pad_mode=*/nullptr,
        /*pad_value=*/nullptr, /*left_padding_num=*/nullptr,
        /*right_padding_num=*/nullptr,
        /*init_out_buffer=*/false, /*init_condition=*/nullptr,
        /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst,
                   Value left_padding_num) {
  build(odsBuilder, odsState, res, src, dst, /*pad_mode=*/nullptr,
        /*pad_value=*/nullptr, left_padding_num,
        /*right_padding_num=*/nullptr, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr, /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        /*left_padding_num=*/nullptr,
        /*right_padding_num=*/nullptr, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr, /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr,
        /*init_out_buffer=*/false, /*init_condition=*/nullptr,
        /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   bool init_out_buffer) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr, init_out_buffer,
        /*init_condition=*/nullptr, /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   Value right_padding_num) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, right_padding_num, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr, /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   bool init_out_buffer, Value init_condition) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr, init_out_buffer,
        init_condition, /*eviction_policy=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, EvictionPolicyAttr eviction_policy) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        /*left_padding_num=*/nullptr,
        /*right_padding_num=*/nullptr,
        /*init_out_buffer=*/false, /*init_condition=*/nullptr, eviction_policy);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   EvictionPolicyAttr eviction_policy) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num,
        /*right_padding_num=*/nullptr,
        /*init_out_buffer=*/false, /*init_condition=*/nullptr, eviction_policy);
}

LogicalResult LoadOp::verify() {
  // check element type of src and dst
  ShapedType srcOperType = getSrcOperandType();
  ShapedType dstOperType = getDstOperandType();
  if (srcOperType.getElementType() != dstOperType.getElementType()) {
    return emitOpError("element types of dst and src should be the same!");
  }

  // check rank of src dst
  if (!srcOperType.hasRank() || !dstOperType.hasRank()) {
    return emitOpError("src and dst should have a known number of dimensions!");
  }

  auto srcShape = srcOperType.getShape();
  auto dstShape = dstOperType.getShape();
  if (srcOperType.getRank() != dstOperType.getRank()) {
    return emitOpError("src and dst should have the same dimensions!");
  }

  // if not set padmode, means dst/src shape is the same
  auto padModeAttr = getPadMode();
  if (!padModeAttr && failed(verifyCompatibleShape(srcShape, dstShape))) {
    return emitOpError(
        "if pad_mode is not set, src and dst shape should be the same!");
  }

  // check pad value
  auto padval = getPadValue();
  if (padModeAttr) {
    PadMode pm = padModeAttr->getPadmode();
    if (pm == PadMode::PadValue && !padval) {
      return emitOpError("if padmode is PadValue, pad_value is required!");
    }
  }

  // check padval dtype
  if (padval && padval.getType() != dstOperType.getElementType()) {
    return emitOpError(
        "dtype of pad_value and element type of dst/src should be the same!");
  }

  // check mem space in case of memref
  if (hasPureBufferSemantics()) {
    return checkLoadOpMemSpace(*this);
  }

  // in case of tensor
  if (hasPureTensorSemantics()) {
    return checkLoadOpTensor(*this);
  }

  return emitOpError("dst/src should be memref/memref or tensor/tensor, res "
                     "should be tensor!");
}

LogicalResult LoadOp::fold(hivm::LoadOp::FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

std::string LoadOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  MemRefType srcMemref = cast<MemRefType>(this->getSrcOperandType());
  assert(srcMemref.getMemorySpace() &&
         "Source should have memory space by now.");
  int64_t rank = srcMemref.getRank();
  auto baseCallName = getLibraryCallNameForCopyLikeOp(
      this->getOpName().str(), this->getSrc().getType(),
      this->getDst().getType(), this->getLoc(), getOpLibraryCallRank(rank));
  return baseCallName;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static LogicalResult checkStoreOpMemSpace(StoreOp &op) {
  auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
  auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
  auto srcMemSpaceAttr = srcMemRefType.getMemorySpace();
  auto dstMemSpaceAttr = dstMemRefType.getMemorySpace();
  if (srcMemSpaceAttr && dstMemSpaceAttr) {
    auto srcAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(srcMemSpaceAttr);
    auto dstAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr);
    if (!srcAddrSpaceAttr) {
      return op.emitOpError("cast src memory space attr failed!");
    }
    if (!dstAddrSpaceAttr) {
      return op.emitOpError("cast dst memory space attr failed!");
    }

    auto srcAddrSpace = srcAddrSpaceAttr.getAddressSpace();
    auto dstAddrSpace = dstAddrSpaceAttr.getAddressSpace();

    bool isUbtoGm =
        srcAddrSpace == AddressSpace::UB && dstAddrSpace == AddressSpace::GM;

    if (!isUbtoGm) {
      return op.emitOpError("only support copy gm to ub or copy ub to gm or "
                            "copy ub to ub currently!");
    }
  }

  return success();
}

static LogicalResult checkStoreOpTensor(StoreOp &op) {
  ShapedType dstOperType = op.getDstOperandType();
  auto resTensorType = cast<RankedTensorType>(op.getResultTensor().getType());
  if (dstOperType.getElementType() != resTensorType.getElementType()) {
    return op.emitOpError(
        "element types of dst src and res should be the same!");
  }

  if (!resTensorType.hasRank()) {
    return op.emitOpError("res should have a known number of dimensions!");
  }

  if (resTensorType.getRank() != dstOperType.getRank()) {
    return op.emitOpError("res and dst should have the same dimensions!");
  }

  return success();
}

void StoreOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange res, Value src, Value dst) {
  build(odsBuilder, odsState, res, src, dst, /*atomic_kind=*/nullptr);
}

LogicalResult StoreOp::verify() {
  // check element type of src and dst
  ShapedType srcOperType = getSrcOperandType();
  ShapedType dstOperType = getDstOperandType();
  if (srcOperType.getElementType() != dstOperType.getElementType()) {
    return emitOpError("element types of dst and src should be the same!");
  }

  // check rank of src dst
  if (!srcOperType.hasRank() || !dstOperType.hasRank()) {
    return emitOpError("src and dst should have a known number of dimensions!");
  }

  if (srcOperType.getRank() != dstOperType.getRank()) {
    return emitOpError("src and dst should have the same dimensions!");
  }

  // check mem space in case of memref
  if (hasPureBufferSemantics()) {
    return checkStoreOpMemSpace(*this);
  }

  // in case of tensor
  if (hasPureTensorSemantics()) {
    return checkStoreOpTensor(*this);
  }

  return success();
}

bool StoreOp::isAtomic() {
  auto atomicKind = getAtomicKind();
  return atomicKind.has_value() && atomicKind.value() != hivm::AtomicKind::NONE;
}

bool StoreOp::isHWAtomic() {
  if (getAtomicKind().has_value()) {
    auto atomicKind = getAtomicKind().value();
    return (atomicKind == hivm::AtomicKind::ADD) ||
           (atomicKind == hivm::AtomicKind::MAX) ||
           (atomicKind == hivm::AtomicKind::MIN);
  }

  return false;
}

bool StoreOp::isSWAtomic() { return isAtomic() && (!isHWAtomic()); }

LogicalResult StoreOp::fold(hivm::StoreOp::FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  if (succeeded(memref::foldMemRefCast(*this)))
    return success();
  if (succeeded(memref::foldMemRefSpaceCast(*this)))
    return success();
  return failure();
}

std::string StoreOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  MemRefType srcMemref = cast<MemRefType>(this->getSrcOperandType());
  assert(srcMemref.getMemorySpace() &&
         "Source should have memory space by now.");
  int64_t rank = srcMemref.getRank();
  auto baseCallName = getLibraryCallNameForCopyLikeOp(
      this->getOpName().str(), this->getSrc().getType(),
      this->getDst().getType(), this->getLoc(), getOpLibraryCallRank(rank));
  return baseCallName;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

static LogicalResult checkCopyOpMemSpace(CopyOp &op) {
  auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
  auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
  auto srcMemSpaceAttr = srcMemRefType.getMemorySpace();
  auto dstMemSpaceAttr = dstMemRefType.getMemorySpace();
  // As infer memscope is supported, memscope is not required.
  // But if memscope exists, only support gm/ub.
  if (srcMemSpaceAttr && dstMemSpaceAttr) {
    auto srcAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(srcMemSpaceAttr);
    auto dstAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr);
    if (!srcAddrSpaceAttr) {
      return op.emitOpError("cast src memory space attr failed!");
    }
    if (!dstAddrSpaceAttr) {
      return op.emitOpError("cast dst memory space attr failed!");
    }

    auto srcAddrSpace = srcAddrSpaceAttr.getAddressSpace();
    auto dstAddrSpace = dstAddrSpaceAttr.getAddressSpace();

    static DenseSet<std::pair<AddressSpace, AddressSpace>> kCopySupported{
        {std::make_pair(AddressSpace::UB, AddressSpace::UB)},
        {std::make_pair(AddressSpace::GM, AddressSpace::L1)},
    };
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (hacc::utils::isAscend950(moduleOp)) {
      kCopySupported.insert(
          {std::make_pair(AddressSpace::UB, AddressSpace::L1)});
    }

    if (!kCopySupported.count(std::make_pair(srcAddrSpace, dstAddrSpace))) {
      auto srcStr = stringifyAddressSpace(srcAddrSpace).str();
      auto dstStr = stringifyAddressSpace(dstAddrSpace).str();
      return op.emitOpError()
             << "Unsupported copy from " << srcStr << " to " << dstStr << "!";
    }
  }

  return success();
}

static LogicalResult checkCopyOpTensor(CopyOp &op) {
  ShapedType dstOperType = op.getDstOperandType();
  RankedTensorType resTensorType = op.getResultTensor().getType();
  if (dstOperType.getElementType() != resTensorType.getElementType()) {
    return op.emitOpError(
        "element types of dst src and res should be the same!");
  }

  if (!resTensorType.hasRank()) {
    return op.emitOpError("res should have a known number of dimensions!");
  }

  if (resTensorType.getRank() != dstOperType.getRank()) {
    return op.emitOpError("res and dst should have the same dimensions!");
  }

  auto resShape = resTensorType.getShape();
  if (!op.getPadMode() &&
      failed(verifyCompatibleShape(resShape, dstOperType.getShape()))) {
    return op.emitOpError(
        "if pad_mode is not set, res and dst shape should be the same!");
  }

  return success();
}

static LogicalResult checkCopyOpMixTensorAndMemRef(CopyOp &op) {
  ShapedType dstOperType = op.getDstOperandType();
  auto resultTensor = op.getResultTensor();
  if (resultTensor) {
    RankedTensorType resTensorType = resultTensor.getType();
    if (dstOperType.getElementType() != resTensorType.getElementType()) {
      return op.emitOpError(
          "element types of dst src and res should be the same!");
    }

    if (!resTensorType.hasRank()) {
      return op.emitOpError("res should have a known number of dimensions!");
    }

    if (resTensorType.getRank() != dstOperType.getRank()) {
      return op.emitOpError("res and dst should have the same dimensions!");
    }

    auto resShape = resTensorType.getShape();
    if (!op.getPadMode() &&
        failed(verifyCompatibleShape(resShape, dstOperType.getShape()))) {
      return op.emitOpError(
          "if pad_mode is not set, res and dst shape should be the same!");
    }
  }

  // check dst memref
  auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
  auto dstMemSpaceAttr = dstMemRefType.getMemorySpace();
  // As infer memscope is supported, memscope is not required.
  // But if memscope exists, only support gm/ub.
  if (dstMemSpaceAttr) {
    auto dstAddrSpaceAttr = dyn_cast<AddressSpaceAttr>(dstMemSpaceAttr);
    if (!dstAddrSpaceAttr) {
      return success();
    }

    auto dstAddrSpace = dstAddrSpaceAttr.getAddressSpace();

    static DenseSet<AddressSpace> kCopySupported{
        AddressSpace::UB,
        AddressSpace::L1,
    };

    if (!kCopySupported.count(dstAddrSpace)) {
      auto dstStr = stringifyAddressSpace(dstAddrSpace).str();
      return op.emitOpError() << "Unsupported copy to " << dstStr << "!";
    }
  }

  return success();
}

void CopyOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst) {
  build(odsBuilder, odsState, res, src, dst, /*pad_mode=*/nullptr,
        /*pad_value=*/nullptr);
}

void CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  detail::getEffectsImpl(effects, cast<HIVMStructuredOp>(getOperation()));
}

LogicalResult CopyOp::verify() {
  // check element type of src and dst
  ShapedType srcOperType = getSrcOperandType();
  ShapedType dstOperType = getDstOperandType();
  if (srcOperType.getElementType() != dstOperType.getElementType()) {
    return emitOpError("element types of dst and src should be the same!");
  }

  // check rank of src dst
  if (!srcOperType.hasRank() || !dstOperType.hasRank()) {
    return emitOpError("src and dst should have a known number of dimensions!");
  }

  auto srcShape = srcOperType.getShape();
  auto dstShape = dstOperType.getShape();
  if (srcOperType.getRank() != dstOperType.getRank()) {
    return emitOpError("src and dst should have the same dimensions!");
  }

  // if not set padmode, means dst/src shape is the same
  auto padModeAttr = getPadMode();
  if (!padModeAttr && failed(verifyCompatibleShape(srcShape, dstShape))) {
    return emitOpError(
        "if pad_mode is not set, src and dst shape should be the same!");
  }

  // check pad value
  auto padval = getPadValue();
  if (padModeAttr) {
    PadMode pm = padModeAttr->getPadmode();
    if (pm == PadMode::PadValue && !padval) {
      return emitOpError("if padmode is PadValue, pad_value is required!");
    }
  }

  // check padval dtype
  if (padval && padval.getType() != dstOperType.getElementType()) {
    return emitOpError(
        "dtype of pad_value and element type of dst/src should be the same!");
  }

  // check mem space in case of memref
  if (hasPureBufferSemantics()) {
    return checkCopyOpMemSpace(*this);
  }

  // in case of tensor
  if (hasPureTensorSemantics()) {
    return checkCopyOpTensor(*this);
  }

  // in case of tensor mix memref
  return checkCopyOpMixTensorAndMemRef(*this);
}

PIPE CopyOp::getPipe() {
  assert(hasPureBufferSemantics() && "Operating on tensor, please bufferize.");
  MemRefType srcMemrefType = dyn_cast<MemRefType>(getSrcOperandType());
  MemRefType dstMemrefType = dyn_cast<MemRefType>(getDstOperandType());
  auto srcMemSpaceAttr = srcMemrefType.getMemorySpace();
  auto dstMemSpaceAttr = dstMemrefType.getMemorySpace();
  assert(srcMemSpaceAttr && "Source should have memory space by now.");
  assert(dstMemSpaceAttr && "dst should have memory space by now.");

  const DenseMap<std::pair<AddressSpace, AddressSpace>, PIPE> kSrcDstSpace2Pipe{
      {std::make_pair(AddressSpace::UB, AddressSpace::UB), PIPE::PIPE_V},
      {std::make_pair(AddressSpace::L0C, AddressSpace::GM), PIPE::PIPE_FIX},
      {std::make_pair(AddressSpace::GM, AddressSpace::L1), PIPE::PIPE_MTE2},
      {std::make_pair(AddressSpace::UB, AddressSpace::L1), PIPE::PIPE_MTE3},

  };

  auto nowSrcDstSpace =
      std::make_pair(cast<AddressSpaceAttr>(srcMemSpaceAttr).getAddressSpace(),
                     cast<AddressSpaceAttr>(dstMemSpaceAttr).getAddressSpace());
  auto iter = kSrcDstSpace2Pipe.find(nowSrcDstSpace);
  if (iter != kSrcDstSpace2Pipe.end()) {
    return iter->second;
  }
  llvm_unreachable("Unknown PIPE!");
}

std::string CopyOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  MemRefType srcMemref = cast<MemRefType>(this->getSrcOperandType());
  assert(srcMemref.getMemorySpace() &&
         "Source should have memory space by now.");
  int64_t rank = srcMemref.getRank();
  auto baseCallName = getLibraryCallNameForCopyLikeOp(
      this->getOpName().str(), this->getSrc().getType(),
      this->getDst().getType(), this->getLoc(), getOpLibraryCallRank(rank));
  return baseCallName;
}

LogicalResult CopyOp::fold(hivm::CopyOp::FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// ND2NZOp
//===----------------------------------------------------------------------===//

void ND2NZOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  detail::getEffectsImpl(effects, cast<HIVMStructuredOp>(getOperation()));
}

std::string ND2NZOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  std::string callName = getOpName().str();
  for (Operation *nextOp : getDst().getUsers()) {
    if (auto mmadl1Op = llvm::dyn_cast<hivm::MmadL1Op>(nextOp)) {
      if (mmadl1Op.getPerChannelBias() == getDst())
        callName = callName + "_forbias";
    }
  }
  Type eleType = getElementTypeOrSelf(this->getDpsInputs()[0].getType());
  auto elemTypeName = hivm::detail::getTypeName(this->getLoc(), eleType);
  return callName + "_" + elemTypeName;
}

void ND2NZOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange res, Value src, Value dst,
                    UnitAttr dst_continuous) {
  build(odsBuilder, odsState, res, src, dst, dst_continuous,
        /*init_out_buffer=*/false,
        /*pad_value=*/nullptr, /*init_condition=*/nullptr);
}

void ND2NZOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange res, Value src, Value dst,
                    UnitAttr dst_continuous, bool init_out_buffer,
                    Value pad_value) {
  build(odsBuilder, odsState, res, src, dst, dst_continuous, init_out_buffer,
        pad_value, /*init_condition=*/nullptr);
}

//===----------------------------------------------------------------------===//
// NZ2NDOp
//===----------------------------------------------------------------------===//

std::string NZ2NDOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  // check address space
  Type srcType = this->getSrc().getType();
#ifndef NDEBUG
  AddressSpace srcScope = getHIVMAddressSpace(srcType);
  assert(srcScope == AddressSpace::L1 && "src scope should be L1");
  Type dstType = this->getDst().getType();
  AddressSpace dstScope = getHIVMAddressSpace(dstType);
  assert(dstScope == AddressSpace::GM && "dst scope should be GM");
#endif
  // get dimensions
  MemRefType srcMemref = cast<MemRefType>(this->getSrcOperandType());
  std::string srcRankStr = std::to_string(srcMemref.getRank()) + "d";
  MemRefType dstMemref = cast<MemRefType>(this->getDstOperandType());
  std::string dstRankStr = std::to_string(dstMemref.getRank()) + "d";
  // get data type
  std::string dataTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));
  // make library function name
  return this->getOpName().str() + "_" + srcRankStr + "_to_" + dstRankStr +
         "_" + dataTypeStr;
}

//===----------------------------------------------------------------------===//
// L12UBOp
//===----------------------------------------------------------------------===//

std::string L12UBOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  // check address space
  Type srcType = this->getSrc().getType();
#ifndef NDEBUG
  AddressSpace srcScope = getHIVMAddressSpace(srcType);
  assert(srcScope == AddressSpace::L1 && "src scope should be L1");
  Type dstType = this->getDst().getType();
  AddressSpace dstScope = getHIVMAddressSpace(dstType);
  assert(dstScope == AddressSpace::UB && "dst scope should be UB");
#endif
  // get dimensions
  MemRefType srcMemref = cast<MemRefType>(this->getSrcOperandType());
  std::string srcRankStr = std::to_string(srcMemref.getRank()) + "d";
  MemRefType dstMemref = cast<MemRefType>(this->getDstOperandType());
  std::string dstRankStr = std::to_string(dstMemref.getRank()) + "d";
  // get data type
  std::string dataTypeStr =
      hivm::detail::getTypeName(this->getLoc(), getElementTypeOrSelf(srcType));
  // make library function name
  return this->getOpName().str() + "_" + srcRankStr + "_to_" + dstRankStr +
         "_" + dataTypeStr;
}

//===----------------------------------------------------------------------===//
// FixpipeOp
//===----------------------------------------------------------------------===//

void FixpipeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, Value dst,
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_dst_mode,
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split,
                      Value quant_scale) {
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond*/ ValueRange{},
        dma_mode, dual_dst_mode, pre_quant, pre_relu, channel_split,
        /*unit_flag_mode*/ ArrayAttr{}, quant_scale);
}

void FixpipeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      Type result, Value src, Value dst,
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_dst_mode,
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split,
                      Value quant_scale) {
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond*/ ValueRange{},
        dma_mode, dual_dst_mode, pre_quant, pre_relu, channel_split,
        /*unit_flag_mode*/ ArrayAttr{}, quant_scale);
}

void FixpipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  detail::getEffectsImpl(effects, cast<HIVMStructuredOp>(getOperation()));
}

std::string FixpipeOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  StringRef baseCallName = this->getOpName();
  // TODO, support 5HD, and other transform mode
  FixpipeDMAMode mode = getDmaMode();
  StringRef modeName = stringifyFixpipeDMAMode(mode);
  Type dstElemType = getElementTypeOrSelf(getDstOperandType());
  Type srcElemType = getElementTypeOrSelf(getSrcOperandType());
  int srcRank = getSrcOperandType().getRank();
  int dstRank = getDstOperandType().getRank();

  Type dstType = getDst().getType();
  std::string dstScopeName = "";
  if (auto dstScope = dyn_cast_if_present<AddressSpaceAttr>(
          dyn_cast<BaseMemRefType>(dstType).getMemorySpace())) {
    dstScopeName = kAddressSpace2LibraryName.at(dstScope.getAddressSpace());
  }
  std::string dualStr = "_";
  FixpipeDualDstModeAttr dualModeAttr = getDualDstModeAttr();
  if (dualModeAttr) {
    dualStr = "_dual_";
  }

  std::stringstream ss;
  ss << baseCallName.data() << "_" << modeName.data() << dualStr
     << hivm::detail::getTypeName(this->getLoc(), srcElemType) << "_to_"
     << hivm::detail::getTypeName(this->getLoc(), dstElemType) << "_" << srcRank
     << "d"
     << "_to_" << dstRank << "d_" << dstScopeName;
  return ss.str();
}

enum FixpipeState {
  Init = -1,
  QuantOrActivation = 0,
  End = 1,
};

int FixpipeOp::needFixpipePreFuse() { return FixpipeState::QuantOrActivation; }

bool FixpipeOp::hasStore() {
  Type inputType = getSrc().getType();
  if (!isa<TensorType>(inputType))
    return false;

  Type outputType = getDst().getType();
  return isa<MemRefType>(outputType);
}

int FixpipeOp::getFixpipeState() {
  bool hasStoreOrLayout = hasStore();
  if (hasStoreOrLayout) {
    return FixpipeState::End;
  }

  auto quant = this->getPreQuant();
  bool hasQuant = quant > FixpipePreQuantMode::NO_QUANT;

  auto activation = this->getPreRelu();
  bool hasActivation = activation > FixpipePreReluMode::NO_RELU;

  if (!hasQuant || !hasActivation) {
    return FixpipeState::QuantOrActivation;
  }

  return FixpipeState::Init;
}

inline static bool isDualDstEnabled(FixpipeDualDstMode dualDstMode) {
  return dualDstMode != FixpipeDualDstMode::NO_DUAL;
}

LogicalResult FixpipeOp::verify() {
  auto moduleOp = this->getOperation()->getParentOfType<mlir::ModuleOp>();
  bool isAscend950 = moduleOp && hacc::utils::isAscend950(moduleOp);
  auto dmaMode = getDmaMode();
  auto dstScope = getOptionalHIVMAddressSpace(getDst().getType());

  // NZ2DN DMA mode is only supported on Ascend950.
  if (dmaMode == FixpipeDMAMode::NZ2DN && !isAscend950) {
    return emitOpError("NZ2DN is only supported on Ascend950!");
  }
  // dst=UB is only supported on Ascend950.
  if (dstScope.has_value() && dstScope.value() == hivm::AddressSpace::UB &&
      !isAscend950) {
    return emitOpError("dst=UB is only supported on Ascend950!");
  }

  // check src and dst of dual_dst_mode
  auto dualDstModeAttr = getDualDstModeAttr();
  if (!dualDstModeAttr) {
    return success();
  }
  auto dualDstMode = dualDstModeAttr.getDualDstMode();
  if (!isDualDstEnabled(dualDstMode)) {
    return success();
  }

  // dual_dst_mode can only be enabled in NZ2ND/NZ2NZ and only on Ascend950.
  if (dmaMode == FixpipeDMAMode::NZ2DN) {
    return emitOpError("dual_dst_mode requires dma_mode to be NZ2ND or "
                       "NZ2NZ, but got NZ2DN!");
  }
  if (!isAscend950) {
    return emitOpError("dual_dst_mode is only supported on Ascend950!");
  }

  auto srcScope = getOptionalHIVMAddressSpace(getSrc().getType());
  bool hasScopeValue = srcScope.has_value() && dstScope.has_value();
  if (hasScopeValue &&
      (srcScope != hivm::AddressSpace::L0C || dstScope != hivm::AddressSpace::UB))
    return emitOpError("if dual_dst_mode is enabled, the data movement must "
                       "be performed from L0C to UB!");

  return success();
}
