//===- HIVMDMAOps.cpp - HIVM DMA ops implementation -----------------------===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "bishengir/Config/bishengir-config.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;

#define ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(OP_NAME)                \
  Value OP_NAME::getSource() { return getSrc(); }                              \
  Value OP_NAME::getTarget() { return getDst(); }

ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(CopyOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(LoadOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(StoreOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(FixpipeOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(ND2NZOp)
ENABLE_DEFAULT_COPYOP_INTERFACE_IMPLEMENTATION(NZ2NDOp)
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
        /*init_out_buffer=*/false, /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst,
                   Value left_padding_num) {
  build(odsBuilder, odsState, res, src, dst, /*pad_mode=*/nullptr,
        /*pad_value=*/nullptr, left_padding_num,
        /*right_padding_num=*/nullptr, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        /*left_padding_num=*/nullptr,
        /*right_padding_num=*/nullptr, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr,
        /*init_out_buffer=*/false, /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   bool init_out_buffer) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr, init_out_buffer,
        /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   Value right_padding_num) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, right_padding_num, /*init_out_buffer=*/false,
        /*init_condition=*/nullptr);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   bool init_out_buffer, Value init_condition) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr, init_out_buffer,
        init_condition);
}

void LoadOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst, PadModeAttr pad_mode,
                   Value pad_value, Value left_padding_num,
                   bool init_out_buffer,
                   bool may_implicit_transpose_with_last_axis) {
  build(odsBuilder, odsState, res, src, dst, pad_mode, pad_value,
        left_padding_num, /*right_padding_num=*/nullptr, init_out_buffer,
        nullptr, may_implicit_transpose_with_last_axis);
}

static LogicalResult verifyPadMode(LoadOp op) {
  ShapedType srcOperType = op.getSrcOperandType();
  ShapedType dstOperType = op.getDstOperandType();
  auto srcShape = srcOperType.getShape();
  auto dstShape = dstOperType.getShape();

  auto padModeAttr = op.getPadMode();
  if (!padModeAttr || padModeAttr->getPadmode() == PadMode::PadNull) {
    // if not set padmode, means dst/src shape is the same
    if (failed(verifyCompatibleShape(srcShape, dstShape))) {
      return op.emitOpError(
          "if pad_mode is not set, src and dst shape should be the same!");
    }

    if (op.getPadValue()) {
      return op.emitOpError(
          "if pad_mode is not set, pad_value should not be set!");
    }
  }

  auto padval = op.getPadValue();
  if (padModeAttr) {
    // check pad value
    PadMode pm = padModeAttr->getPadmode();
    if (pm == PadMode::PadValue && !padval) {
      return op.emitOpError("if padmode is PadValue, pad_value is required!");
    }

    if (pm == PadMode::PadFirstElem && padval) {
      return op.emitOpError(
          "if padmode is PadFirstElem, pad_value should not be set!");
    }

    if (failed(
            verifyCompatibleShape(srcShape.drop_back(), dstShape.drop_back())))
      return op.emitOpError() << "only the last dimension can be different";

    if (!ShapedType::isDynamic(srcShape.back()) &&
        !ShapedType::isDynamic(dstShape.back()) &&
        dstShape.back() < srcShape.back())
      return op.emitOpError() << "dst last dimension cannot be less than src's";
  }

  // check padval dtype
  if (padval && padval.getType() != dstOperType.getElementType()) {
    return op.emitOpError(
        "dtype of pad_value and element type of dst/src should be the same!");
  }

  return success();
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

  if (srcOperType.getRank() != dstOperType.getRank()) {
    return emitOpError("src and dst should have the same dimensions!");
  }

  if (failed(verifyPadMode(*this)))
    return failure();

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
      return op.emitOpError("only support store ub to gm currently!");
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

  if (srcOperType.getRank() != dstOperType.getRank() ||
      failed(verifyCompatibleShape(srcOperType.getShape(),
                                   dstOperType.getShape()))) {
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
        {std::make_pair(AddressSpace::GM, AddressSpace::UB)},
        {std::make_pair(AddressSpace::UB, AddressSpace::GM)},
        {std::make_pair(AddressSpace::UB, AddressSpace::UB)},
        {std::make_pair(AddressSpace::GM, AddressSpace::L1)},
    };
#if BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
    kCopySupported.insert({std::make_pair(AddressSpace::UB, AddressSpace::L1)});
#endif
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

void CopyOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange res, Value src, Value dst) {
  build(odsBuilder, odsState, res, src, dst, /*pad_mode=*/nullptr,
        /*pad_value=*/nullptr, /*collapse_reassociation=*/nullptr);
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

  return emitOpError("dst/src should be memref/memref or tensor/tensor, res "
                     "should be tensor!");
}

SmallVector<ReassociationIndices, 4>
CopyOp::getReassociationIndices(bool isCollapse) {
  if (!isCollapse)
    llvm_unreachable("Unsupported");

  SmallVector<ReassociationIndices, 4> reassociationIndices;
  auto collapseReassociation = getCollapseReassociation();
  if (!collapseReassociation.has_value()) {
    return reassociationIndices;
  }
  for (auto attr : collapseReassociation.value())
    reassociationIndices.push_back(llvm::to_vector<2>(
        llvm::map_range(cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
          return cast<IntegerAttr>(indexAttr).getInt();
        })));
  return reassociationIndices;
}

SmallVector<AffineMap, 4> CopyOp::getReassociationMaps(bool isCollapse) {
  if (!isCollapse)
    llvm_unreachable("Unsupported");

  return getSymbolLessAffineMaps(getReassociationExprs(isCollapse));
}

SmallVector<ReassociationExprs, 4>
CopyOp::getReassociationExprs(bool isCollapse) {
  if (!isCollapse)
    llvm_unreachable("Unsupported");

  return convertReassociationIndicesToExprs(
      getContext(), getReassociationIndices(isCollapse));
}

//===----------------------------------------------------------------------===//
// ND2NZOp
//===----------------------------------------------------------------------===//

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
// FixpipeOp
//===----------------------------------------------------------------------===//

void FixpipeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, Value dst,
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
                      UnitAttr enable_nz2nd,
#else
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_mode,
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split) {
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond*/ Value{},
        enable_nz2nd, pre_quant, pre_relu, channel_split,
        /*unit_flag_mode*/ UnitFlagAttr{});
#else
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond=*/nullptr,
        dma_mode, /*dual_dst_mode=*/dual_mode, pre_quant, pre_relu,
        channel_split,
        /*unit_flag_mode=*/nullptr);
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
}

void FixpipeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      Type result, Value src, Value dst,
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
                      UnitAttr enable_nz2nd,
#else
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_mode,
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split) {
#if (!BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES)
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond=*/nullptr,
        enable_nz2nd, pre_quant, pre_relu, channel_split,
        /*unit_flag_mode=*/nullptr);
#else
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond=*/nullptr,
        dma_mode, /*dual_dst_mode=*/dual_mode, pre_quant, pre_relu,
        channel_split,
        /*unit_flag_mode=*/nullptr);
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
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

#if BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES
LogicalResult FixpipeOp::verify() {
  // TODO: check that the nz2nd mode is enabled before the dual_dst_mode.
  // check src and dst of dual_dst_mode.
  auto dualDstModeAttr = getDualDstModeAttr();
  if (dualDstModeAttr) {
    auto dualDstMode = dualDstModeAttr.getDualDstMode();
    auto srcScope = getOptionalHIVMAddressSpace(getSrc().getType());
    auto dstScope = getOptionalHIVMAddressSpace(getDst().getType());
    bool hasScopeValue = srcScope.has_value() && dstScope.has_value();
    if (hasScopeValue && dualDstMode != FixpipeDualDstMode::NO_DUAL &&
        (srcScope != hivm::AddressSpace::L0C ||
         dstScope != hivm::AddressSpace::UB))
      return emitOpError("if dual_dst_mode is enabled, the data movement must "
                         "be performed from L0C to UB!");
  }

  return success();
}
#endif // BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES

//===----------------------------------------------------------------------===//
// AtomicCasOp
//===----------------------------------------------------------------------===//
void AtomicCasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto &op : getSrcMutable()) {
    if (!isa<MemRefType>(op.get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), &op,
                         SideEffects::DefaultResource::get());
  }
  if (isa<MemRefType>(getDst().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDstMutable(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                         SideEffects::DefaultResource::get());
  }
}
