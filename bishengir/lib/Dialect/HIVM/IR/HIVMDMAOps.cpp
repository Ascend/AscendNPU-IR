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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/Support/VersionTuple.h"

#include "bishengir/Config/bishengir-config.h"
#include "mlir/IR/ValueRange.h"

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
  Type srcType = srcOperType.getElementType();

  auto moduleOp =
      this->getOperation()->template getParentOfType<mlir::ModuleOp>();
  if (!hacc::utils::isAscend910_95(moduleOp) &&
      (llvm::isa<mlir::Float8E4M3FNType>(srcType) || llvm::isa<mlir::Float8E5M2Type>(srcType)))
    return emitOpError("Current hardware doesn't support fp8 type");

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
  Type srcType = srcOperType.getElementType();

  auto moduleOp =
      this->getOperation()->template getParentOfType<mlir::ModuleOp>();
  if (!hacc::utils::isAscend910_95(moduleOp) &&
      (llvm::isa<mlir::Float8E4M3FNType>(srcType) || llvm::isa<mlir::Float8E5M2Type>(srcType)))
    return emitOpError("Current hardware doesn't support fp8 type");
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
        {std::make_pair(AddressSpace::UB, AddressSpace::L1)},
    };
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (hacc::utils::isAscend910_95(moduleOp))
      kCopySupported.insert(
          {std::make_pair(AddressSpace::UB, AddressSpace::L1)});
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
        /*pad_value=*/nullptr, /*collapse_reassociation=*/nullptr);
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
  Type srcType = srcOperType.getElementType();

  auto moduleOp =
      this->getOperation()->template getParentOfType<mlir::ModuleOp>();
  if (!hacc::utils::isAscend910_95(moduleOp) &&
      (llvm::isa<mlir::Float8E4M3FNType>(srcType) || llvm::isa<mlir::Float8E5M2Type>(srcType)))
    return emitOpError("Current hardware doesn't support fp8 type");
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
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_mode,
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split,
                      Value quant_scale) {
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond=*/ ValueRange{},
        dma_mode, /*dual_dst_mode=*/dual_mode, pre_quant, pre_relu,
        channel_split,
        /*unit_flag_mode=*/ ArrayAttr{}, quant_scale);
}

void FixpipeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      Type result, Value src, Value dst,
                      FixpipeDMAModeAttr dma_mode,
                      FixpipeDualDstModeAttr dual_mode,
                      FixpipePreQuantModeAttr pre_quant,
                      FixpipePreReluModeAttr pre_relu, BoolAttr channel_split,
                      Value quant_scale) {
  build(odsBuilder, odsState, result, src, dst, /*unit_flag_cond=*/ ValueRange{},
        dma_mode, /*dual_dst_mode=*/dual_mode, pre_quant, pre_relu,
        channel_split,
        /*unit_flag_mode=*/ ArrayAttr{}, quant_scale);
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

void getElidedAttrs(FixpipeOp op,
                    llvm::SmallVector<llvm::StringRef, 2> &elidedAttrs) {
  elidedAttrs.push_back("unit_flag_mode");
  Builder odsBuilder(op.getContext());
  {
    Attribute attr = op.getDmaModeAttr();
    if (attr && (attr == FixpipeDMAModeAttr::get(odsBuilder.getContext(),
                                                 FixpipeDMAMode::NZ2NZ)))
      elidedAttrs.push_back("dma_mode");
  }
  {
    Attribute attr = op.getDualDstModeAttr();
    if (attr &&
        (attr == FixpipeDualDstModeAttr::get(odsBuilder.getContext(),
                                             FixpipeDualDstMode::NO_DUAL)))
      elidedAttrs.push_back("dual_dst_mode");
  }
  {
    Attribute attr = op.getPreQuantAttr();
    if (attr &&
        (attr == FixpipePreQuantModeAttr::get(odsBuilder.getContext(),
                                              FixpipePreQuantMode::NO_QUANT)))
      elidedAttrs.push_back("pre_quant");
  }
  {
    Attribute attr = op.getPreReluAttr();
    if (attr &&
        (attr == FixpipePreReluModeAttr::get(odsBuilder.getContext(),
                                             FixpipePreReluMode::NO_RELU)))
      elidedAttrs.push_back("pre_relu");
  }
  {
    Attribute attr = op.getChannelSplitAttr();
    if (attr && (attr == odsBuilder.getBoolAttr(false)))
      elidedAttrs.push_back("channel_split");
  }
}

/*
FixpipeOp assemblyFormat in HIVMC version < 0.2.0:
```
  attr-dict
  `ins` `(` $src `:` type($src) `)`
  `outs` `(` $dst `:` type($dst) `)`
  (`dual_dst_mode` `=` $dual_dst_mode^)?
  (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
  (`->` type($result_tensor)^)?
```
Attribute supported in HIVMC version < 0.2.0:
```
OptionalAttr<UnitAttr>:$enable_nz2nd
```
*/
static void printVersion0_1(FixpipeOp &op, OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("unit_flag_mode");
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getPreQuantAttr();
    if (attr &&
        (attr == ::mlir::hivm::FixpipePreQuantModeAttr::get(
                     odsBuilder.getContext(), FixpipePreQuantMode::NO_QUANT)))
      elidedAttrs.push_back("pre_quant");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getPreReluAttr();
    if (attr &&
        (attr == ::mlir::hivm::FixpipePreReluModeAttr::get(
                     odsBuilder.getContext(), FixpipePreReluMode::NO_RELU)))
      elidedAttrs.push_back("pre_relu");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getChannelSplitAttr();
    if (attr && (attr == odsBuilder.getBoolAttr(false)))
      elidedAttrs.push_back("channel_split");
  }

  // backward compatibility
  auto addInterleave = [&](auto idx, auto size) {
    if (static_cast<int64_t>(idx) < static_cast<int64_t>(size) - 1)
      _odsPrinter << ", ";
    return;
  };
  auto printFilteredAttrs = [&](auto filteredAttrs) {
    auto sizeFilteredAttrs =
        std::distance(filteredAttrs.begin(), filteredAttrs.end());
    _odsPrinter << " {";
    for (auto [idx, attr] : llvm::enumerate(filteredAttrs)) {
      if (attr.getName() == FixpipeDMAModeAttr::getMnemonic()) {
        _odsPrinter << "enable_nz2nd";
      } else {
        _odsPrinter << attr.getName().str();
        if (!isa<UnitAttr>(attr.getValue())) {
          _odsPrinter << " = ";
          _odsPrinter.printAttribute(attr.getValue());
        }
      }
      addInterleave(idx, sizeFilteredAttrs);
    }
    _odsPrinter << "}";
  };

  llvm::SmallDenseSet<StringRef> elidedAttrsSet(elidedAttrs.begin(),
                                                elidedAttrs.end());
  auto filteredAttrs =
      llvm::make_filter_range(op->getAttrs(), [&](NamedAttribute attr) {
        return !elidedAttrsSet.contains(attr.getName().strref());
      });
  if (!filteredAttrs.empty())
    printFilteredAttrs(filteredAttrs);

  _odsPrinter << ' ' << "ins";
  _odsPrinter << "(";
  _odsPrinter << op.getSrc();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = op.getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::ShapedType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "outs";
  _odsPrinter << "(";
  _odsPrinter << op.getDst();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = op.getDst().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::ShapedType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  if (op.getUnitFlagModeAttr()) {
    _odsPrinter << ' ' << "unit_flag";
    _odsPrinter << "[";
    _odsPrinter.printStrippedAttrOrType(op.getUnitFlagModeAttr());
    /*
    // TODO: latest AscendNPU-IR is not compatible with HIVMC 0.1.0
    // unit_flag_cond
    if (op.getUnitFlagCond()) {
      _odsPrinter << ","; _odsPrinter << ' ';
      if (::mlir::Value value = op.getUnitFlagCond())
      _odsPrinter << value;
    }
    */
    _odsPrinter << "]";
  }
  if (op.getResultTensor()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << (op.getResultTensor() ? ::llvm::ArrayRef<::mlir::Type>(
                                               op.getResultTensor().getType())
                                         : ::llvm::ArrayRef<::mlir::Type>());
  }
}

/*
FixpipeOp assemblyFormat in HIVMC version == 0.2.*
```
  attr-dict
  `ins` `(` $src `:` type($src) `)`
  `outs` `(` $dst `:` type($dst) `)`
  (`dual_dst_mode` `=` $dual_dst_mode^)?
  (`unit_flag_mode` `(` $unit_flag_mode^ `)` )?
  (`unit_flag_cond` `(` $unit_flag_cond^ `)` )?
  (`->` type($result_tensor)^)?
```
*/
static void printVersion0_2(FixpipeOp &op, OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("dual_dst_mode");
  elidedAttrs.push_back("unit_flag_mode");
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getDmaModeAttr();
    if (attr && (attr == ::mlir::hivm::FixpipeDMAModeAttr::get(
                             odsBuilder.getContext(), FixpipeDMAMode::NZ2NZ)))
      elidedAttrs.push_back("dma_mode");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getDualDstModeAttr();
    if (attr && (attr == ::mlir::hivm::FixpipeDualDstModeAttr::get(
                             odsBuilder.getContext(),
                             ::mlir::hivm::FixpipeDualDstMode::NO_DUAL)))
      elidedAttrs.push_back("dual_dst_mode");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getPreQuantAttr();
    if (attr &&
        (attr == ::mlir::hivm::FixpipePreQuantModeAttr::get(
                     odsBuilder.getContext(), FixpipePreQuantMode::NO_QUANT)))
      elidedAttrs.push_back("pre_quant");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getPreReluAttr();
    if (attr &&
        (attr == ::mlir::hivm::FixpipePreReluModeAttr::get(
                     odsBuilder.getContext(), FixpipePreReluMode::NO_RELU)))
      elidedAttrs.push_back("pre_relu");
  }
  {
    ::mlir::Builder odsBuilder(op.getContext());
    ::mlir::Attribute attr = op.getChannelSplitAttr();
    if (attr && (attr == odsBuilder.getBoolAttr(false)))
      elidedAttrs.push_back("channel_split");
  }
  _odsPrinter.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "ins";
  _odsPrinter << "(";
  _odsPrinter << op.getSrc();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = op.getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::ShapedType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "outs";
  _odsPrinter << "(";
  _odsPrinter << op.getDst();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = op.getDst().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::ShapedType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  if (op.getDualDstModeAttr()) {
    _odsPrinter << ' ' << "dual_dst_mode";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printStrippedAttrOrType(op.getDualDstModeAttr());
  }
  if (op.getUnitFlagModeAttr()) {
    _odsPrinter << ' ' << "unit_flag_mode";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(op.getUnitFlagModeAttr());
    _odsPrinter << ")";
  }
  if (!op.getUnitFlagCond().empty()) {
    _odsPrinter << ' ' << "unit_flag_cond";
    _odsPrinter << "(";
    _odsPrinter << op.getUnitFlagCond();
    _odsPrinter << ")";
  }
  if (op.getResultTensor()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << (op.getResultTensor() ? ::llvm::ArrayRef<::mlir::Type>(
                                               op.getResultTensor().getType())
                                         : ::llvm::ArrayRef<::mlir::Type>());
  }
}

void FixpipeOp::print(OpAsmPrinter &p) {
  auto moduleOp =
      this->getOperation()->template getParentOfType<mlir::ModuleOp>();
  auto compatiblePrint =
      moduleOp->getAttrOfType<BoolAttr>(hacc::HIVMCCompatiblePrintAttr::name);
  if (!compatiblePrint || !compatiblePrint.getValue()) {
    printVersion0_2(*this, p);
    return;
  }
  if (hacc::utils::getHIVMCVersion(moduleOp) < llvm::VersionTuple(0, 2, 0))
    printVersion0_1(*this, p);
  else
    printVersion0_2(*this, p);
}

/*
generated with the following assemblyFormat
```
  attr-dict
  `ins` `(` $src `:` type($src) `)`
  `outs` `(` $dst `:` type($dst) `)`
  (`dual_dst_mode` `=` $dual_dst_mode^)?
  (`unit_flag_mode` `(` $unit_flag_mode^ `)` )?
  (`unit_flag_cond` `(` $unit_flag_cond^ `)` )?
  (`->` type($result_tensor)^)?
```
Supported  `{enable_nz2nd}` for backward compatibility
TODO: Support `unit_flag_mode[$unit_flag_mode, $unit_flag_cond]`
*/
ParseResult FixpipeOp::parse(::mlir::OpAsmParser &parser,
                             ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(
      &srcRawOperand, 1);
  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(
      &dstRawOperand, 1);
  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::Type dstRawType{};
  ::llvm::ArrayRef<::mlir::Type> dstTypes(&dstRawType, 1);
  ::mlir::hivm::FixpipeDualDstModeAttr dual_dst_modeAttr;
  ::mlir::ArrayAttr unit_flag_modeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      unit_flag_condOperands;
  ::llvm::SMLoc unit_flag_condOperandsLoc;
  (void)unit_flag_condOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> result_tensorTypes;
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return ::mlir::failure();
    // backward compatibility support
    for (auto attr : result.attributes) {
      auto name = attr.getName();
      if (name == "enable_nz2nd") {
        result.attributes.erase(name);
        Builder odsBuilder(parser.getContext());
        result.attributes.append(
            FixpipeDMAModeAttr::getMnemonic(),
            FixpipeDMAModeAttr::get(odsBuilder.getContext(),
                                    FixpipeDMAMode::NZ2ND));
      }
    }
  }
  if (parser.parseKeyword("ins"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::ShapedType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseKeyword("outs"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::ShapedType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dstRawType = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("dual_dst_mode"))) {
    if (parser.parseEqual())
      return ::mlir::failure();

    if (parser.parseCustomAttributeWithFallback(dual_dst_modeAttr,
                                                ::mlir::Type{})) {
      return ::mlir::failure();
    }
    if (dual_dst_modeAttr)
      result.getOrAddProperties<FixpipeOp::Properties>().dual_dst_mode =
          dual_dst_modeAttr;
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("unit_flag_mode"))) {
    if (parser.parseLParen())
      return ::mlir::failure();

    if (parser.parseCustomAttributeWithFallback(
            unit_flag_modeAttr,
            parser.getBuilder().getType<::mlir::NoneType>())) {
      return ::mlir::failure();
    }
    if (unit_flag_modeAttr)
      result.getOrAddProperties<FixpipeOp::Properties>().unit_flag_mode =
          unit_flag_modeAttr;
    if (parser.parseRParen())
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("unit_flag_cond"))) {
    if (parser.parseLParen())
      return ::mlir::failure();

    unit_flag_condOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(unit_flag_condOperands))
      return ::mlir::failure();
    if (parser.parseRParen())
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {

    {
      ::mlir::Type optionalType;
      ::mlir::OptionalParseResult parseResult =
          parser.parseOptionalType(optionalType);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        result_tensorTypes.push_back(optionalType);
      }
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  result.addTypes(result_tensorTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(unit_flag_condOperands, odsBuildableType0,
                             unit_flag_condOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LogicalResult FixpipeOp::verify() {
  mlir::ModuleOp moduleOp = (*this)->getParentOfType<mlir::ModuleOp>();
  bool is910_95 = hacc::utils::isAscend910_95(moduleOp);

  // TODO: check that the nz2nd mode is enabled before the dual_dst_mode.
  // check src and dst of dual_dst_mode.
  auto dualDstModeAttr = getDualDstModeAttr();
  if (dualDstModeAttr) {
    auto dualDstMode = dualDstModeAttr.getDualDstMode();
    auto srcScope = getOptionalHIVMAddressSpace(getSrc().getType());
    auto dstScope = getOptionalHIVMAddressSpace(getDst().getType());
    bool hasScopeValue = srcScope.has_value() && dstScope.has_value();
    if (!is910_95 && dualDstMode != FixpipeDualDstMode::NO_DUAL) {
      return emitOpError(
          "The current hardware does not support dual_dst_mode is enabled");
    }
    if (is910_95 && hasScopeValue &&
        dualDstMode != FixpipeDualDstMode::NO_DUAL &&
        (srcScope != hivm::AddressSpace::L0C ||
         dstScope != hivm::AddressSpace::UB))
      return emitOpError("if dual_dst_mode is enabled, the data movement must "
                         "be performed from L0C to UB!");
  }

  if (!is910_95) {
    auto DMAModeAttr = getDmaMode();
    if (DMAModeAttr != FixpipeDMAMode::NZ2ND &&
        DMAModeAttr != FixpipeDMAMode::NZ2NZ)
      return emitOpError(
          "The current hardware does not support dma mode is nz2nz");

    if (getChannelSplit())
      return emitOpError("The current hardware does not channel split");
  }

  return success();
}

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
