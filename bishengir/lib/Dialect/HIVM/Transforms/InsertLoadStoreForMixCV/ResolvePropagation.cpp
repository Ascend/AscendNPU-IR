//===- ResolvePropagation.cpp --- Resolve conflict of scope propagation ---===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/ResolvePropagation.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "insert-load-store-resolve-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

static std::optional<hivm::AddressSpace>
getSinglePropagatedAddressSpace(UnrealizedConversionCastOp propagateOp) {
  auto addressSpaces = PropagatorUtil::getAddressSpace(propagateOp);
  if (addressSpaces.size() == 1)
    return addressSpaces[0];
  return std::nullopt;
}

static LogicalResult
resolveByInsertingStoreAndLoad(UnrealizedConversionCastOp downPropOp,
                               UnrealizedConversionCastOp upPropOp,
                               PatternRewriter &rewriter) {
  auto dstAddressSpace = getSinglePropagatedAddressSpace(upPropOp);
  auto [storeOp, loadOp] = PropagatorUtil::insertStoreAndLoad(
      downPropOp->getResult(0), downPropOp.getLoc(), rewriter, dstAddressSpace);
  auto isBufferized = isa<MemRefType>(loadOp.getDstOperandType());
  Value loadedValue;
  if (isBufferized) {
    loadedValue = loadOp.getDst();
  } else {
    loadedValue = loadOp.getResult(0);
  }
  // Rewire the up propagator to consume the value reloaded from GM.
  rewriter.modifyOpInPlace(
      upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(loadedValue); });
  // Recreate propagation constraints around the inserted store/load so address
  // space and core-type information can continue to propagate correctly.
  auto intermediateAddrSpace = dstAddressSpace.value_or(hivm::AddressSpace::GM);
  PropagatorUtil::createPropagatorUp(&storeOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&storeOp.getDstMutable(),
                                     intermediateAddrSpace, rewriter);
  PropagatorUtil::createPropagatorsDown(storeOp, intermediateAddrSpace,
                                        rewriter);
  PropagatorUtil::createPropagatorUp(&loadOp.getSrcMutable(),
                                     intermediateAddrSpace, rewriter);
  PropagatorUtil::createPropagatorUp(&loadOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(loadOp, upPropOp, rewriter);
  return success();
}

[[maybe_unused]] static LogicalResult
resolveLocalToLocal(UnrealizedConversionCastOp downPropOp,
                    UnrealizedConversionCastOp upPropOp,
                    PatternRewriter &rewriter) {
  auto downCoreType = PropagatorUtil::getCoreType(downPropOp);
  auto upCoreType = PropagatorUtil::getCoreType(upPropOp);
  if (downCoreType != TCoreType::CUBE_AND_VECTOR &&
      upCoreType != TCoreType::CUBE_AND_VECTOR) {
    return resolveByInsertingStoreAndLoad(downPropOp, upPropOp, rewriter);
  }
  LogicalResult result = failure();
  llvm::SmallDenseSet<hivm::AddressSpace, 2> mergedAddressSpace;
  auto downAddressSpaces = PropagatorUtil::getAddressSpace(downPropOp);
  auto upAddressSpaces = PropagatorUtil::getAddressSpace(upPropOp);
  mergedAddressSpace.insert(downAddressSpaces.begin(), downAddressSpaces.end());
  mergedAddressSpace.insert(upAddressSpaces.begin(), upAddressSpaces.end());
  auto newAddressAttr = rewriter.getAttr<ArrayAttr>(llvm::map_to_vector(
      mergedAddressSpace, [&rewriter](auto addressSpace) -> Attribute {
        return rewriter.getAttr<hivm::AddressSpaceAttr>(addressSpace);
      }));
  if (downCoreType == TCoreType::CUBE_AND_VECTOR &&
      downAddressSpaces.size() != newAddressAttr.size()) {
    rewriter.modifyOpInPlace(downPropOp, [&]() {
      downPropOp->setAttr(hivm::AddressSpaceAttr::name, newAddressAttr);
    });
    PropagatorUtil::createPropagatorUp(&downPropOp.getInputsMutable()[0],
                                       downPropOp, rewriter);
    result = success();
  }
  if (upCoreType == TCoreType::CUBE_AND_VECTOR &&
      upAddressSpaces.size() != newAddressAttr.size()) {
    rewriter.modifyOpInPlace(upPropOp, [&]() {
      upPropOp->setAttr(hivm::AddressSpaceAttr::name, newAddressAttr);
    });
    PropagatorUtil::createPropagatorDown(upPropOp->getResult(0), upPropOp,
                                         rewriter);
    result = success();
  }
  return result;
}

[[maybe_unused]] static LogicalResult
resolveGMtoLocal(UnrealizedConversionCastOp downPropOp,
                 UnrealizedConversionCastOp upPropOp,
                 PatternRewriter &rewriter) {
  auto loadOp = PropagatorUtil::insertLoad(downPropOp->getResult(0),
                                           downPropOp.getLoc(), rewriter);
  Value loadedValue;
  if (isa<RankedTensorType>(loadOp.getDstOperandType())) {
    loadedValue = loadOp.getResult(0);
  } else {
    loadedValue = loadOp.getDst();
  }
  rewriter.setInsertionPointAfter(loadOp);
  rewriter.modifyOpInPlace(
      upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(loadedValue); });
  PropagatorUtil::createPropagatorUp(&loadOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&loadOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(loadOp, upPropOp, rewriter);
  return success();
}

static LogicalResult resolveLocaltoGM(UnrealizedConversionCastOp downPropOp,
                                      UnrealizedConversionCastOp upPropOp,
                                      PatternRewriter &rewriter) {
  auto dstAddressSpace = getSinglePropagatedAddressSpace(upPropOp);
  auto storeOp = PropagatorUtil::insertStore(
      downPropOp->getResult(0), downPropOp.getLoc(), rewriter, dstAddressSpace);
  Value loadedValue;
  if (isa<RankedTensorType>(storeOp.getDstOperandType())) {
    loadedValue = storeOp.getResult(0);
  } else {
    loadedValue = storeOp.getDst();
  }
  rewriter.modifyOpInPlace(
      upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(loadedValue); });
  PropagatorUtil::createPropagatorUp(&storeOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&storeOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(storeOp, upPropOp, rewriter);
  return success();
}

LogicalResult TightCoupledBufferResolvePropagationPattern::resolveL0CToUB(
    UnrealizedConversionCastOp downPropOp, UnrealizedConversionCastOp upPropOp,
    PatternRewriter &rewriter) const {
  auto fixpipeOp = PropagatorUtil::insertFixpipe(downPropOp->getResult(0),
                                                 downPropOp.getLoc(), rewriter,
                                                 inferFixpipeDmaMode);
  if (isa<RankedTensorType>(fixpipeOp.getDstOperandType())) {
    Value loadedValue = fixpipeOp.getResult(0);
    rewriter.modifyOpInPlace(
        upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(loadedValue); });
  } else {
    llvm_unreachable("inserted fixpipe should return tensor type");
  }
  rewriter.setInsertionPointAfter(fixpipeOp);
  PropagatorUtil::createPropagatorUp(&fixpipeOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&fixpipeOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(fixpipeOp, upPropOp, rewriter);
  return success();
}

static LogicalResult ensureFixpipeToUB(UnrealizedConversionCastOp downPropOp,
                                       UnrealizedConversionCastOp upPropOp,
                                       PatternRewriter &rewriter) {
  // Insert load from L0C to rematerialize the value in UB
  // for fixpipe op.
  auto fixpipeOp = downPropOp.getInputs()[0].getDefiningOp<FixpipeOp>();
  if (!fixpipeOp)
    return failure();
  ArrayRef<int64_t> maybeStaticTotalSize =
      fixpipeOp.getDstOperandType().getShape();
  // Replicate InsertCVTightCoupledBuffer.cpp: Trace fixpipe src to Mmad and use
  // Mmad type as static shaped type
  if (!fixpipeOp.getDstOperandType().hasStaticShape()) {
    if (auto maybeMmadOp = traceDefOp<hivm::MmadL1Op>(fixpipeOp.getSrc())) {
      LDBG("traced to mmad op " << maybeMmadOp);
      mlir::Type mmadType =
          dyn_cast<MmadL1Op>(maybeMmadOp.value())->getResultTypes()[0];
      if (auto shapedMmadType = dyn_cast<ShapedType>(mmadType))
        maybeStaticTotalSize = shapedMmadType.getShape();
    }
  }
  auto [coupledBuffer, toTensorOp] =
      PropagatorUtil::insertTightCoupledBufferToUB(
          fixpipeOp.getDst(), downPropOp.getLoc(), rewriter,
          maybeStaticTotalSize);
  Value bufferizedValue = toTensorOp.getResult();
  // Create new fixpipe writing to UB memref (returns void)
  SmallVector<Value> oprs({fixpipeOp.getSrc(), coupledBuffer.spacedMemref});
  if (auto quantScale = fixpipeOp.getQuantScale())
    oprs.push_back(quantScale);
  rewriter.create<FixpipeOp>(downPropOp.getLoc(), TypeRange{}, oprs,
                             fixpipeOp->getAttrs());
  rewriter.replaceOp(fixpipeOp, toTensorOp.getResult());

  rewriter.modifyOpInPlace(
      upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(bufferizedValue); });
  PropagatorUtil::createPropagatorUp(&toTensorOp.getMemrefMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(toTensorOp, upPropOp, rewriter);
  return success();
}

[[maybe_unused]] static LogicalResult
resolveL1ToUB(UnrealizedConversionCastOp downPropOp,
              UnrealizedConversionCastOp upPropOp, PatternRewriter &rewriter) {
  Value srcValue = downPropOp->getResult(0);
  Location loc = downPropOp.getLoc();

  auto tensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!tensorType)
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  auto elemType = tensorType.getElementType();
  auto shape = tensorType.getShape();

  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto ubMemrefType = MemRefType::get(shape, elemType, nullptr, ubSpaceAttr);
  auto plainMemrefType = MemRefType::get(shape, elemType);

  rewriter.setInsertionPointAfter(downPropOp);
  Value alloc = rewriter.create<memref::AllocOp>(loc, ubMemrefType);
  Value noUb =
      rewriter.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, alloc);
  auto copyOp = rewriter.create<hivm::CopyOp>(loc, TypeRange{}, srcValue, noUb);

  auto toTensor = rewriter.create<bufferization::ToTensorOp>(loc, tensorType,
                                                             noUb, true, true);

  Value loadedValue = toTensor.getResult();
  rewriter.modifyOpInPlace(
      upPropOp, [&]() { upPropOp.getInputsMutable()[0].set(loadedValue); });

  PropagatorUtil::createPropagatorUp(&copyOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&copyOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(copyOp, upPropOp, rewriter);
  return success();
}

//===----------------------------------------------------------------------===//
// TightCoupledBufferResolvePropagationPattern
//===----------------------------------------------------------------------===//

LogicalResult TightCoupledBufferResolvePropagationPattern::resolveUBToL1(
    UnrealizedConversionCastOp downPropOp, UnrealizedConversionCastOp upPropOp,
    PatternRewriter &rewriter) const {
  Value value = downPropOp->getResult(0);
  auto emptyOp = PropagatorUtil::insertTensor(
      value, downPropOp.getLoc(), rewriter,
      cast<RankedTensorType>(value.getType()).getShape(), AddressSpace::L1);
  auto copyOp = rewriter.create<CopyOp>(downPropOp.getLoc(), emptyOp.getType(),
                                        value, emptyOp);
  copyOp->setAttr(hivm::kInsertedCopyAttr::name, rewriter.getUnitAttr());

  // Update operand to use the L1-buffered tensor
  rewriter.modifyOpInPlace(upPropOp, [&]() {
    upPropOp.getInputsMutable()[0].set(copyOp.getResultTensor());
  });
  PropagatorUtil::createPropagatorUp(&copyOp.getSrcMutable(), downPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorUp(&copyOp.getDstMutable(), upPropOp,
                                     rewriter);
  PropagatorUtil::createPropagatorsDown(copyOp, upPropOp, rewriter);
  return success();
}

LogicalResult TightCoupledBufferResolvePropagationPattern::matchAndRewrite(
    UnrealizedConversionCastOp upPropOp, PatternRewriter &rewriter) const {
  if (!upPropOp->hasAttr(kPropagateUpAttr))
    return failure();
  auto downPropOp =
      upPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!downPropOp || !downPropOp->hasAttr(kPropagateDownAttr))
    return failure();
  auto [downCoreType, downAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(downPropOp);
  auto [upCoreType, upAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(upPropOp);
  rewriter.setInsertionPointAfter(downPropOp);
  if (llvm::find(downAddressSpace, hivm::AddressSpace::L0C) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::UB) !=
          upAddressSpace.end()) {
    LDBG("Resolving CC to UB: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveL0CToUB(downPropOp, upPropOp, rewriter);
  }
  if (downCoreType == TCoreType::CUBE_AND_VECTOR &&
      llvm::find(upAddressSpace, hivm::AddressSpace::UB) !=
          upAddressSpace.end()) {
    LDBG("Ensuring Fixpipe to UB: " << downPropOp << "\n" << upPropOp << "\n");
    return ensureFixpipeToUB(downPropOp, upPropOp, rewriter);
  }
  if (downCoreType != TCoreType::CUBE_AND_VECTOR &&
      upCoreType != TCoreType::CUBE_AND_VECTOR &&
      llvm::find(downAddressSpace, hivm::AddressSpace::UB) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::L1) !=
          upAddressSpace.end()) {
    LDBG("Resolving UB to L1: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveUBToL1(downPropOp, upPropOp, rewriter);
  }
  if (downCoreType != TCoreType::CUBE_AND_VECTOR &&
      upCoreType != TCoreType::CUBE_AND_VECTOR &&
      llvm::find(downAddressSpace, hivm::AddressSpace::L1) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::UB) !=
          upAddressSpace.end()) {
    LDBG("Resolving L1 to UB: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveL1ToUB(downPropOp, upPropOp, rewriter);
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ResolvePropagationPattern
//===----------------------------------------------------------------------===//

LogicalResult
ResolvePropagationPattern::matchAndRewrite(UnrealizedConversionCastOp upPropOp,
                                           PatternRewriter &rewriter) const {
  if (!upPropOp->hasAttr(kPropagateUpAttr))
    return failure();
  auto downPropOp =
      upPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!downPropOp || !downPropOp->hasAttr(kPropagateDownAttr))
    return failure();
  auto [downCoreType, downAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(downPropOp);
  auto [upCoreType, upAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(upPropOp);
  rewriter.setInsertionPointAfter(downPropOp);
  if (llvm::find(downAddressSpace, hivm::AddressSpace::GM) !=
          downAddressSpace.end() &&
      upCoreType == TCoreType::CUBE_AND_VECTOR) {
    LDBG("Resolving GM to Local: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveGMtoLocal(downPropOp, upPropOp, rewriter);
  }
  if (downCoreType == TCoreType::CUBE_AND_VECTOR &&
      llvm::find(upAddressSpace, hivm::AddressSpace::GM) !=
          upAddressSpace.end()) {
    LDBG("Resolving Local to GM: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveLocaltoGM(downPropOp, upPropOp, rewriter);
  }
  if (upCoreType == TCoreType::CUBE_AND_VECTOR ||
      downCoreType == TCoreType::CUBE_AND_VECTOR) {
    LDBG("Resolving Local to Local: " << downPropOp << "\n"
                                      << upPropOp << "\n");
    return resolveLocalToLocal(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::UB) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::L1) !=
          upAddressSpace.end()) {
    LDBG("Resolving UB to L1: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveLocalToLocal(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::L1) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::UB) !=
          upAddressSpace.end()) {
    LDBG("Resolving L1 to UB: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveLocalToLocal(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::GM) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::UB) !=
          upAddressSpace.end()) {
    LDBG("Resolving GM to UB: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveGMtoLocal(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::GM) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::L1) !=
          upAddressSpace.end()) {
    LDBG("Resolving GM to L1: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveGMtoLocal(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::UB) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::GM) !=
          upAddressSpace.end()) {
    LDBG("Resolving UB to GM: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveLocaltoGM(downPropOp, upPropOp, rewriter);
  }
  if (llvm::find(downAddressSpace, hivm::AddressSpace::L1) !=
          downAddressSpace.end() &&
      llvm::find(upAddressSpace, hivm::AddressSpace::GM) !=
          upAddressSpace.end()) {
    LDBG("Resolving L1 to GM: " << downPropOp << "\n" << upPropOp << "\n");
    return resolveLocaltoGM(downPropOp, upPropOp, rewriter);
  }
  return failure();
}

LogicalResult RemoveRedundantPropagationPattern::matchAndRewrite(
    UnrealizedConversionCastOp upPropOp, PatternRewriter &rewriter) const {
  if (!upPropOp->hasAttr(kPropagateUpAttr))
    return failure();
  auto midDownPropOp =
      upPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!midDownPropOp || !midDownPropOp->hasAttr(kPropagateDownAttr))
    return failure();
  auto midUpPropOp =
      midDownPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!midUpPropOp || !midUpPropOp->hasAttr(kPropagateUpAttr) ||
      !midUpPropOp->hasOneUse())
    return failure();
  auto downPropOp =
      midUpPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!downPropOp || !downPropOp->hasAttr(kPropagateDownAttr))
    return failure();
  LDBG("Removing redundant propagation: " << downPropOp << "\n"
                                          << midUpPropOp << "\n"
                                          << midDownPropOp << "\n"
                                          << upPropOp);
  rewriter.replaceOp(midDownPropOp, downPropOp);
  rewriter.eraseOp(midUpPropOp);
  return success();
}

/// Resolve multi address propagator
///  down1(CUBE) down2(VECTOR)
///         \          /
///  down(cubeOperand, vectorOperand)
///        /            \
///    up1(CUBE)     up2(VECTOR)
///
/// to
///
///  down1(CUBE)     down2(VECTOR)
///       |                |
///   up1(CUBE)       up2(VECTOR)
static LogicalResult
resolveMultipleAddressSpacePropagator(UnrealizedConversionCastOp op,
                                      PatternRewriter &rewriter) {
  if (!op->hasAttr(kPropagateDownAttr))
    return failure();
  if (op->getNumOperands() == 1u)
    return failure();
  llvm::SmallDenseMap<hivm::AddressSpace, Value, 2> addressSpaceToOperandMap;
  for (auto operand : op.getInputs()) {
    auto downPropOp = operand.getDefiningOp<UnrealizedConversionCastOp>();
    if (!downPropOp || !downPropOp->hasAttr(kPropagateDownAttr))
      return rewriter.notifyMatchFailure(
          op, "Invalid case for resolving multi address "
              "propagator: incorrect structure");
    auto addressSpaces = PropagatorUtil::getAddressSpace(downPropOp);
    if (addressSpaces.size() != 1u)
      return rewriter.notifyMatchFailure(
          op, "Invalid case for resolving multi address "
              "propagator: incorrect structure");
    addressSpaceToOperandMap[addressSpaces[0]] = operand;
  }

  LogicalResult result = failure();
  SmallVector<OpOperand *> uses;
  for (auto &use : op->getUses())
    uses.push_back(&use);
  for (auto *use : uses) {
    auto upPropOp = dyn_cast<UnrealizedConversionCastOp>(use->getOwner());
    if (!upPropOp || !upPropOp->hasAttr(kPropagateUpAttr))
      return rewriter.notifyMatchFailure(
          op, "Invalid case for resolving multi address "
              "propagator: incorrect structure");
    auto addressSpaces = PropagatorUtil::getAddressSpace(upPropOp);
    if (addressSpaces.size() != 1u)
      continue;
    auto it = addressSpaceToOperandMap.find(addressSpaces[0]);
    if (it != addressSpaceToOperandMap.end()) {
      rewriter.modifyOpInPlace(upPropOp, [&]() { use->set(it->second); });
      result = success();
    }
  }
  if (op->use_empty()) {
    rewriter.eraseOp(op);
    result = success();
  }
  return result;
}

/// Resolve multi address operation
///
///          up
///          |
///          op
///          |
///         down
///
///  to
///
///      up1    up2
///       |      |
///      op1    op2
///       |      |
///     down1  down2
///       \      /
///         down
static LogicalResult
resolveMultipleAddressSpaceOperation(Operation *op, PatternRewriter &rewriter) {
  llvm::SmallDenseMap<hivm::AddressSpace, Operation *, 2> addressSpaceMap;
  for (auto &operand : op->getOpOperands()) {
    if (auto upProp = PropagatorUtil::getUpPropagator(&operand)) {
      auto [coreType, addressSpaces] =
          PropagatorUtil::extractPropagatorInfo(upProp);
      if (coreType != TCoreType::CUBE_AND_VECTOR || addressSpaces.size() <= 1u)
        continue;
      for (auto addressSpace : addressSpaces) {
        if (!addressSpaceMap.contains(addressSpace)) {
          addressSpaceMap[addressSpace] = rewriter.clone(*op);
        }
      }
    }
  }
  for (auto res : op->getResults()) {
    auto downProp = PropagatorUtil::getDownPropagator(res);
    if (!downProp)
      return rewriter.notifyMatchFailure(
          op, "Invalid case for resolving multi address "
              "propagator: cannot detect propagator");
    auto [coreType, addressSpaces] =
        PropagatorUtil::extractPropagatorInfo(downProp);
    if (coreType != TCoreType::CUBE_AND_VECTOR || addressSpaces.size() <= 1u)
      return rewriter.notifyMatchFailure(
          op, "Invalid case for resolving multi address "
              "propagator: cannot detect propagator");
    for (auto addressSpace : addressSpaces) {
      if (!addressSpaceMap.contains(addressSpace)) {
        addressSpaceMap[addressSpace] = rewriter.clone(*op);
      }
    }
  }
  if (addressSpaceMap.empty())
    return failure();
  SmallVector<Operation *> toErase = {op};
  for (auto &operand : op->getOpOperands()) {
    if (auto upProp = PropagatorUtil::getUpPropagator(&operand)) {
      auto [coreType, addressSpaces] =
          PropagatorUtil::extractPropagatorInfo(upProp);
      toErase.push_back(upProp);
      if (coreType != TCoreType::CUBE_AND_VECTOR ||
          addressSpaces.size() <= 1u) {
        for (auto [addressSpace, newOp] : addressSpaceMap) {
          rewriter.setInsertionPoint(newOp);
          auto newUpProp = rewriter.clone(*upProp)->getResult(0);
          rewriter.modifyOpInPlace(newOp, [&, newOp = newOp]() {
            newOp->getOpOperand(operand.getOperandNumber()).set(newUpProp);
          });
        }
        continue;
      }
      auto input = upProp.getInputs()[0];
      for (auto [addressSpace, newOp] : addressSpaceMap) {
        rewriter.setInsertionPoint(newOp);
        auto newUpProp = PropagatorUtil::createPropagator(
            input, kPropagateUpAttr,
            PropagatorUtil::kAddressSpace2CoreType.at(addressSpace),
            addressSpace, rewriter);
        rewriter.modifyOpInPlace(newOp, [&, newOp = newOp]() {
          newOp->getOpOperand(operand.getOperandNumber()).set(newUpProp);
        });
      }
    }
  }
  for (auto res : op->getResults()) {
    if (auto downProp = PropagatorUtil::getDownPropagator(res)) {
      auto [coreType, addressSpaces] =
          PropagatorUtil::extractPropagatorInfo(downProp);
      SmallVector<Value> newDownOperands;
      for (auto [addressSpace, newOp] : addressSpaceMap) {
        rewriter.setInsertionPointAfter(newOp);
        auto newDownProp = PropagatorUtil::createPropagator(
            newOp->getResult(res.getResultNumber()), kPropagateDownAttr,
            PropagatorUtil::kAddressSpace2CoreType.at(addressSpace),
            addressSpace, rewriter);
        newDownOperands.push_back(newDownProp);
      }
      rewriter.setInsertionPoint(downProp);
      auto newDownProp = rewriter.create<UnrealizedConversionCastOp>(
          downProp.getLoc(), downProp->getResult(0).getType(), newDownOperands);
      newDownProp->setAttrs(downProp->getAttrs());
      rewriter.replaceOp(downProp, newDownProp);
    }
  }
  for (auto *op : toErase)
    rewriter.eraseOp(op);
  return success();
}

LogicalResult CloneMultipleAddressSpaceOperation::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  if (auto propagateOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    auto [coreType, addressSpaces] =
        PropagatorUtil::extractPropagatorInfo(propagateOp);
    if (coreType != TCoreType::CUBE_AND_VECTOR || addressSpaces.size() != 2u)
      return failure();
    return resolveMultipleAddressSpacePropagator(propagateOp, rewriter);
  }
  return resolveMultipleAddressSpaceOperation(op, rewriter);
}

} // namespace mlir::hivm
