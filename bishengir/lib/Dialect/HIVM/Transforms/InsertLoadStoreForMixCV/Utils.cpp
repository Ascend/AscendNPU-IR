//===---- Utils.cpp ---- utility of Insert Load Store for Mix CV ----------===//
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

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/TypeSwitch.h"

#include <utility>

#define DEBUG_TYPE "insert-load-store-util"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

namespace PropagatorUtil {

std::pair<TCoreType, SmallVector<hivm::AddressSpace, 2>>
extractPropagatorInfo(UnrealizedConversionCastOp propagateOp) {
  auto coreType = TCoreType::CUBE_OR_VECTOR;
  if (auto coreTypeAttr = propagateOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name))
    coreType = coreTypeAttr.getTcoretype();
  SmallVector<hivm::AddressSpace, 2> addressSpaces;
  if (auto addressSpaceAttr =
          propagateOp->getAttrOfType<ArrayAttr>(hivm::AddressSpaceAttr::name)) {
    addressSpaces = llvm::map_to_vector(addressSpaceAttr, [](auto attr) {
      return cast<hivm::AddressSpaceAttr>(attr).getAddressSpace();
    });
  }
  return std::make_pair(coreType, addressSpaces);
}

static void updatePropagator(Operation *uccOp, StringRef directionAttrName,
                             TCoreType coreType,
                             ArrayRef<hivm::AddressSpace> addressSpaces,
                             OpBuilder &builder) {
  if (coreType != TCoreType::CUBE_OR_VECTOR) {
    uccOp->setAttr(hivm::TCoreTypeAttr::name,
                   builder.getAttr<hivm::TCoreTypeAttr>(coreType));
  }
  SmallVector<Attribute> addressSpaceAttrs;
  for (auto addressSpace : addressSpaces) {
    addressSpaceAttrs.push_back(
        builder.getAttr<hivm::AddressSpaceAttr>(addressSpace));
  }
  uccOp->setAttr(hivm::AddressSpaceAttr::name,
                 builder.getAttr<ArrayAttr>(addressSpaceAttrs));
  uccOp->setAttr(directionAttrName, builder.getUnitAttr());
}

Value createPropagator(Value v, StringRef directionAttrName, TCoreType coreType,
                       ArrayRef<hivm::AddressSpace> addressSpaces,
                       OpBuilder &builder) {
  if (!isa<ShapedType>(v.getType()))
    return v;
  auto uccOp =
      builder.create<UnrealizedConversionCastOp>(v.getLoc(), v.getType(), v);
  updatePropagator(uccOp, directionAttrName, coreType, addressSpaces, builder);
  return uccOp->getResult(0);
}

Value createPropagator(Value v, StringRef directionAttrName, TCoreType coreType,
                       OpBuilder &builder) {
  return createPropagator(v, directionAttrName, coreType, {}, builder);
}

Value createPropagator(Value v, StringRef directionAttrName,
                       ArrayRef<hivm::AddressSpace> addressSpaces,
                       OpBuilder &builder) {
  return createPropagator(v, directionAttrName, TCoreType::CUBE_OR_VECTOR,
                          addressSpaces, builder);
}

Value createPropagator(Value v, StringRef directionAttrName,
                       UnrealizedConversionCastOp propagateOp,
                       OpBuilder &builder) {
  auto [coreType, addressSpaces] = extractPropagatorInfo(propagateOp);
  return createPropagator(v, directionAttrName, coreType, addressSpaces,
                          builder);
}

void createPropagatorUp(OpOperand *operand, TCoreType coreType,
                        ArrayRef<hivm::AddressSpace> addressSpaces,
                        PatternRewriter &rewriter) {
  auto *op = operand->getOwner();
  if (op->hasAttr(kPropagateUpAttr)) {
    updatePropagator(op, kPropagateUpAttr, coreType, addressSpaces, rewriter);
    return;
  }
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.modifyOpInPlace(op, [&]() {
    auto newOperand = createPropagator(operand->get(), kPropagateUpAttr,
                                       coreType, addressSpaces, rewriter);
    operand->set(newOperand);
  });
}

void createPropagatorUp(OpOperand *operand, TCoreType coreType,
                        PatternRewriter &rewriter) {
  createPropagatorUp(operand, coreType, {}, rewriter);
}

void createPropagatorUp(OpOperand *operand,
                        ArrayRef<hivm::AddressSpace> addressSpaces,
                        PatternRewriter &rewriter) {
  createPropagatorUp(operand, TCoreType::CUBE_OR_VECTOR, addressSpaces,
                     rewriter);
}

void createPropagatorUp(OpOperand *operand,
                        UnrealizedConversionCastOp propagateOp,
                        PatternRewriter &rewriter) {
  auto [coreType, addressSpace] = extractPropagatorInfo(propagateOp);
  createPropagatorUp(operand, coreType, addressSpace, rewriter);
}

void createPropagatorDown(Value res, TCoreType coreType,
                          ArrayRef<hivm::AddressSpace> addressSpaces,
                          PatternRewriter &rewriter) {
  if (auto *defOp = res.getDefiningOp();
      (defOp && defOp->hasAttr(kPropagateDownAttr))) {
    updatePropagator(defOp, kPropagateDownAttr, coreType, addressSpaces,
                     rewriter);
    return;
  }
  if (res.hasOneUse() && res.user_begin()->hasAttr(kPropagateDownAttr)) {
    updatePropagator(*res.user_begin(), kPropagateDownAttr, coreType,
                     addressSpaces, rewriter);
    return;
  }
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(res);
  auto newRes = createPropagator(res, kPropagateDownAttr, coreType,
                                 addressSpaces, rewriter);
  rewriter.replaceAllUsesExcept(res, newRes, newRes.getDefiningOp());
}

void createPropagatorDown(Value res, TCoreType coreType,
                          PatternRewriter &rewriter) {
  createPropagatorDown(res, coreType, {}, rewriter);
}

void createPropagatorDown(Value res, ArrayRef<hivm::AddressSpace> addressSpaces,
                          PatternRewriter &rewriter) {
  createPropagatorDown(res, TCoreType::CUBE_OR_VECTOR, addressSpaces, rewriter);
}

void createPropagatorDown(Value res, UnrealizedConversionCastOp propagateOp,
                          PatternRewriter &rewriter) {
  auto [coreType, addressSpaces] = extractPropagatorInfo(propagateOp);
  createPropagatorDown(res, coreType, addressSpaces, rewriter);
}

void createPropagatorsUp(Operation *op, TCoreType coreType,
                         ArrayRef<hivm::AddressSpace> addressSpaces,
                         PatternRewriter &rewriter) {
  for (auto &operand : op->getOpOperands())
    createPropagatorUp(&operand, coreType, addressSpaces, rewriter);
}

void createPropagatorsUp(Operation *op, TCoreType coreType,
                         PatternRewriter &rewriter) {
  createPropagatorsUp(op, coreType, {}, rewriter);
}

void createPropagatorsUp(Operation *op,
                         ArrayRef<hivm::AddressSpace> addressSpaces,
                         PatternRewriter &rewriter) {
  createPropagatorsUp(op, TCoreType::CUBE_OR_VECTOR, addressSpaces, rewriter);
}

void createPropagatorsUp(Operation *op, UnrealizedConversionCastOp propagateOp,
                         PatternRewriter &rewriter) {
  auto [coreType, addressSpaces] = extractPropagatorInfo(propagateOp);
  createPropagatorsUp(op, coreType, addressSpaces, rewriter);
}

void createPropagatorsDown(Operation *op, TCoreType coreType,
                           ArrayRef<hivm::AddressSpace> addressSpaces,
                           PatternRewriter &rewriter) {
  for (auto res : op->getResults())
    createPropagatorDown(res, coreType, addressSpaces, rewriter);
}

void createPropagatorsDown(Operation *op, TCoreType coreType,
                           PatternRewriter &rewriter) {
  createPropagatorsDown(op, coreType, {}, rewriter);
}

void createPropagatorsDown(Operation *op,
                           ArrayRef<hivm::AddressSpace> addressSpaces,
                           PatternRewriter &rewriter) {
  createPropagatorsDown(op, TCoreType::CUBE_OR_VECTOR, addressSpaces, rewriter);
}

void createPropagatorsDown(Operation *op,
                           UnrealizedConversionCastOp propagateOp,
                           PatternRewriter &rewriter) {
  auto [coreType, addressSpace] = extractPropagatorInfo(propagateOp);
  createPropagatorsDown(op, coreType, addressSpace, rewriter);
}

TCoreType getCoreType(UnrealizedConversionCastOp op) {
  auto coreTypeAttr =
      op->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr)
    return TCoreType::CUBE_OR_VECTOR;
  return coreTypeAttr.getTcoretype();
}

SmallVector<hivm::AddressSpace, 2>
getAddressSpace(UnrealizedConversionCastOp op) {
  auto addressSpaceAttr =
      op->getAttrOfType<ArrayAttr>(hivm::AddressSpaceAttr::name);
  if (!addressSpaceAttr)
    return {};
  return llvm::map_to_vector(addressSpaceAttr, [](auto attr) {
    return cast<hivm::AddressSpaceAttr>(attr).getAddressSpace();
  });
}

UnrealizedConversionCastOp getUpPropagator(OpOperand *operand) {
  auto defOp = operand->get().getDefiningOp<UnrealizedConversionCastOp>();
  if (defOp && defOp->hasAttr(kPropagateUpAttr))
    return defOp;
  return nullptr;
}

UnrealizedConversionCastOp getDownPropagator(OpResult res) {
  if (!res.hasOneUse())
    return nullptr;
  auto uccOp = dyn_cast<UnrealizedConversionCastOp>(*res.user_begin());
  if (uccOp && uccOp->hasAttr(kPropagateDownAttr))
    return uccOp;
  return nullptr;
}

hivm::StoreOp insertStore(Value value, Location loc,
                          PatternRewriter &rewriter) {
  Type type = value.getType();
  auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType) {
    Value storeInit = utils::createEmptyOp(rewriter, loc, value);
    return rewriter.create<hivm::StoreOp>(loc, TypeRange(), value, storeInit);
  }
  Value storeInit = getLocalWorkSpaceTensor(
      rewriter, value.getLoc(), tensorType.getShape(),
      hivm::getTensorDynamicValues(rewriter, value.getLoc(), value),
      tensorType.getElementType());
  auto storeOp =
      rewriter.create<hivm::StoreOp>(loc, TypeRange(type), value, storeInit);
  storeOp->setAttr("inserted-store", rewriter.getUnitAttr());
  return storeOp;
}

hivm::LoadOp insertLoad(Value value, Location loc, PatternRewriter &rewriter) {
  Type type = value.getType();
  Type elemType = getElementTypeOrSelf(type);
  bool isBufferized = !isa<TensorType>(type);

  Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
      rewriter, loc, value, elemType, MemRefLayoutAttrInterface{});
  auto loadOp = rewriter.create<hivm::LoadOp>(
      loc, isBufferized ? TypeRange() : TypeRange(type), value, loadInit);
  loadOp->setAttr("inserted-load", rewriter.getUnitAttr());
  return loadOp;
}

std::pair<hivm::StoreOp, hivm::LoadOp>
insertStoreAndLoad(Value value, Location loc, PatternRewriter &rewriter) {
  Type type = value.getType();
  bool isBufferized = !isa<TensorType>(type);
  auto storeOp = insertStore(value, loc, rewriter);
  auto loadOp = insertLoad(
      isBufferized ? storeOp.getDst() : storeOp.getResult(0), loc, rewriter);
  return std::make_pair(storeOp, loadOp);
}

} // namespace PropagatorUtil

template <typename... Args>
static void printPropagator(StringRef prefix, TCoreType coreType,
                            ArrayRef<hivm::AddressSpace> addressSpaces,
                            Args &&...args);
template <typename... Args>
static void printPropagator(StringRef prefix, UnrealizedConversionCastOp op,
                            Args &&...args);
static void printPropagator() {}

template <typename... Args>
static void printPropagator(StringRef prefix, TCoreType coreType,
                            ArrayRef<hivm::AddressSpace> addressSpaces,
                            Args &&...args) {
  LDBG(prefix << " core type: " << coreType);
  LDBG(prefix << " address space: "
              << utils::debugger::to_string(addressSpaces));
  printPropagator(std::forward<Args>(args)...);
}

template <typename... Args>
static void printPropagator(StringRef prefix, UnrealizedConversionCastOp op,
                            Args &&...args) {
  auto coreType = PropagatorUtil::getCoreType(op);
  auto addressSpace = PropagatorUtil::getAddressSpace(op);
  printPropagator(prefix, coreType, addressSpace, std::forward<Args>(args)...);
}

template <typename... Args>
static WalkResult verifyFail(StringRef msg, Args &&...args) {
  LDBG("Failed to verify: " << msg);
  printPropagator(std::forward<Args>(args)...);
  return WalkResult::interrupt();
}

template <typename... Args>
static WalkResult verifyFail(Operation *op, StringRef msg, Args &&...args) {
  LDBG("Failed to verify(" << op->getName() << "): " << msg);
  LDBG(*op);
  printPropagator(std::forward<Args>(args)...);
  return WalkResult::interrupt();
}

template <typename... Args>
static WalkResult verifyFail(Value upOp, Operation *downOp, StringRef msg,
                             Args &&...args) {
  std::string upOpInfo;
  llvm::raw_string_ostream rso(upOpInfo);
  if (auto *defOp = upOp.getDefiningOp()) {
    rso << defOp->getName().getStringRef();
  } else {
    auto blockArg = cast<BlockArgument>(upOp);
    auto *parentOp = blockArg.getParentBlock()->getParentOp();
    rso << blockArg.getArgNumber() << "th BlockArgument of "
        << parentOp->getName();
  }
  LDBG("Failed to verify(" << upOpInfo << " -> " << downOp->getName()
                           << "): " << msg);
  LDBG(upOp);
  LDBG(*downOp);
  printPropagator(std::forward<Args>(args)...);
  return WalkResult::interrupt();
}

static WalkResult verifyUpPropagator(UnrealizedConversionCastOp upPropOp) {
  auto downPropOp =
      upPropOp.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!downPropOp || !downPropOp->hasAttr(kPropagateDownAttr)) {
    if (upPropOp.getInputs()[0].getType().isIntOrIndexOrFloat())
      return WalkResult::advance();
    return verifyFail(upPropOp, "Up propagation is not done correctly");
  }
  if (!upPropOp->hasOneUse())
    return verifyFail(upPropOp, "Up propagation has several use");
  auto [upCoreType, upAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(upPropOp);
  auto [downCoreType, downAddressSpace] =
      PropagatorUtil::extractPropagatorInfo(downPropOp);
  if (upCoreType == downCoreType || upCoreType == TCoreType::CUBE_AND_VECTOR ||
      downCoreType == TCoreType::CUBE_AND_VECTOR)
    return WalkResult::advance();
  auto upOp = downPropOp.getInputs()[0];
  auto *downOp = *upPropOp->user_begin();
  return verifyFail(upOp, downOp, "Unresolved propagator conflict", "Down",
                    downPropOp, "up", upPropOp);
}

static WalkResult verifyDownPropagator(UnrealizedConversionCastOp downPropOp) {
  if (downPropOp->getResultTypes()[0].isIntOrIndexOrFloat())
    return WalkResult::advance();
  for (auto *user : downPropOp->getUsers()) {
    auto upPropOp = dyn_cast<UnrealizedConversionCastOp>(user);
    if (!upPropOp || !upPropOp->hasAttr(kPropagateUpAttr)) {
      return verifyFail(downPropOp, "Down propagation is not done correctly");
    }
  }
  return WalkResult::advance();
}

LogicalResult verifyPropagation(func::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](Operation *op) {
    if (auto uccOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      LDBG("Verifying propagator: " << *op);
      if (op->hasAttr(kPropagateUpAttr))
        return verifyUpPropagator(uccOp);
      if (op->hasAttr(kPropagateDownAttr))
        return verifyDownPropagator(uccOp);
      return verifyFail(op, "UccOp must be propagator");
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

} // namespace mlir::hivm
