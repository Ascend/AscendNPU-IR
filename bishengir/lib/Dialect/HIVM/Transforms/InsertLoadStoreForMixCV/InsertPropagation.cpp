//===- InsertPropagation.cpp - Insert Propagation for trivial operations --===//
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

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/InsertPropagation.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "insert-load-store-insert-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

LogicalResult
InsertPropagationPattern::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  auto maybeResult = handleSpecialCase(op, rewriter);
  if (maybeResult.has_value())
    return maybeResult.value();
  if (isPropagatorInserted(op))
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
          >([&](Operation *op) {
        return insertPropagatorForCubeOp(op, rewriter);
      })
      .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
          , hivm::BitcastOp, tensor::ExtractOp, tensor::InsertOp,
          tensor::InsertSliceOp>([&](Operation *op) {
        return insertPropagatorForVectorOp(op, rewriter);
      })
      .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
          >(
          [&](Operation *op) { return insertPropagatorForDMAOp(op, rewriter); })
      .Case<tensor::EmptyOp>([&](auto op) {
        if (isa<hivm::FixpipeOp>(*op->user_begin()))
          return failure();
        PropagatorUtil::createPropagatorDown(op, TCoreType::CUBE_AND_VECTOR,
                                             rewriter);
        return success();
      })
      .Default([&](auto *op) { return failure(); });
}

bool InsertPropagationPattern::isPropagatorInserted(Operation *op) const {
  bool downPropInserted =
      !op->use_empty() && llvm::all_of(op->getUsers(), [](auto *user) {
        return user->hasAttr(kPropagateDownAttr);
      });
  bool upPropInserted = llvm::all_of(op->getOperands(), [](auto opr) {
    auto *op = opr.getDefiningOp();
    if (!isa<ShapedType>(opr.getType()))
      return true;
    return op && op->hasAttr(kPropagateUpAttr);
  });
  return upPropInserted || downPropInserted;
}

LogicalResult InsertPropagationPattern::insertPropagatorForCubeOp(
    Operation *op, PatternRewriter &rewriter) const {
  auto hivmOp = dyn_cast<HIVMStructuredOp>(op);
  if (!hivmOp)
    return op->emitOpError("CubeOp is not HIVM structured op");
  for (auto *src : hivmOp.getDpsInputOperands()) {
    auto addressSpace = hivm::AddressSpace::L1;
    if (isa<hivm::MatmulOp, hivm::MixMatmulOp, hivm::MixGroupMatmulOp>(op))
      addressSpace = hivm::AddressSpace::GM;
    PropagatorUtil::createPropagatorUp(src, TCoreType::CUBE,
                                       addressSpace, rewriter);
  }
  for (auto &dst : hivmOp.getDpsInitsMutable()) {
    PropagatorUtil::createPropagatorUp(&dst, TCoreType::CUBE,
                                       hivm::AddressSpace::L0C, rewriter);
  }
  PropagatorUtil::createPropagatorsDown(op, TCoreType::CUBE,
                                        hivm::AddressSpace::L0C, rewriter);
  return success();
}

LogicalResult InsertPropagationPattern::insertPropagatorForVectorOp(
    Operation *op, PatternRewriter &rewriter) const {
  PropagatorUtil::createPropagatorsUp(op, TCoreType::VECTOR,
                                      hivm::AddressSpace::UB, rewriter);
  PropagatorUtil::createPropagatorsDown(op, TCoreType::VECTOR,
                                        hivm::AddressSpace::UB, rewriter);
  return success();
}

LogicalResult InsertPropagationPattern::insertPropagatorForDMAOp(
    Operation *op, PatternRewriter &rewriter) const {
  // Assumes other operands are inferrable from other op or function argument.
  LDBG("Inserting propagator for DMA Op: " << *op);
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hivm::LoadOp>([&](auto op) {
        auto *srcOp = &op.getSrcMutable();
        auto *dstOp = &op.getDstMutable();
        if (PropagatorUtil::getUpPropagator(srcOp))
          return failure();
        PropagatorUtil::createPropagatorUp(srcOp, hivm::AddressSpace::GM,
                                           rewriter);
        if (auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
            forOp && forOp->hasAttr(hivm::ParallelLoopAttr::name)) {
          // Unstructured load is always UB load
          PropagatorUtil::createPropagatorUp(dstOp, TCoreType::VECTOR,
                                             hivm::AddressSpace::UB, rewriter);
          PropagatorUtil::createPropagatorsDown(
              op, TCoreType::VECTOR, hivm::AddressSpace::UB, rewriter);
        } else {
          PropagatorUtil::createPropagatorUp(dstOp, TCoreType::CUBE_AND_VECTOR,
                                             rewriter);
          PropagatorUtil::createPropagatorsDown(op, TCoreType::CUBE_AND_VECTOR,
                                                rewriter);
        }
        return success();
      })
      .Case<hivm::StoreOp>([&](auto op) {
        auto *srcOp = &op.getSrcMutable();
        auto *dstOp = &op.getDstMutable();
        if (PropagatorUtil::getUpPropagator(dstOp))
          return failure();
        PropagatorUtil::createPropagatorUp(srcOp, TCoreType::VECTOR,
                                           hivm::AddressSpace::UB, rewriter);
        PropagatorUtil::createPropagatorUp(dstOp, hivm::AddressSpace::GM,
                                           rewriter);
        PropagatorUtil::createPropagatorsDown(op, hivm::AddressSpace::GM,
                                              rewriter);
        return success();
      })
      .Case<hivm::FixpipeOp>([&](auto op) {
        auto *dstOp = &op.getDstMutable();
        if (op->use_empty() || isPropagatorInserted(op))
          return failure();
        PropagatorUtil::createPropagatorUp(dstOp, hivm::AddressSpace::GM,
                                           rewriter);
        PropagatorUtil::createPropagatorsDown(op, hivm::AddressSpace::GM,
                                              rewriter);
        return success();
      })
      .Default([](auto *op) { return failure(); });
}

std::optional<LogicalResult>
InsertPropagationPattern::handleSpecialCase(Operation *op,
                                            PatternRewriter &rewriter) const {
  // Handle scalar case
  if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (isPropagatorInserted(op)) {
      return failure();
    }
    if (auto storeOp = extractOp.getTensor().getDefiningOp<hivm::StoreOp>();
        storeOp && storeOp->hasAttr("inserted-store")) {
      PropagatorUtil::createPropagatorsUp(op, hivm::AddressSpace::GM, rewriter);
      return success();
    }
    return insertPropagatorForVectorOp(op, rewriter);
  }

  // Handle MayImplicitTransposeWithLastAxis
  if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
        toTensorOp->getResult(0), "MayImplicitTransposeWithLastAxis");
    if (maybeAnnotateOp.has_value() || toTensorOp->getAttr("gather_load") ||
        toTensorOp->getAttr("index_select_simd")) {
      if (isPropagatorInserted(op))
        return failure();
      return insertPropagatorForVectorOp(toTensorOp, rewriter);
    }
    return failure();
  }

  // Handle maybeUnCollapsibleReshape
  if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    auto maybeAnnotation = mlir::utils::getAnnotateOpWithAttr(
        collapseOp.getResult(), "maybeUnCollapsibleReshape");
    if (maybeAnnotation.has_value() && !isPropagatorInserted(op)) {
      return insertPropagatorForVectorOp(collapseOp, rewriter);
    }
    return failure();
  }

  // Handle unstructured load
  if (utils::isUnstructuredMemAccLoop(op)) {
    if (!isPropagatorInserted(op)) {
      return insertPropagatorForVectorOp(op, rewriter);
    }
    return failure();
  }

  // Handle elide_after_bufferize
  if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op);
      insertSliceOp && insertSliceOp->hasAttr("elide_after_bufferize"))
    return failure();

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// A5InsertionPattern
//===----------------------------------------------------------------------===//

LogicalResult
A5InsertionPattern::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  if (isPropagatorInserted(op))
    return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<func::CallOp>([&](auto callOp) {
        if (!isVFCall(callOp))
          return failure();
        PropagatorUtil::createPropagatorsUp(callOp, TCoreType::VECTOR,
                                            hivm::AddressSpace::UB, rewriter);
        PropagatorUtil::createPropagatorsDown(callOp, TCoreType::VECTOR,
                                              hivm::AddressSpace::UB, rewriter);
        return success();
      })
      .Case<hivm::FixpipeOp>([&](auto op) {
        auto *dstOp = &op.getDstMutable();
        if (op->use_empty())
          return failure();
        PropagatorUtil::createPropagatorUp(dstOp, TCoreType::CUBE_AND_VECTOR,
                                           rewriter);
        PropagatorUtil::createPropagatorsDown(op, TCoreType::CUBE_AND_VECTOR, hivm::AddressSpace::UB, rewriter);
        return success();
      })
      .Case<hivm::IndirectLoadOp>([&](auto op) {
        if (op->use_empty())
          return failure();
        auto *srcOp = &op.getSrcMutable();
        PropagatorUtil::createPropagatorUp(srcOp, hivm::AddressSpace::GM,
                                          rewriter);
        for (auto &input : op.getDpsInputOperands()) {
          if (input == srcOp)
            continue;
          PropagatorUtil::createPropagatorUp(input, TCoreType::VECTOR,
                                            hivm::AddressSpace::UB, rewriter);
        }
        auto *dstOp = &op.getDstMutable();
        PropagatorUtil::createPropagatorUp(dstOp, TCoreType::VECTOR,
                                           hivm::AddressSpace::UB, rewriter);
        PropagatorUtil::createPropagatorsDown(
            op, TCoreType::VECTOR, hivm::AddressSpace::UB, rewriter);
        return success();
      })
      .Case<hivm::IndirectStoreOp>([&](auto op) {
        auto *srcOp = &op.getSrcMutable();
        PropagatorUtil::createPropagatorUp(srcOp, TCoreType::VECTOR, hivm::AddressSpace::UB,
                                          rewriter);
        for (auto &input : op.getDpsInputOperands()) {
          if (input == srcOp)
            continue;
          PropagatorUtil::createPropagatorUp(input, TCoreType::VECTOR,
                                            hivm::AddressSpace::UB, rewriter);
        }
        auto *dstOp = &op.getDstMutable();
        PropagatorUtil::createPropagatorUp(dstOp,
                                           hivm::AddressSpace::GM, rewriter);
        PropagatorUtil::createPropagatorsDown(
            op, hivm::AddressSpace::GM, rewriter);
        return success();
      })
      .Default([&](auto *op) { return failure(); });
}

bool A5InsertionPattern::isPropagatorInserted(Operation *op) const {
  bool downPropInserted =
      !op->use_empty() && llvm::all_of(op->getUsers(), [](auto *user) {
        return user->hasAttr(kPropagateDownAttr);
      });
  bool upPropInserted = llvm::all_of(op->getOperands(), [](auto opr) {
    auto *op = opr.getDefiningOp();
    if (!isa<ShapedType>(opr.getType()))
      return true;
    return op && op->hasAttr(kPropagateUpAttr);
  });
  return upPropInserted || downPropInserted;
}

//===----------------------------------------------------------------------===//
// InsertCVDataMovementPattern
//===----------------------------------------------------------------------===//

LogicalResult
InsertCVDataMovementPattern::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  if (isPropagatorInserted(op))
  return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hivm::FixpipeOp>(
          [&](Operation *op) { return insertPropagatorForDMAOp(op, rewriter); })
      .Default([&](auto *op) { return failure(); });
}

bool InsertCVDataMovementPattern::isPropagatorInserted(Operation *op) const {
  bool downPropInserted =
      !op->use_empty() && llvm::all_of(op->getUsers(), [](auto *user) {
        return user->hasAttr(kPropagateDownAttr);
      });
  bool upPropInserted = llvm::all_of(op->getOperands(), [](auto opr) {
    auto *op = opr.getDefiningOp();
    if (!isa<ShapedType>(opr.getType()))
      return true;
    return op && op->hasAttr(kPropagateUpAttr);
  });
  return upPropInserted || downPropInserted;
}

LogicalResult InsertCVDataMovementPattern::insertPropagatorForDMAOp(
    Operation *op, PatternRewriter &rewriter) const {
  // Assumes other operands are inferrable from other op or function argument.
  LDBG("[A5-CV-DATA-MOVEMENT] Inserting propagator for DMA Op: " << *op);
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hivm::FixpipeOp>([&](auto op) {
        auto *dstOp = &op.getDstMutable();
        if (op->use_empty() || isPropagatorInserted(op))
          return failure();
        PropagatorUtil::createPropagatorUp(dstOp, TCoreType::CUBE_AND_VECTOR,
                                           rewriter);
        PropagatorUtil::createPropagatorsDown(op, TCoreType::CUBE_AND_VECTOR, hivm::AddressSpace::UB, rewriter);
        return success();
      })
      .Default([](auto *op) { return failure(); });
}

//===----------------------------------------------------------------------===//
// InsertTightCoupledBufferPattern
//===----------------------------------------------------------------------===//

LogicalResult
InsertTightCoupledBufferPattern::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  if (isPropagatorInserted(op))
  return failure();

  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hivm::FixpipeOp>(
          [&](Operation *op) { return insertPropagatorForDMAOp(op, rewriter); })
      .Default([&](auto *op) { return failure(); });
}

bool InsertTightCoupledBufferPattern::isPropagatorInserted(Operation *op) const {
  bool downPropInserted =
      !op->use_empty() && llvm::all_of(op->getUsers(), [](auto *user) {
        return user->hasAttr(kPropagateDownAttr);
      });
  bool upPropInserted = llvm::all_of(op->getOperands(), [](auto opr) {
    auto *op = opr.getDefiningOp();
    if (!isa<ShapedType>(opr.getType()))
      return true;
    return op && op->hasAttr(kPropagateUpAttr);
  });
  return upPropInserted || downPropInserted;
}

LogicalResult InsertTightCoupledBufferPattern::insertPropagatorForDMAOp(
    Operation *op, PatternRewriter &rewriter) const {
  // Assumes other operands are inferrable from other op or function argument.
  LDBG("[A5-CV-TightCoupledBuffer] Inserting propagator for DMA Op: " << *op);
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<hivm::FixpipeOp>([&](auto op) {
        auto *dstOp = &op.getDstMutable();
        if (op->use_empty() || isPropagatorInserted(op))
          return failure();
        PropagatorUtil::createPropagatorUp(dstOp, TCoreType::CUBE_AND_VECTOR,
                                           rewriter);
        PropagatorUtil::createPropagatorsDown(op, TCoreType::CUBE_AND_VECTOR, hivm::AddressSpace::UB, rewriter);
        return success();
      })
      .Default([](auto *op) { return failure(); });
}

} // namespace mlir::hivm