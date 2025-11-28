//===- InsertLoadStoreForMixCV.cpp ------------------------------*- C++ -*-===//
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
// This pass inserts load/store op for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTLOADSTOREFORMIXCV
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "insert-load-store"

namespace {
struct InsertLoadStoreForMixCVPass
    : public impl::InsertLoadStoreForMixCVBase<InsertLoadStoreForMixCVPass> {
  using Base::Base;
  void runOnOperation() override;
};

enum class InsertMode { LoadOnly = 0, StoreOnly, LoadAndStore };

Value insertLoadOperation(PatternRewriter &rewriter, Location loc,
                          OpOperand *consumerOperand, Operation **lastInsertOp,
                          std::optional<Value> insertInit = std::nullopt) {
  Type type = consumerOperand->get().getType();
  Type elemType = getElementTypeOrSelf(type);
  bool isBufferized = !isa<TensorType>(type);
  Value loadInit = insertInit.has_value()
                       ? insertInit.value()
                       : mlir::utils::createEmptyOpWithTargetElemType(
                             rewriter, loc, consumerOperand->get(), elemType,
                             MemRefLayoutAttrInterface{});
  *lastInsertOp = rewriter.create<hivm::LoadOp>(
      loc, isBufferized ? TypeRange() : TypeRange(type), consumerOperand->get(),
      loadInit);
  return isBufferized ? loadInit : (*lastInsertOp)->getResult(0);
}

Value insertStoreOperation(PatternRewriter &rewriter, Location loc,
                           OpOperand *consumerOperand, Operation **lastInsertOp,
                           std::optional<Value> insertInit = std::nullopt) {
  Type type = consumerOperand->get().getType();
  bool isBufferized = !isa<TensorType>(type);

  Value storeInit =
      insertInit.has_value()
          ? insertInit.value()
          : utils::createEmptyOp(rewriter, loc, consumerOperand->get());
  *lastInsertOp = rewriter.create<hivm::StoreOp>(
      loc, isBufferized ? TypeRange() : TypeRange(type), consumerOperand->get(),
      storeInit);
  return isBufferized ? storeInit : (*lastInsertOp)->getResult(0);
}

Value inertLoadStoreOperation(PatternRewriter &rewriter, Location loc,
                              OpOperand *consumerOperand,
                              Operation **lastInsertOp,
                              std::optional<Value> insertInit = std::nullopt) {
  Type type = consumerOperand->get().getType();
  Type elemType = getElementTypeOrSelf(type);
  bool isBufferized = !isa<TensorType>(type);

  Value storeInit = utils::createEmptyOp(rewriter, loc, consumerOperand->get());
  auto storeOp = rewriter.create<hivm::StoreOp>(
      loc, isBufferized ? TypeRange() : TypeRange(type), consumerOperand->get(),
      storeInit);
  Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
      rewriter, loc, consumerOperand->get(), elemType,
      MemRefLayoutAttrInterface{});
  *lastInsertOp = rewriter.create<hivm::LoadOp>(
      loc, isBufferized ? TypeRange() : TypeRange(type),
      isBufferized ? storeInit : storeOp->getResults()[0], loadInit);
  return isBufferized ? loadInit : (*lastInsertOp)->getResult(0);
}

LogicalResult
insertLoadStoreOp(PatternRewriter &rewriter, Location loc,
    const llvm::SmallVector<OpOperand *> &consumerOperands,
                  InsertMode insertMode,
                  std::optional<Value> insertInit = std::nullopt) {
  if (consumerOperands.empty()) {
    return failure();
  }

  Value replaceOperand;
  for (OpOperand *consumerOperand : consumerOperands) {
    Operation *lastInsertOp = nullptr;
    rewriter.setInsertionPointAfterValue(consumerOperand->get());
    if (insertMode == InsertMode::LoadOnly) {
      replaceOperand = insertLoadOperation(rewriter, loc, consumerOperand,
                                           &lastInsertOp, insertInit);
    } else if (insertMode == InsertMode::StoreOnly) {
      replaceOperand = insertStoreOperation(rewriter, loc, consumerOperand,
                                            &lastInsertOp, insertInit);
    } else if (insertMode == InsertMode::LoadAndStore) {
      replaceOperand = inertLoadStoreOperation(rewriter, loc, consumerOperand,
                                               &lastInsertOp, insertInit);
    }
    if (!lastInsertOp) {
      llvm_unreachable("lastInsertOp not defined");
      return failure();
    }
    rewriter.modifyOpInPlace(consumerOperand->getOwner(),
                             [&]() { consumerOperand->set(replaceOperand); });
  }

  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// InsertLoadOpBetweenStoreLikeAndVectorOrCube
//===----------------------------------------------------------------------===//

/// Pattern to insert load op between store-like operation and consumer.
///
/// For example:
/// ```
/// store       ins(...) outs(%dst)
/// consumer    ins(%dst)
/// ```
///
/// Is convert into:
/// ```
/// store       ins(...) outs(%dst)
/// load        ins(%dst) outs(%tmp)
/// consumer    ins(%tmp)
/// ```
template <typename OpType>
struct InsertLoadOpBetweenStoreLikeAndVectorOrCube
    : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!isa<hivm::HIVMStructuredOp>(op.getOperation()) &&
        !isa<tensor::ExtractOp>(op.getOperation())) {
      return failure();
    }

    if (isa<tensor::ExtractOp>(op.getOperation())) {
      // TODO: improve InsertWorkSpaceForMixCV.cpp to include tensor.extract
      // as a kind of load operation; then remove this part and the above
      // tensor::ExtractOp case
      if (op.getOperation()->hasAttr(
              "DuplicateTensorExtractForCube::newExtractLabel")) {
        return failure();
      }
    }

    Operation *opPtr = op.getOperation();
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : opPtr->getOpOperands()) {
      if (traceDefOp<hivm::FixpipeOp>(operand.get()).has_value() ||
          traceDefOp<hivm::StoreOp>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, opPtr->getLoc(), consumerOperands,
                             InsertMode::LoadOnly);
  }
};

//===----------------------------------------------------------------------===//
// InsertStoreOpBetweenVectorAndLoad
//===----------------------------------------------------------------------===//

/// Pattern to insert store op between vector and load operation.
///
/// For example:
/// ```
/// vector ins(%src) outs(%dst)
/// load   ins(%dst)
/// ```
///
/// Is convert into:
/// ```
/// vector ins(%src) outs(%dst)
/// store  ins(%dst) outs(%tmp)
/// load   ins(%tmp)
/// ```
template <typename OpType>
struct InsertStoreOpBetweenVectorAndLoad
    : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::StoreOnly);
  }
};

//===----------------------------------------------------------------------===//
// InsertLoadStoreOpBetweenVectorAndCube
//===----------------------------------------------------------------------===//

/// Pattern to insert load/store ops between producer and consumer.
///
/// For example:
/// ```
/// producer    ins(...) outs(%src)
/// consumer    ins(%src)
/// ```
///
/// Is convert into:
/// ```
/// producer    ins(...) outs(%src)
/// store       ins(%src) outs(%tmp)
/// load        ins(%tmp) outs(%tmp')
/// consumer    ins(%tmp')
/// ```
template <typename OpType>
struct InsertLoadStoreOpBetweenVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

template <typename OpType>
struct InsertLoadStoreOpBetweenCrossLoopVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (!isa<BlockArgument>(operand.get())) {
        continue;
      }

      auto scfForOp = dyn_cast<scf::ForOp>(op->getParentOp());
      if (!scfForOp) {
        continue;
      }

      auto blockArg = cast<BlockArgument>(operand.get());
      auto *yield = scfForOp.getTiedLoopYieldedValue(blockArg);
      if (!yield) {
        continue;
      }
      
      if (traceDefOp<OpType>(yield->get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

/// Specialized case for indirect memory access.
///
/// `scf.for` with attr "ExtractedLoadOrStore" describes the process of
/// discretely loading scalars to UB.
/// For example:
/// ```
/// for i in 16 {
///   dst[i] = src[offset[i]]
/// } {ExtractedLoadOrStore}
/// mmadl1(dst)
/// ```
///
/// Is converted into:
/// ```
/// for i in 16 {
///   dst[i] = src[offset[i]]
/// } {ExtractedLoadOrStore}
/// gm_dst = store ins(dst) outs(gm)
/// l1_dst = load  ins(gm_dst) outs(tmp)
/// mmadl1(l1_dst)
/// ```
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto scfForDef = traceDefOp<scf::ForOp>(operand.get());
      if (scfForDef.has_value()) {
        auto forOp = llvm::cast<scf::ForOp>(scfForDef.value());
        if (forOp->getAttr("ExtractedLoadOrStore") != nullptr) {
          consumerOperands.push_back(&operand);
        }
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

/// Specialized case for implicit transpose.
///
/// `bufferization.to_tensor` with attr "MayImplicitTransposeWithLastAxis"
/// describes the process of transposing data on UB. Store & load op will be
/// added here in order to make transpose operation happen in vector.
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto toTensorOpDef = traceDefOp<bufferization::ToTensorOp>(operand.get());
      if (!toTensorOpDef.has_value())
        continue;
      auto toTensorOp =
          llvm::cast<bufferization::ToTensorOp>(toTensorOpDef.value());
      auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
          toTensorOp->getResult(0), "MayImplicitTransposeWithLastAxis");
      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
      }

      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
      } else if (toTensorOp->getAttr("gather_load") != nullptr) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

/// Specialized case for reassocicative reshapes that might be noncontiguous.
///
/// `tensor.collapse_shape` with attr "maybeUnCollapsibleReshape" means that
/// it's likely that the collapse shape will become noncontiguous. Since only
/// vector core is able to such case, we need to insert load/store.
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      std::optional<Operation *> defOp =
          traceDefOp<tensor::CollapseShapeOp>(operand.get());
      if (!defOp.has_value())
        continue;
      auto collapse = cast<tensor::CollapseShapeOp>(defOp.value());
      std::optional<Operation *> maybeAnnotation =
          mlir::utils::getAnnotateOpWithAttr(collapse.getResult(),
                                             "maybeUnCollapsibleReshape");
      if (maybeAnnotation.has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

//===----------------------------------------------------------------------===//
// InsertStoreForSCFYield
//===----------------------------------------------------------------------===//

/// Pattern to insert store op for yielded value in `scf.for` op.
///
/// For example:
/// ```
/// %1 = fixpipe
/// %4 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    scf.yield %3
/// }
/// ```
///
/// Is converted into:
/// ```
/// %1 = fixpipe
/// %5 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    %4 = %store (%3)
///    scf.yield %4
/// }
/// ```
struct InsertStoreForSCFYield : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.hasPureTensorSemantics()) {
      return failure();
    }
    auto blockArg = dyn_cast_if_present<BlockArgument>(loadOp.getSrc());
    if (!blockArg) {
      return failure();
    }
    auto scfForOp =
        dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!scfForOp) {
      return failure();
    }
    OpOperand *yieldOperand = scfForOp.getTiedLoopYieldedValue(blockArg);
    if (traceDefOp<hivm::FixpipeOp>(yieldOperand->get()).has_value() ||
        traceDefOp<hivm::StoreOp>(yieldOperand->get()).has_value()) {
      return failure();
    }
    auto yieldOp = cast<scf::YieldOp>(scfForOp.getBody()->getTerminator());
    return insertLoadStoreOp(rewriter, yieldOp.getLoc(),
                             llvm::SmallVector<OpOperand *>{yieldOperand},
                             InsertMode::StoreOnly, blockArg);
  }
};

/// pattern5 (for tensor.extract)

struct DuplicateTensorExtractForCube
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  constexpr static llvm::StringRef visitedLabel =
      "DuplicateTensorExtractForCube::visitedLabel";
  constexpr static llvm::StringRef newExtractLabel =
      "DuplicateTensorExtractForCube::newExtractLabel";
  constexpr static llvm::StringRef replacementLabel =
      "DuplicateTensorExtractForCube::replacementLabel";
  constexpr static llvm::StringRef cubeErasureLabel =
      "DuplicateTensorExtractForCube::cubeErasureLabel";

  void markCoreType(PatternRewriter &rewriter, Location location, Value value,
                    TCoreType tCoreType) const {
    auto markOp = rewriter.create<annotation::MarkOp>(location, value);
    markOp->setAttr(
        mlir::hivm::TCoreTypeMarkerAttr::name,
        mlir::hivm::TCoreTypeMarkerAttr::get(markOp->getContext(), tCoreType));
  }

  bool findCubeUser(tensor::ExtractOp extractOp) const {
    bool hasCubeUser = false;
    SmallVector<Operation *> worklist;
    
    if (extractOp->getNumResults() > 0) {
      for (Operation *userOp : extractOp->getResult(0).getUsers()) {
        worklist.push_back(userOp);
      }
    } else {
      return false; 
    }

    SmallPtrSet<Operation *, 16> visited;
    while (!worklist.empty()) {
      Operation *currentOp = worklist.pop_back_val();

      if (!visited.insert(currentOp).second) {
        continue;
      }

      currentOp->walk([&hasCubeUser](Operation *nestedOp) {
        if (getCoreType(nestedOp) == TCoreType::CUBE) {
          hasCubeUser = true;
          return WalkResult::interrupt();
        } else if (getCoreType(nestedOp) == TCoreType::VECTOR) {
          return WalkResult::skip();
        }
        return WalkResult::advance();
      });

      if (hasCubeUser) {
        return true;
      }

      for (Value result : currentOp->getResults()) {
        for (Operation *userOp : result.getUsers()) {
          worklist.push_back(userOp);
        }
      }
    }

    return false;
  }

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // check if it has already been visited
    if (extractOp.getOperation()->hasAttr(visitedLabel)) {
      return failure();
    }
    extractOp.getOperation()->setAttr(visitedLabel,
                                      rewriter.getI32IntegerAttr(1));

    // if the extractOp is in for loop, and the for loop is gather_load,
    // the for loop should be atomic, don't insert any other op.
    auto forOp = extractOp->getParentOfType<scf::ForOp>();
    if (forOp && forOp->hasAttrOfType<UnitAttr>(hivm::ParallelLoopAttr::name)) {
      return failure();
    }

    // only process cases with vector sources
    Value originTensor = extractOp.getTensor();
    if (getElementTypeOrSelf(originTensor) == rewriter.getI1Type()) {
      // TODO: handle i1 cases for every load/store in this file
      return failure();
    }
    Operation *definingOp = originTensor.getDefiningOp();
    if (!definingOp) {
      return failure();
    }
    TensorType tensorType = cast<TensorType>(originTensor.getType());
    TCoreType originCoreType = getCoreType(definingOp).value();
    if (originCoreType != TCoreType::VECTOR) {
      // handle the case of direct load
      // TODO: (plan A) bubble up (plan B) infer load to vector type
      auto presumedAllocOp = traceDefOp<memref::AllocOp>(originTensor);
      if (presumedAllocOp.has_value()) {
        auto allocOp = cast<memref::AllocOp>(presumedAllocOp.value());
        Value memrefValue = allocOp.getMemref();
        bool foundLoad = false;
        bool foundBufferization = false;
        SmallVector<Operation *, 2> tmpOps;
        for (Operation *userOp : memrefValue.getUsers()) {
          if (isa<hivm::LoadOp>(userOp) &&
              dyn_cast<hivm::LoadOp>(userOp).getDst() == memrefValue) {
            foundLoad = true;
            tmpOps.push_back(userOp);
          }
          if (isa<bufferization::ToTensorOp>(userOp) &&
              dyn_cast<bufferization::ToTensorOp>(userOp).getOperand() ==
                  memrefValue) {
            foundBufferization = true;
            tmpOps.push_back(userOp);
          }
        }
        if (!(tmpOps.size() == 2 && foundLoad && foundBufferization)) {
          return failure();
        } else {
          // the op need eraseLabel only if when the bufferization is from load
          allocOp->setAttr(cubeErasureLabel, rewriter.getI32IntegerAttr(1));
          for (auto op: tmpOps) {
            op->setAttr(cubeErasureLabel, rewriter.getI32IntegerAttr(1));
          }
        }
      } else {
        return failure();
      }
    }

    // only process cases with cube users
    if (!findCubeUser(extractOp)) {
      return failure();
    }

    // prepare for insertion
    Location loc = extractOp->getLoc();
    rewriter.setInsertionPointAfterValue(extractOp.getResult());

    // insert operations
    Value workSpaceTensor = getLocalWorkSpaceTensor(
        rewriter, loc, tensorType.getShape(), getElementTypeOrSelf(tensorType));
    hivm::StoreOp storeOp = rewriter.create<hivm::StoreOp>(
        loc, TypeRange(tensorType), originTensor, workSpaceTensor);
    markCoreType(rewriter, loc, storeOp.getResults()[0], TCoreType::VECTOR);
    tensor::ExtractOp newExtractOp = rewriter.create<tensor::ExtractOp>(
        loc, storeOp.getResultTensor(), extractOp.getIndices());
    newExtractOp.getOperation()->setAttr(visitedLabel,
                                         rewriter.getI32IntegerAttr(1));
    newExtractOp.getOperation()->setAttr(newExtractLabel,
                                         rewriter.getI32IntegerAttr(1));
    Operation *markOpForReplacement = rewriter.create<annotation::MarkOp>(
        loc, extractOp.getResult(), ValueRange{newExtractOp.getResult()},
        rewriter.getArrayAttr(SmallVector<Attribute>()));
    markOpForReplacement->setAttr(replacementLabel,
                                  rewriter.getI32IntegerAttr(1));
    return success();
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<OpType>,
               InsertStoreOpBetweenVectorAndLoad<OpType>,
               InsertLoadOpBetweenStoreLikeAndVectorOrCube<OpType>,
               InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType>>(
      patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateInsertLoadStorePattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
  registerOne<tensor::InsertSliceOp>(patterns);
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>>(
      patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>>(
          patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<tensor::ExtractOp>>(
      patterns.getContext());
}

void InsertLoadStoreForMixCVPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  populateInsertLoadStorePattern(patterns);
  patterns.insert<InsertStoreForSCFYield>(patterns.getContext());
  // TODO: move InferFuncCoreType to previous places; then this pass may return
  // immediately depending on FuncCoreType
  bool hasCube = false;
  funcOp->walk([&hasCube](Operation *nestedOp) {
    if (isa<hivm::MmadL1Op>(nestedOp)) {
      hasCube = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasCube) {
    patterns.insert<DuplicateTensorExtractForCube>(patterns.getContext());
  }
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertLoadStoreForMixCVPass() {
  return std::make_unique<InsertLoadStoreForMixCVPass>();
}
