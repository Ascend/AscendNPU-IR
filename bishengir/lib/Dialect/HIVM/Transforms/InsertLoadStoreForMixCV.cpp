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
//===-------------------- InsertLoadStoreForMixCV.cpp----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/InsertPropagation.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/PropagateOp.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/ResolvePropagation.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static const llvm::StringLiteral insertedConvertLayout = "inserted-convert-layout";

llvm::SmallVector<size_t> getIntegersOfArrayAttr(Operation *op, llvm::StringLiteral attrName) {
  ArrayAttr arr = op->getAttrOfType<ArrayAttr>(attrName);
  llvm::SmallVector<size_t> result;
  if (!arr)
    return result;
  result.reserve(arr.size());
  for (Attribute attr : arr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    // In production you may want to assert or handle errors.
    assert(intAttr && "ArrayAttr element is not an IntegerAttr");
    // Use getValue().getZExtValue() to interpret as unsigned.
    result.push_back(intAttr.getValue().getZExtValue());
  }
  return result;
}

void insertToIntegerAttrOfArrayAttr(Operation *op, llvm::StringLiteral attrName, size_t value, PatternRewriter &rewriter) {
  auto arr = getIntegersOfArrayAttr(op, attrName);
  if (std::find(arr.begin(), arr.end(), value) != arr.end())
    return;
  arr.push_back(value);
  std::sort(arr.begin(), arr.end());
  
  SmallVector<Attribute> newAttrs;
  for(auto idx : arr) {
    newAttrs.push_back(rewriter.getIntegerAttr(rewriter.getIntegerType(64), idx));
  }
  op->setAttr(attrName, rewriter.getArrayAttr(newAttrs));
}
struct InsertLoadStoreForMixCVPass
    : public impl::InsertLoadStoreForMixCVBase<InsertLoadStoreForMixCVPass> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  void runLegacyInsertLoadStoreForMixCV();
};

namespace {

// TODO: change certain places to trace
// e.g. InsertStoreForSCFYield loadOp.getSrc()

bool isGM(Value v) {
  // TODO: include func's args
  // TODO: use interface and use this function for storelike throughout this
  // file
  return traceDefOp<hivm::FixpipeOp>(v).has_value() ||
         traceDefOp<hivm::StoreOp>(v).has_value();
}

template <typename... OpType> bool canTraceTo(Value v) {
  return (false || ... || traceDefOp<OpType>(v).has_value());
}

bool isVec(Value v) {
  // TODO: use interface or else, including tensor.insert_slice, etc.
  return canTraceTo<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
             >(v) ||
         traceDefOp<tensor::InsertOp>(v).has_value() ||
         traceDefOp<tensor::InsertSliceOp>(v).has_value();
}

void markToAvoidDCE(PatternRewriter &rewriter, Location location, Value value) {
  rewriter.setInsertionPointAfterValue(value);
  auto markOp = rewriter.create<annotation::MarkOp>(location, value);
  markOp->setAttr("InsertLoadStoreForMixCV::markToAvoidDCE",
                  rewriter.getI32IntegerAttr(1));
}

} // namespace

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
private:
  void runLegacyInsertLoadStoreForMixCV();
  bool isA5Target();
  bool isEnabledTightCoupledBuffer();
  LogicalResult insertPropagationOp(func::FuncOp funcOp);
  LogicalResult propagateAndResolve(func::FuncOp funcOp);
  LogicalResult runPropagateOpPatterns(func::FuncOp funcOp,
    PropagationStep step);
  LogicalResult ensureAllocInUbAddressSpace(func::FuncOp funcOp);
  LogicalResult addConvertLayoutUBToL1(func::FuncOp funcOp);
};

bool InsertLoadStoreForMixCVPass::isA5Target() {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  auto moduleTarget = hacc::utils::getTargetDevice(moduleOp);
  if (moduleTarget.has_value()) {
    return hacc::utils::isAscend950(moduleTarget.value());
  }
  return hacc::utils::isAscend950(this->target);
}

bool InsertLoadStoreForMixCVPass::isEnabledTightCoupledBuffer()  {
  if (disableTightCoupledBuffer || enableDotScaledCompile) return false;
  return isA5Target();
}


enum class InsertMode { LoadOnly = 0, StoreOnly, LoadAndStore };

LogicalResult
InsertOpHelper(PatternRewriter &rewriter, Location loc,
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
      llvm::report_fatal_error("lastInsertOp not defined");
      return failure();
    }
    rewriter.modifyOpInPlace(consumerOperand->getOwner(),
                             [&]() { consumerOperand->set(replaceOperand); });
  for (OpOperand *consumerOperand : consumerOperands) {
    Operation *lastInsertOp = nullptr;
    rewriter.setInsertionPointAfterValue(consumerOperand->get());
    Type type = consumerOperand->get().getType();
    Type elemType = getElementTypeOrSelf(type);
    if (insertMode == InsertMode::LoadOnly) {
      Value loadInit =
          insertInit.has_value()
              ? insertInit.value()
              : mlir::utils::createEmptyOpWithTargetElemType(
                    rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), consumerOperand->get(), loadInit);
    } else if (insertMode == InsertMode::StoreOnly) {
      Value storeInit =
          insertInit.has_value()
              ? insertInit.value()
              : utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      lastInsertOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
    } else if (insertMode == InsertMode::LoadAndStore) {
      Value storeInit =
          utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      auto storeOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
      Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
          rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), storeOp->getResults()[0], loadInit);
    }
    if (!lastInsertOp) {
      llvm_unreachable("lastInsertOp not defined");
      return failure();
    }
    rewriter.modifyOpInPlace(consumerOperand->getOwner(), [&]() {
      consumerOperand->set(lastInsertOp->getResult(0));
    });
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
// InsertLoadStoreOp
//===----------------------------------------------------------------------===//
/// pattern1 : fixpipe op / store op + vector/mmadl1
/// convert into fixpipe op / store op  + hivm.load + vector/mmadl1
template <typename OpType>
struct InsertLoadOpBetweenStoreLikeAndVectorOrCube
    : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!isa<hivm::HIVMStructuredOp>(op.getOperation()) &&
        !isa<tensor::ExtractOp>(op.getOperation()) &&
        !isa<tensor::InsertOp>(op.getOperation()) &&
        !isa<tensor::InsertSliceOp>(op.getOperation())) {
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

    if (isa<tensor::InsertSliceOp>(op.getOperation())) {
      if (op.getOperation()->hasAttr("elide_after_bufferize")) {
        return failure();
      }
    }

    Operation *opPtr = op.getOperation();
    llvm::SmallVector<OpOperand *> consumerOperands;
    TypeSwitch<Operation *>(opPtr)
        .Case([&](hivm::StoreOp storeOp) {
          if (!storeOp->getOpOperands().empty()) {
            OpOperand &firstOperand = storeOp->getOpOperands().front();
            if (traceDefOp<hivm::FixpipeOp>(firstOperand.get(), false, true)
                    .has_value() ||
                traceDefOp<hivm::StoreOp>(firstOperand.get(), false, true)
                    .has_value()) {
              consumerOperands.push_back(&firstOperand);
            }
          }
        })
        .Default([&](Operation *genericOp) {
          for (OpOperand &operand : genericOp->getOpOperands()) {
            if (traceDefOp<hivm::FixpipeOp>(operand.get(), false, true)
                    .has_value() ||
                traceDefOp<hivm::StoreOp>(operand.get(), false, true)
                    .has_value()) {
              consumerOperands.push_back(&operand);
            }
          }
        });
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
  virtual ~InsertStoreOpBetweenVectorAndLoad() = default;
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    OpOperand &srcOperand = op.getSrcMutable();

    if (traceDefOp<OpType>(srcOperand.get()).has_value()) {
      consumerOperands.push_back(&srcOperand);
    }

    if (consumerOperands.empty()) {
      return failure();
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
template <typename OpType, typename CubeOpType>
struct InsertLoadStoreOpBetweenVectorAndCube
    : public OpRewritePattern<CubeOpType> {
  using OpRewritePattern<CubeOpType>::OpRewritePattern;

  virtual ~InsertLoadStoreOpBetweenVectorAndCube() = default;

  LogicalResult matchAndRewrite(CubeOpType op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    if (!isa<hivm::HIVMStructuredOp, hivm::BitcastOp>(op.getOperation())) {
      return failure();
    }

    auto hivmOp = op.getOperation();
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : hivmOp->getOpOperands()) {
      if (traceDefOp<hivm::FixpipeOp>(operand.get()).has_value() ||
          traceDefOp<hivm::StoreOp>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, hivmOp->getLoc(), consumerOperands,
                          InsertMode::LoadOnly);
  }
};

/// pattern2 : vector(src, dst) + hivm.load(dst)
/// convert into  vector(src, dst) + hivm.store(dst) + hivm.load
template <typename OpType>
struct InsertStoreOpBetweenVectorAndLoad
    : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  ~InsertStoreOpBetweenVectorAndLoad() override = default;
  LogicalResult matchAndRewrite(hivm::LoadOp op,
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

template <typename OpType, typename CubeOpType>
struct InsertLoadStoreOpBetweenCrossLoopVectorAndCube
    : public OpRewritePattern<CubeOpType> {
  using OpRewritePattern<CubeOpType>::OpRewritePattern;

  virtual ~InsertLoadStoreOpBetweenCrossLoopVectorAndCube() = default;

  LogicalResult matchAndRewrite(CubeOpType op,
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
template <typename CubeOpType>
struct InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp, CubeOpType>
    : public OpRewritePattern<CubeOpType> {
  using OpRewritePattern<CubeOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CubeOpType op,
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::StoreOnly);
  }
};

/// pattern3 : vector(dst) + cube(dst)
/// convert into vector + hivm.store + hivm.load  +cube
template <typename OpType>
struct InsertLoadStoreOpBetweenVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  ~InsertLoadStoreOpBetweenVectorAndCube() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : scf::for{discrete load src using scalar} with attr {ExtractedLoadOrStore} + cube(src)
/// ExtractedLoadOrStore describes the process of discretely loading scalars on ub.
/// convert into scf::for{discrete load src using scalar} + hivm.store + hivm.load  + cube
/// for example
/// dst = tensor.empty : 16
/// for i in 16
///   dst[i]=src[offset[i]]
/// mmadl1(dst)
/// convert into
/// dst = tensor.empty : 16
/// for i in 16
///    dst[i]=src[offset[i]]
///  gm_dst = store dst to gm
///  l1_dst = load dst from gm
///  mmadl1(l1_dst)
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
        if (forOp->getAttr(ExtractLoadStoreAttr) != nullptr) {
          consumerOperands.push_back(&operand);
        }
      }
    }
    return insertLoadStoreOp(rewriter, op.getLoc(), consumerOperands,
                             InsertMode::LoadAndStore);
  }
};

//===----------------------------------------------------------------------===//
// InsertLoadBeforeSCFInitArgs
//===----------------------------------------------------------------------===//

/// Pattern to insert load op before scf.for/while loop if its init args come
/// from store-like operations (Store/Fixpipe).
///
/// Example:
/// ```
/// %1 = hivm.fixpipe ...
/// scf.for ... iter_args(%arg = %1)
/// ```
///
/// Is converted into:
/// ```
/// %1 = hivm.fixpipe ...
/// %loaded = hivm.load %1
/// scf.for ... iter_args(%arg = %loaded)
/// ```
template <typename LoopOp>
struct InsertLoadBeforeSCFInitArgs : public OpRewritePattern<LoopOp> {
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const override {
    if (op.walk([](hivm::MmadL1Op) {
            return WalkResult::interrupt();
          }).wasInterrupted()) {
      return failure();
    }

    auto isStoreLike = [](Value v) {
      return traceDefOp<hivm::FixpipeOp>(v).has_value() ||
             traceDefOp<hivm::StoreOp>(v).has_value();
    };

    auto isLoadInLoop = [](Value innerArg) {
      return llvm::any_of(innerArg.getUsers(), [&](Operation *user) {
        auto loadOp = dyn_cast<hivm::LoadOp>(user);
        return loadOp && loadOp.getSrc() == innerArg;
      });
    };

    llvm::SmallVector<OpOperand *> operandsToFix;

    for (auto [initOperand, innerArg] :
         llvm::zip(op.getInitsMutable(), op.getRegionIterArgs())) {

      if (!isStoreLike(initOperand.get())) {
        continue;
      }

      if (isLoadInLoop(innerArg)) {
        continue;
      }

      operandsToFix.push_back(&initOperand);
    }

    return insertLoadStoreOp(rewriter, op.getLoc(), operandsToFix,
                             InsertMode::LoadOnly);
  }
};

/// Specialized case for implicit transpose.
///
/// `bufferization.to_tensor` with attr "MayImplicitTransposeWithLastAxis"
/// describes the process of transposing data on UB. Store & load op will be
/// added here in order to make transpose operation happen in vector.
template <typename CubeOpType>
struct InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp,
                                             CubeOpType>
    : public OpRewritePattern<CubeOpType> {
  using OpRewritePattern<CubeOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CubeOpType op,
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : bufferization::ToTensorOp with ToBeTransposed mark + cube
/// describe there is an implicit transpose for the input of cube. store & load
/// op will be added here in order to make transpose operation happen in vector
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
          toTensorOp->getResult(0), kMayImplicitTransposeWithLastAxis);

      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
      } else if (toTensorOp->getAttr("gather_load") != nullptr) {
        consumerOperands.push_back(&operand);
      } else if (toTensorOp->getAttr("index_select_simd") != nullptr) {
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
template <typename CubeOpType>
struct InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp,
                                             CubeOpType>
    : public OpRewritePattern<CubeOpType> {
  using OpRewritePattern<CubeOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CubeOpType op,
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
          toTensorOp->getResult(0), "ToBeTransposed");
      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
        rewriter.eraseOp(maybeAnnotateOp.value());
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern4
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
///
/// is converted into
///
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
        if (getCoreType(nestedOp) == TCoreType::CUBE ||
            getCoreType(nestedOp) == TCoreType::CUBE_OR_VECTOR) {
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

    // only process cases with cube users
    if (!findCubeUser(extractOp)) {
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
          for (auto op : tmpOps) {
            op->setAttr(cubeErasureLabel, rewriter.getI32IntegerAttr(1));
          }
        }
      } else {
        return failure();
      }
    }

    // prepare for insertion
    Location loc = extractOp->getLoc();
    rewriter.setInsertionPointAfterValue(extractOp.getResult());

    // insert operations
    Value workSpaceTensor = getLocalWorkSpaceTensor(
        rewriter, loc, tensorType.getShape(),
        hivm::getTensorDynamicValues(rewriter, loc, originTensor),
        getElementTypeOrSelf(tensorType));
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

struct InsertStoreForSCFIF : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp.elseBlock()) {
      return failure();
    }
    auto thenYield = ifOp.thenYield();
    auto elseYield = ifOp.elseYield();
    for (const auto &[thenOpr, elseOpr] :
         llvm::zip(thenYield->getOpOperands(), elseYield->getOpOperands())) {
      if (isGM(thenOpr.get()) ^ isGM(elseOpr.get())) {
        // TODO: replace storelike's workspace dst with memref to avoid DEC
        markToAvoidDCE(rewriter, ifOp.getLoc(), ifOp.getResults()[0]);
        if (!isGM(thenOpr.get())) {
          return insertLoadStoreOp(rewriter, thenYield.getLoc(),
                                   llvm::SmallVector<OpOperand *>{&thenOpr},
                                   InsertMode::StoreOnly);
        } else {
          return insertLoadStoreOp(rewriter, elseYield.getLoc(),
                                   llvm::SmallVector<OpOperand *>{&elseOpr},
                                   InsertMode::StoreOnly);
        }
      }
    }
    return failure();
  }
};

template <>
struct InsertLoadOpBetweenStoreLikeAndVectorOrCube<scf::YieldOp>
    : public OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern<scf::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    auto scfForOp = dyn_cast_if_present<scf::ForOp>(yieldOp->getParentOp());
    if (!scfForOp) {
      return failure();
    }
    for (const auto &[yieldOpr, initVal] :
         llvm::zip(yieldOp->getOpOperands(), scfForOp.getInitArgs())) {
      if (isGM(yieldOpr.get())) {
        if (isVec(initVal) || traceDefOp<hivm::LoadOp>(initVal).has_value()) {
          return insertLoadStoreOp(rewriter, yieldOp->getLoc(),
                                   llvm::SmallVector<OpOperand *>{&yieldOpr},
                                   InsertMode::LoadOnly);
        }
      }
    }
    return failure();
  }
};

// TODO: while loop (consider before and after regions)

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<
      InsertLoadStoreOpBetweenVectorAndCube<OpType, hivm::MmadL1Op>,
      InsertLoadStoreOpBetweenVectorAndCube<OpType, hivm::Conv1DL1Op>,
      InsertLoadStoreOpBetweenVectorAndCube<OpType, hivm::Conv2DL1Op>,
      InsertLoadStoreOpBetweenVectorAndCube<OpType, hivm::Conv3DL1Op>,
      InsertLoadStoreOpBetweenVectorAndCube<OpType, hivm::BatchMmadL1Op>,
      InsertStoreOpBetweenVectorAndLoad<OpType>,
      InsertLoadOpBetweenStoreLikeAndVectorOrCube<OpType>,
      InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType,
                                                     hivm::BatchMmadL1Op>,
      InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType, hivm::MmadL1Op>,
      InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType, hivm::Conv1DL1Op>,
      InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType, hivm::Conv2DL1Op>,
      InsertLoadStoreOpBetweenCrossLoopVectorAndCube<OpType, hivm::Conv3DL1Op>>(
    if (!yieldOperand) {
      return failure();
    }
    auto correspondYieldVal = yieldOperand->get();
    if (traceDefOp<hivm::FixpipeOp>(correspondYieldVal).has_value() ||
        traceDefOp<hivm::StoreOp>(correspondYieldVal).has_value()) {
      return failure();
    }
    auto yieldOp = cast<scf::YieldOp>(scfForOp.getBody()->getTerminator());
    return InsertOpHelper(rewriter, yieldOp.getLoc(),
                          llvm::SmallVector<OpOperand *>{yieldOperand},
                          InsertMode::StoreOnly, blockArg);
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<OpType>,
               InsertStoreOpBetweenVectorAndLoad<OpType>,
               InsertLoadOpBetweenStoreLikeAndVectorOrCube<OpType>>(
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
  registerOne<tensor::InsertOp>(patterns);
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StoreOp>>(
      patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp, hivm::MmadL1Op>>(
          patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp, hivm::Conv1DL1Op>>(
          patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp, hivm::Conv2DL1Op>>(
          patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp, hivm::Conv3DL1Op>>(
          patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp,
                                                     hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp,
                                                     hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp,
                                                     hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp,
                                                     hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp,
                                                     hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp,
                                                     hivm::Conv1DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp,
                                                     hivm::Conv2DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<tensor::CollapseShapeOp,
                                                     hivm::Conv3DL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<tensor::ExtractOp>>(
      patterns.getContext());
}

LogicalResult applyInsertLoadBeforeSCFInitArgs(MLIRContext *context,
                                               Operation *funcOp) {
  RewritePatternSet patterns(context);
  patterns.insert<InsertLoadBeforeSCFInitArgs<scf::ForOp>,
                  InsertLoadBeforeSCFInitArgs<scf::WhileOp>>(
      patterns.getContext());
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

LogicalResult preProcessComplexControlFlow(MLIRContext *context,
                                           Operation *funcOp) {
  RewritePatternSet patterns(context);
  patterns.insert<InsertStoreForSCFIF>(patterns.getContext());
  patterns.insert<InsertLoadOpBetweenStoreLikeAndVectorOrCube<scf::YieldOp>>(
      patterns.getContext());
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

void InsertLoadStoreForMixCVPass::runLegacyInsertLoadStoreForMixCV() {
  auto funcOp = getOperation();
  auto *ctx = funcOp.getContext();
  if (failed(preProcessComplexControlFlow(ctx, funcOp))) {
    signalPassFailure();
  }
  if (failed(applyInsertLoadBeforeSCFInitArgs(ctx, funcOp))) {
    signalPassFailure();
  }
  RewritePatternSet patterns(ctx);
  populateInsertLoadStorePattern(patterns);
  patterns.insert<InsertStoreForSCFYield>(patterns.getContext());
  // TODO: move InferFuncCoreType to previous places; then this pass may
  // return immediately depending on FuncCoreType
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
  registerOne<func::CallOp>(patterns);
  registerOne<hivm::IndirectLoadOp>(patterns);
  registerOne<hivm::StrideLoadOp>(patterns);
  registerOne<hivm::BitcastOp>(patterns);
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::IndirectStoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StrideStoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>>(
      patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>>(
          patterns.getContext());
}

void InsertLoadStoreForMixCVPass::runLegacyInsertLoadStoreForMixCV() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  populateInsertLoadStorePattern(patterns);
  patterns.insert<InsertStoreForSCFYield>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

static LogicalResult runPropagateOpPatterns(func::FuncOp funcOp,
LogicalResult InsertLoadStoreForMixCVPass::runPropagateOpPatterns(func::FuncOp funcOp,
                                            PropagationStep step) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<PropagateUpPattern, PropagateDownPattern>(patterns.getContext(),
                                                         step);

  patterns.add<ResolvePropagationPattern, RemoveRedundantPropagationPattern>(
      patterns.getContext());
  rewriteConfig.fold = false;
  patterns.add<ResolvePropagationPattern, RemoveRedundantPropagationPattern>(
      patterns.getContext());
  rewriteConfig.fold = false;

  if (isEnabledTightCoupledBuffer()) {
    patterns.add<TightCoupledBufferResolvePropagationPattern>(patterns.getContext());
  }
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  LDBG("After propagation step " << static_cast<int>(step));
  LDBG(funcOp);
  return success();
}

static LogicalResult insertPropagationOp(func::FuncOp funcOp) {
LogicalResult InsertLoadStoreForMixCVPass::insertPropagationOp(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto arg : funcOp.getArguments()) {
    if (isa<ShapedType>(arg.getType())) {
      Value newArg = PropagatorUtil::createPropagator(
          arg, kPropagateDownAttr, hivm::AddressSpace::GM, rewriter);
      rewriter.replaceAllUsesExcept(arg, newArg, newArg.getDefiningOp());
    }
  }

  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<InsertPropagationPattern>(patterns.getContext());
  patterns.add<A5InsertionPattern>(patterns.getContext());
  if (isEnabledTightCoupledBuffer()) {
    patterns.add<TightCoupledBufferInsertionPattern>(patterns.getContext());
  }
  // TODO: add case for InsertUBAfterFixpipePattern
  // case of InsertL1BeforeMmadPattern tested by compile-triton-hstu-attn-fwd.mlir
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  return success();
}

static LogicalResult propagateAndResolve(func::FuncOp funcOp) {
  SmallVector<PropagationStep> propagations = {
      PropagationStep::LOCAL, PropagationStep::GM, PropagationStep::UB,
      PropagationStep::L1, PropagationStep::ALL};
LogicalResult InsertLoadStoreForMixCVPass::propagateAndResolve(func::FuncOp funcOp) {
  SmallVector<PropagationStep> propagations = {
      PropagationStep::LOCAL, PropagationStep::UB, PropagationStep::L1,
      PropagationStep::ALL};
  for (auto step : propagations) {
    if (failed(runPropagateOpPatterns(funcOp, step))) {
      return failure();
    }
  }
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<CloneMultipleAddressSpaceOperation>(patterns.getContext());
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  LDBG("After clone multiple address space: " << funcOp);
  if (failed(runPropagateOpPatterns(funcOp, PropagationStep::ALL))) {
    return failure();
  }
  return success();
}

// TODO: use this pattern to ensure alloc in UB for tight coupled buffer
struct EnsureAllocInUbAddressSpace
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  ~EnsureAllocInUbAddressSpace() override = default;
  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    auto oldType = dyn_cast<MemRefType>(allocOp.getMemref().getType());
    if (!oldType)
      return failure();
    auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(oldType);
    if (maybeSpace.has_value())
      return failure();
    if (!any_of(allocOp->getUsers(), [](Operation *op) -> bool {
      return isa<bufferization::ToTensorOp>(op);
    }))
      return failure();
    auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
    auto newType = mlir::MemRefType::get(oldType.getShape(),
                                        oldType.getElementType(),
                                        oldType.getLayout(),
                                        ubSpaceAttr);
    if (newType == oldType)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.modifyOpInPlace(allocOp,
                            [&]() { allocOp.getMemref().setType(newType); });

    rewriter.setInsertionPointAfter(allocOp);
    auto memSpaceCastOp = rewriter.create<memref::MemorySpaceCastOp>(
        allocOp.getLoc(), oldType, allocOp);
    rewriter.replaceOpUsesWithIf(allocOp, memSpaceCastOp.getResult(), [](OpOperand &opr) {
      return !isa<memref::MemorySpaceCastOp>(opr.getOwner());
    });
    return success();
  }
};

LogicalResult InsertLoadStoreForMixCVPass::ensureAllocInUbAddressSpace(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<EnsureAllocInUbAddressSpace>(patterns.getContext());
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  return success();
}


namespace mlir::hivm {

constexpr static llvm::StringLiteral maybeUnCollapsibleReshape =
       "maybeUnCollapsibleReshape";

static uint64_t getElemBytesForAlign(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return (uint64_t)((ft.getWidth() + 7) / 8);
  if (auto it = dyn_cast<IntegerType>(t))
    return (uint64_t)((it.getWidth() + 7) / 8);
  if (isa<IndexType>(t))
    return 8ULL;
  if (auto ct = dyn_cast<ComplexType>(t))
    return 2ULL * getElemBytesForAlign(ct.getElementType());
  return 0ULL;
}

static FailureOr<uint64_t> getBlockElemsFor32BAlign(Type elemType) {
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes <= 0)
    return failure();
  if (elemBytes >= kAlignBytes)
    return 1;
  if (kAlignBytes % elemBytes != 0)
    return failure();
  return kAlignBytes / elemBytes;
}

static void ensureAllocInUbAddressSpaceIfNeeded(PatternRewriter &rewriter,
                                                memref::AllocOp allocOp) {
  auto oldType = dyn_cast<MemRefType>(allocOp.getMemref().getType());
  if (!oldType)
    return;
  auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(oldType);
  if (maybeSpace.has_value())
    return;

  MLIRContext *ctx = rewriter.getContext();
  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto newType = mlir::MemRefType::get(oldType.getShape(),
                                       oldType.getElementType(),
                                       oldType.getLayout(),
                                       ubSpaceAttr);

  SmallVector<OpOperand *> originalUses;
  for (OpOperand &use : allocOp.getMemref().getUses())
    originalUses.push_back(&use);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.modifyOpInPlace(allocOp,
                           [&]() { allocOp.getMemref().setType(newType); });

  rewriter.setInsertionPointAfter(allocOp);
  Value replacement = allocOp.getMemref();
  if (replacement.getType() != oldType) {
    replacement = rewriter.create<memref::MemorySpaceCastOp>(
        allocOp.getLoc(), oldType, replacement);
  }
  for (OpOperand *use : originalUses)
    use->set(replacement);
}

/// pattern2
/// %53 = hivm.hir.vcast ins(%42 : tensor<16x16xf32>) outs(%19 : ...) -> ...
/// %54 = tensor.empty() : tensor<16x16xf32>
/// %55 = hivm.hir.mmadL1 ins(%53, %52, %true, %c16, %c16, %c16 : ...
///
/// is converted into
///
/// %53 = hivm.hir.vcast ins(%42 : tensor<16x16xf32>) outs(%19 : ...) -> ...
/// %alloc_0 = memref.alloc() : memref<..., #hivm.address_space<cbuf>>
/// %memspacecast_2 = memref.memory_space_cast %alloc_0
/// : memref<..., #hivm.address_space<cbuf>> to ...
/// %empty = bufferization.to_tensor %memspacecast_2 restrict writable : ...
/// %copy = hivm.hir.copy ins(%53 : ...) outs(%empty : ...) -> ...
/// %54 = tensor.empty() : tensor<16x16xf32>
/// %55 = hivm.hir.mmadL1 ins(%copy, %52, %true, %c16, %c16, %c16 : ...


LogicalResult insertConvertLayout(PatternRewriter &rewriter,
      hivm::MmadL1Op mmadOp,
      const llvm::SmallVector<OpOperand *> &consumerOperands) {
  if (consumerOperands.empty()) {
    return failure();
  }

  bool changed = false;
  for (OpOperand *consumerOperand : consumerOperands) {
    Value origTensor = consumerOperand->get();
    // TODO: enhance support for dynamic shape
    auto tensorType = mlir::dyn_cast<RankedTensorType>(origTensor.getType());
    if (!tensorType)
      continue;

    Operation *consumerOp = consumerOperand->getOwner();
    Location loc = consumerOp->getLoc();
    rewriter.setInsertionPoint(consumerOp);

    // TODO: Consider encapsulating it as an nd2dz function
    int64_t M = tensorType.getDimSize(0);
    int64_t N = tensorType.getDimSize(1);
    bool isA = (origTensor == mmadOp.getA());
    bool isTranspose = isA ? mmadOp.getATranspose().has_value()
                           : mmadOp.getBTranspose().has_value();
    auto blockSizes = mmadOp.getBlockSizesTile(origTensor, isTranspose, isA);
    int32_t alignM = static_cast<int32_t>(blockSizes[0]);
    int32_t alignN = static_cast<int32_t>(blockSizes[1]);
    auto elemType = tensorType.getElementType();
    int64_t newN = static_cast<int64_t>(AlignUp(static_cast<uint64_t>(N),
                            static_cast<uint64_t>(alignN)));
    if ((M != ShapedType::kDynamic && (M % alignM)) || (newN != N)) {
      int64_t newM = static_cast<int64_t>(
        AlignUp(static_cast<uint64_t>(M), static_cast<uint64_t>(alignM)));
      auto paddedType = RankedTensorType::get({newM, newN}, elemType);
      Value zeroConst = rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getZeroAttr(elemType));

      Value emptyForVbrc = rewriter.create<tensor::EmptyOp>(
              loc, paddedType.getShape(), elemType);
      auto vbrcOp = rewriter.create<hivm::VBrcOp>(
              loc, paddedType, zeroConst, emptyForVbrc);
      Value initializedMatrix = vbrcOp->getResult(0);
      SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(M), rewriter.getIndexAttr(N)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
      origTensor = rewriter.create<tensor::InsertSliceOp>(
              loc, origTensor, initializedMatrix, offsets, sizes, strides);
      tensorType = mlir::cast<RankedTensorType>(origTensor.getType());
      M = newM;
      N = newN;
    }

    SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2}};
    auto blkOr = getBlockElemsFor32BAlign(elemType);
    if (failed(blkOr)) {
      return consumerOp->emitOpError()
              << "unsupported element type for 32B-aligned expand_shape: "
              << elemType;
    }
    int64_t blk = (int64_t)*blkOr;

    int64_t M1 = M / alignM;
    // TODO: enhance UB alignment
    int64_t N1 = N / blk;
    auto dstTy = RankedTensorType::get({M, N1, blk}, elemType);
    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, dstTy, origTensor, reassociation);
    auto emptyTensorType = RankedTensorType::get({N1, M, blk}, elemType);
    auto emptyTransposed = rewriter.create<tensor::EmptyOp>(
        loc, emptyTensorType.getShape(), emptyTensorType.getElementType());
    SmallVector<int64_t> premVec = {1, 0, 2};
    auto transposed = rewriter.create<hivm::VTransposeOp>(
        loc, emptyTransposed->getResultTypes(), expandOp.getResult(),
        emptyTransposed.getResult(), rewriter.getDenseI64ArrayAttr(premVec));
    auto nzTy = RankedTensorType::get({N1, M1, alignM, blk}, elemType);
    SmallVector<ReassociationIndices> nzReassoc = {{0}, {1, 2}, {3}};
    auto nzOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, nzTy, transposed->getResult(0), nzReassoc);
    rewriter.modifyOpInPlace(consumerOp,
                              [&]() { consumerOperand->set(nzOp); });
    changed = true;
  }
  return changed ? success() : failure();
}

static bool isVFCall(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (callOp->hasAttr(hivm::VectorFunctionAttr::name))
      return true;
  }
  return false;
}

template<typename OpType>
struct AddConvertLayoutUBToL1
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  ~AddConvertLayoutUBToL1() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
  bool changed = false;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    Value beforeValue = operand->get();
    auto operandIdx = operand->getOperandNumber();
    auto inserted = getIntegersOfArrayAttr(op, insertedConvertLayout);
    if (std::find(inserted.begin(), inserted.end(), operandIdx) != inserted.end()) continue;
    auto producerOps = traceDefOps<OpType>(beforeValue);
    if (producerOps.empty())
      continue;

    bool matched = false;
    for (Operation *producer : producerOps) {
      if constexpr (std::is_same_v<OpType, mlir::scf::ForOp>) {
        auto scfForOp = llvm::cast<mlir::scf::ForOp>(producer);
        if (!scfForOp->hasAttr(ExtractLoadStoreAttr)) {
          continue;
        }
      }
      if constexpr (std::is_same_v<OpType, func::CallOp>) {
        if (!isVFCall(producer))
          continue;
      }
      if constexpr (std::is_same_v<OpType, memref::AllocOp>) {
        auto allocOp = llvm::cast<memref::AllocOp>(producer);
        auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(allocOp.getMemref().getType());
        if (!maybeSpace.has_value() || maybeSpace.value() != hivm::AddressSpace::UB)
          continue;
      }
      if constexpr (std::is_same_v<OpType, bufferization::ToTensorOp>) {
        auto toTensorOp = llvm::cast<bufferization::ToTensorOp>(producer);
        auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
            toTensorOp.getResult(), kMayImplicitTransposeWithLastAxis);
        if (!maybeAnnotateOp.has_value()) {
          maybeAnnotateOp = utils::getAnnotateOpWithAttr(
              toTensorOp.getMemref(), kMayImplicitTransposeWithLastAxis);
        }
        if (!maybeAnnotateOp.has_value())
          continue;
      }
      if constexpr (std::is_same_v<OpType, tensor::CollapseShapeOp>) {
        auto collapseShapeOp = llvm::cast<tensor::CollapseShapeOp>(producer);
        auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
            collapseShapeOp.getResult(), maybeUnCollapsibleReshape);
        if (!maybeAnnotateOp.has_value())
          continue;
      }
      matched = true;
      break;
    }
    if (!matched)
      continue;

    auto allocOps = traceDefOps<memref::AllocOp>(beforeValue);
    llvm::SmallVector<OpOperand *> consumerOperands{operand};
    bool isBiasOperand = (beforeValue == op.getPerChannelBias());
    auto tensorType = mlir::dyn_cast<RankedTensorType>(beforeValue.getType());
    // Only rank-2 tensors (original ND format) need to be transposed.
    // After ND2NZ conversion the tensor becomes rank-4 (NZ format), so
    // transposition is not required.
    bool needTransposed = (tensorType.getRank() == 2);
    if (isBiasOperand || !needTransposed)
      continue;
    LogicalResult result = insertConvertLayout(rewriter, op, consumerOperands);
    if (failed(result))
      continue;
    insertToIntegerAttrOfArrayAttr(op, insertedConvertLayout, operandIdx, rewriter);

    // Handle case where multiple operands of `op` are same SSA value.
    // e.g., `hivm.hir.mmadL1 ins(%vec, %vec, ...)` where both operands are the same vector.
    // In this case, we only want to rewrite the operand once, and make sure that other operand
    // is updated with the new value as well.
    Value converted = operand->get();
    rewriter.modifyOpInPlace(op, [&beforeValue, &converted, &op, &rewriter]() {
      // Use for loop to insert operand index for `insertedConvertLayout`
      for (OpOperand *operand : op.getDpsInputOperands()) {
        if(operand->get() != beforeValue)
          continue;
        op->setOperand(operand->getOperandNumber(), converted);
        insertToIntegerAttrOfArrayAttr(op, insertedConvertLayout, operand->getOperandNumber(), rewriter);
      }
    });

    for (Operation *allocOp : allocOps)
      ensureAllocInUbAddressSpaceIfNeeded(rewriter, llvm::cast<memref::AllocOp>(allocOp));
    changed = true;
  }
  return changed ? success() : failure();
}
};

template<>
struct AddConvertLayoutUBToL1<hivm::FixpipeOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  ~AddConvertLayoutUBToL1() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    for (OpOperand &operand : op->getOpOperands()) {
      auto operandIdx = operand.getOperandNumber();
      auto inserted = getIntegersOfArrayAttr(op, insertedConvertLayout);
      if (std::find(inserted.begin(), inserted.end(), operandIdx) != inserted.end()) continue;
      auto maybeFixpipe = traceDefOp<hivm::FixpipeOp>(operand.get());
      if (!maybeFixpipe)
        continue;
      llvm::SmallVector<OpOperand *> consumerOperands{&operand};

      Value valueAfterUb = operand.get();

      bool isBiasOperand = (operand.get() == op.getPerChannelBias());
      if (isBiasOperand)
        continue;
      LogicalResult l1Result = insertConvertLayout(rewriter, op, consumerOperands);
      if (failed(l1Result))
        continue;
      insertToIntegerAttrOfArrayAttr(op.getOperation(), insertedConvertLayout, operand.getOperandNumber(), rewriter);

      Value convertedToL1 = operand.get();
      if (valueAfterUb != convertedToL1) {
        rewriter.modifyOpInPlace(
            op, [&]() { op->replaceUsesOfWith(valueAfterUb, convertedToL1); });
      }
      changed = true;
    }
    return changed ? success() : failure();
  }
};

std::optional<hivm::MmadL1Op> getMmadL1OpFromUpPropOp(UnrealizedConversionCastOp downPropOp,
                                   UnrealizedConversionCastOp upPropOp,
                                   PatternRewriter &rewriter) {
  // trace uppropOp, check if it's mmadop
  Operation *cur = upPropOp.getOperation();
  while(isa<UnrealizedConversionCastOp>(cur)) {
    if (!cur->hasOneUse()) // unhandled case if not one user.
      return std::nullopt;
    cur = *cur->user_begin();
  }
  return dyn_cast<hivm::MmadL1Op>(cur);
}

} // namespace mlir::hivm

template <typename ...OpTypes>
void populateAddConvertLayoutUBToL1(RewritePatternSet &patterns) {
  (patterns.add<AddConvertLayoutUBToL1<OpTypes>>(patterns.getContext()), ...);
}

LogicalResult InsertLoadStoreForMixCVPass::addConvertLayoutUBToL1(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  populateAddConvertLayoutUBToL1<
    hivm::FixpipeOp,
    func::CallOp,
    mlir::scf::ForOp,
    hivm::IndirectLoadOp,
    hivm::StrideLoadOp,
    mlir::hivm::GatherLoadOp,
    mlir::scope::ScopeOp,
    tensor::CollapseShapeOp,
    bufferization::ToTensorOp,
    memref::AllocOp,
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
  >(patterns);
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }

  funcOp->walk([](hivm::MmadL1Op op) -> void {
    op->removeAttr(insertedConvertLayout);
  });
  return success();
}
void InsertLoadStoreForMixCVPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *ctx = &getContext();
  auto funcOp = getOperation();

  if (enableLegacy)
    return runLegacyInsertLoadStoreForMixCV();
  if (!hacc::utils::isDeviceEntry(funcOp))
    return;

  PassManager pm(ctx);
  pm.addPass(tensor::createReplicateOutEmptyTensorPass());
  if (enableLegacy || !isA5Target())
    return runLegacyInsertLoadStoreForMixCV();
  if (!hacc::utils::isDeviceEntry(funcOp))
    return;
  
  PassManager pm(ctx);

  if (isEnabledTightCoupledBuffer() && failed(addConvertLayoutUBToL1(funcOp))) {
    return signalPassFailure();
  }

  if (failed(pm.run(funcOp))) {
    return signalPassFailure();
  }
  // To maintain propagator, rewriteConfig.fold should be false
  if (failed(insertPropagationOp(funcOp))) {
    return signalPassFailure();
  }
  LDBG("After inserting Propagation Op: " << funcOp);
  if (failed(propagateAndResolve(funcOp))) {
    return signalPassFailure();
  }
  LDBG("After propagation & resolve: " << funcOp);
  LLVM_DEBUG({
    if (failed(verifyPropagation(funcOp))) {
      return signalPassFailure();
    }
  });

  /// Postprocess of adding core type attribute
  funcOp->walk([](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](hivm::DebugOp op) {
          auto upProp = PropagatorUtil::getUpPropagator(&op.getArgMutable());
          if (upProp) {
            auto coreType = PropagatorUtil::getCoreType(upProp);
            auto addressSpaces = PropagatorUtil::getAddressSpace(upProp);
            if (!addressSpaces.empty()) {
              auto memScopeAttr = hivm::AddressSpaceAttr::get(op.getContext(),
                                                              addressSpaces[0]);
              auto memScopeAttr = hivm::AddressSpaceAttr::get(
                  op.getContext(), addressSpaces[0]);
              op.setMemscopeAttr(memScopeAttr);
            }
            if (coreType != TCoreType::CUBE_AND_VECTOR) {
              op.setTcoretypeAttr(
                  TCoreTypeAttr::get(op.getContext(), coreType));
            } else {
              auto addressSpace = addressSpaces.empty() ? hivm::AddressSpace::UB
                                                        : addressSpaces[0];
              op.setTcoretypeAttr(TCoreTypeAttr::get(
                  op.getContext(),
                  PropagatorUtil::kAddressSpace2CoreType.at(addressSpace)));
            }
          }
        })
        .Case([&](hivm::LoadOp op) {
          auto upProp = PropagatorUtil::getUpPropagator(&op.getDstMutable());
          if (upProp) {
            auto coreType = PropagatorUtil::getCoreType(upProp);
            if (coreType != TCoreType::CUBE_AND_VECTOR) {
              op.setTcoretypeAttr(
                  TCoreTypeAttr::get(op.getContext(), coreType));
            } else {
              auto addressSpaces = PropagatorUtil::getAddressSpace(upProp);
              auto addressSpace = addressSpaces.empty() ? hivm::AddressSpace::UB
                                                        : addressSpaces[0];
              op.setTcoretypeAttr(TCoreTypeAttr::get(
                  op.getContext(),
                  PropagatorUtil::kAddressSpace2CoreType.at(addressSpace)));
            }
          }
        });
  });

  PassManager pm2(ctx);
  pm2.addPass(bishengir::createExtendedCanonicalizerPass());
  if (failed(pm2.run(funcOp))) {
    return signalPassFailure();
  }
  pm2.addPass(createCanonicalizerPass());
  if (failed(pm2.run(funcOp))) {
    return signalPassFailure();
  }

  PassManager pm3(ctx);
  pm3.addPass(createInsertLoadStoreForScalarPass());
  if (failed(pm3.run(funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertLoadStoreForMixCVPass(
    const InsertLoadStoreForMixCVOptions &options) {
  return std::make_unique<InsertLoadStoreForMixCVPass>(options);
}
