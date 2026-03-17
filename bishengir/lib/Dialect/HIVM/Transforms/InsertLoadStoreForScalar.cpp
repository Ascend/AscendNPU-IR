//===-------------------- InsertLoadStoreForScalar.cpp-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass handles special cases for tensor.extract when consumed by cube ops.
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
 
namespace mlir {
#define GEN_PASS_DEF_INSERTLOADSTOREFORSCALAR
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-insert-load-store-for-scalar"

namespace {

using namespace mlir;
using namespace mlir::hivm;

struct InsertLoadStoreForScalarPass
    : public impl::InsertLoadStoreForScalarBase<InsertLoadStoreForScalarPass> {
  using Base::Base;
  void runOnOperation() override;
};

template <typename... OpTypes>
static bool traceVector(Value v) {
  return ((traceDefOp<OpTypes>(v) != std::nullopt) || ...);
}
 
//===----------------------------------------------------------------------===//
// DuplicateTensorExtractForCube
//===----------------------------------------------------------------------===//
/// Handles tensor.extract ops that feed into cube operations.
/// Inserts explicit store/load to ensure data movement between vector and cube cores.
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
        mlir::hivm::TCoreTypeAttr::name,
        mlir::hivm::TCoreTypeAttr::get(markOp->getContext(), tCoreType));
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
 
      // Check current operation and its nested operations
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
      
      // Add all users of currentOp to worklist for indirect user checking
      for (auto result : currentOp->getResults()) {
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
 
    // only process cases with vector sources or direct load
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
    bool originCoreTypeIsVector = traceVector<
    #define GET_OP_LIST
    #include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(originTensor);
    if (!originCoreTypeIsVector) {
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
 
void InsertLoadStoreForScalarPass::runOnOperation() {
  auto funcOp = getOperation();
  auto *context = &getContext();
  RewritePatternSet patterns(context);
  // Only process functions containing cube operations
  bool hasCube = false;
  funcOp.walk([&hasCube](Operation *op) {
    if (isa<hivm::MmadL1Op>(op)) {
      hasCube = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
 
  if (hasCube) {
    patterns.insert<DuplicateTensorExtractForCube>(patterns.getContext());
  }
 
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
} // anonymous namespace
std::unique_ptr<Pass> mlir::hivm::createInsertLoadStoreForScalarPass() {
  return std::make_unique<InsertLoadStoreForScalarPass>();
}