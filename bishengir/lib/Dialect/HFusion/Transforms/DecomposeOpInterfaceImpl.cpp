//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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

#include "bishengir/Dialect/HFusion/Transforms/DecomposeOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace hfusion;

namespace {

// Convert multi-axis linalg.transpose to binary linalg.transpose
// linalg.tranpose(tensor(2x16x8x4x3xf32)) -> tensor(2x3x4x8x16xf32)
// permutation = [0,4,3,2,1] is normalized into
// tensor(2x16x8x4x3xf32) -> tensor(2x3x8x4x16xf32)  perm = [0,4,2,3,1]
// tensor(2x3x8x4x16xf32) -> tensor(2x3x4x8x16xf32)  perm = [0,1,3,2,4]
struct TransposeDecomposeInterface
    : public bishengir::BiShengIRAggregatedOpInterface::ExternalModel<
          TransposeDecomposeInterface, linalg::TransposeOp> {
  bool needDecompose(ArrayRef<int64_t> arr) const {
    int mismatch = 0;
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
      if (arr[i] != i)
        mismatch++;
    }
    return (mismatch > 2);
  }

  void calculateMinSwaps(ArrayRef<int64_t> perm,
                         SmallVector<SmallVector<int64_t, 2>> &swaps) const {
    int64_t N = static_cast<int64_t>(perm.size());
    SmallVector<std::pair<int64_t, int64_t>, 8> permIndexVec(N);
    for (int64_t i = 0; i < N; i++)
      permIndexVec[i] = {perm[i], i};

    std::sort(permIndexVec.begin(), permIndexVec.end());

    for (int64_t i = 0; i < N; i++) {
      if (permIndexVec[i].second == i)
        continue;
      swaps.push_back({i, permIndexVec[i].second});
      std::swap(permIndexVec[i], permIndexVec[permIndexVec[i].second]);
      i--;
    }
  }

  Value decomposeTransposeOp(linalg::TransposeOp op, OpBuilder &rewriter,
                             SmallVector<SmallVector<int64_t, 2>> swaps,
                             int64_t length) const {
    Value inputVal = op.getInput();
    auto inputTensor = dyn_cast<TensorType>(inputVal.getType());
    Type targetElemType = getElementTypeOrSelf(inputTensor);

    for (auto &swap : swaps) {
      auto curInputStaticShape =
          cast<ShapedType>(inputVal.getType()).getShape();
      SmallVector<int64_t> curOutputStaticShape;
      SmallVector<Value> curOutputDynamicShape;

      int64_t idx1 = swap[0], idx2 = swap[1];
      SmallVector<int64_t> tempPerm;
      // populate intermediate permutations and shapes
      for (int64_t i = 0; i < length; ++i) {
        int64_t targetIdx;
        if (i == idx1) {
          targetIdx = idx2;
        } else if (i == idx2) {
          targetIdx = idx1;
        } else {
          targetIdx = i;
        }

        tempPerm.push_back(targetIdx);
        if (curInputStaticShape[targetIdx] == ShapedType::kDynamic) {
          Operation *dynDimOp =
              rewriter.create<tensor::DimOp>(op->getLoc(), inputVal, targetIdx);
          curOutputStaticShape.push_back(ShapedType::kDynamic);
          curOutputDynamicShape.push_back(dynDimOp->getResults()[0]);
        } else {
          curOutputStaticShape.push_back(curInputStaticShape[targetIdx]);
        }
      }
      // create empty tensor holding the intermediate result shape
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), curOutputStaticShape, targetElemType,
          curOutputDynamicShape);
      auto intermediateTransposeOp = rewriter.create<linalg::TransposeOp>(
          op->getLoc(), inputVal, emptyTensor, tempPerm);
      inputVal = intermediateTransposeOp->getResult(0);
    }

    return inputVal;
  }

  FailureOr<SmallVector<Value>> decomposeOperation(Operation *op,
                                                   OpBuilder &rewriter) const {
    auto transposeOp = llvm::dyn_cast<mlir::linalg::TransposeOp>(op);
    if (!transposeOp)
      return failure();

    auto perm = transposeOp.getPermutation();
    // skip binary transpose
    if (!needDecompose(perm))
      return failure();

    // the order of swaps to be proceeded
    SmallVector<SmallVector<int64_t, 2>> swaps;
    calculateMinSwaps(perm, swaps);

    // Create tensor.empty and decomposed linalg.transpose Ops
    return SmallVector<Value>{
        decomposeTransposeOp(transposeOp, rewriter, swaps, perm.size())};
  }

  bishengir::DecomposePhase getDecomposePhase(Operation *op) const {
    return bishengir::DecomposePhase::AFTER_HFUSION_FLATTEN;
  }
};

// SortDecomposeInterface implements the decomposition logic for hfusion.sort.
// It lowers the high-level sort operation into nested loops (scf.for) using 
// a bubble sort algorithm specifically targeting the last axis of the tensor.
struct SortDecomposeInterface
    : public bishengir::BiShengIRAggregatedOpInterface::ExternalModel<
          SortDecomposeInterface, hfusion::SortOp> {
private:
  // Implements sorting logic on the innermost dimension using nested scf.for loops.
  Value emitSortOnLastAxis(OpBuilder &rewriter, Location loc, Value tensor,
                           ArrayRef<Value> outerIndices, int64_t innerDim,
                           Type elemTy, bool descending) const {
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value innerBoundMinus1 =
        rewriter.create<arith::ConstantIndexOp>(loc, innerDim - 1);

    // Bubble sort outer pass loop (n-1 passes).
    auto passFor = rewriter.create<scf::ForOp>(
        loc, c0, innerBoundMinus1, c1, ValueRange{tensor});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(passFor.getBody());

      Value passTensor = passFor.getRegionIterArgs()[0];

      // Generate loop to compare and swap adjacent elements.
      auto innerFor = rewriter.create<scf::ForOp>(
          loc, c0, innerBoundMinus1, c1, ValueRange{passTensor});

      {
        OpBuilder::InsertionGuard guard2(rewriter);
        rewriter.setInsertionPointToStart(innerFor.getBody());

        Value j = innerFor.getInductionVar();
        Value curTensor = innerFor.getRegionIterArgs()[0];
        Value jNext = rewriter.create<arith::AddIOp>(loc, j, c1);

        // Build full indices for the two elements to be compared.
        SmallVector<Value> lhsIndices(outerIndices.begin(), outerIndices.end());
        lhsIndices.push_back(j);

        SmallVector<Value> rhsIndices(outerIndices.begin(), outerIndices.end());
        rhsIndices.push_back(jNext);

        Value lhs =
            rewriter.create<tensor::ExtractOp>(loc, curTensor, lhsIndices);
        Value rhs =
            rewriter.create<tensor::ExtractOp>(loc, curTensor, rhsIndices);

        // Generate comparison based on element type and sort direction.
        Value needSwap;
        if (mlir::isa<FloatType>(elemTy)) {
          // Use Unordered predicates (ULT/UGT) to ensure NaN values are 
          // consistently pushed to the end of the sequence.
          needSwap = descending
                         ? rewriter.create<arith::CmpFOp>(
                               loc, arith::CmpFPredicate::ULT, lhs, rhs)
                         : rewriter.create<arith::CmpFOp>(
                               loc, arith::CmpFPredicate::UGT, lhs, rhs);
        } else {
          // Standard signed comparison for integers.
          needSwap = descending
                         ? rewriter.create<arith::CmpIOp>(
                               loc, arith::CmpIPredicate::slt, lhs, rhs)
                         : rewriter.create<arith::CmpIOp>(
                               loc, arith::CmpIPredicate::sgt, lhs, rhs);
        }

        // Conditionally swap elements based on the comparison result.
        auto ifOp = rewriter.create<scf::IfOp>(
            loc, TypeRange{curTensor.getType()}, needSwap,
            /*withElseRegion=*/true);

        // Then block: Swap elements and update tensor state.
        {
          Block *thenBlock = &ifOp.getThenRegion().front();
          OpBuilder thenBuilder = OpBuilder::atBlockBegin(thenBlock);

          Value t0 = thenBuilder.create<tensor::InsertOp>(loc, rhs, curTensor,
                                                          lhsIndices);
          Value t1 = thenBuilder.create<tensor::InsertOp>(loc, lhs, t0,
                                                          rhsIndices);
          thenBuilder.create<scf::YieldOp>(loc, t1);
        }

        // Else block: Maintain original tensor state.
        {
          Block *elseBlock = &ifOp.getElseRegion().front();
          OpBuilder elseBuilder = OpBuilder::atBlockBegin(elseBlock);
          elseBuilder.create<scf::YieldOp>(loc, curTensor);
        }

        rewriter.create<scf::YieldOp>(loc, ifOp.getResults());
      }

      rewriter.setInsertionPointAfter(innerFor);
      rewriter.create<scf::YieldOp>(loc, innerFor.getResults());
    }

    return passFor.getResult(0);
  }

  // Recursively build loops to traverse other dimensions.
  Value emitOuterLoops(OpBuilder &rewriter, Location loc, Value tensor,
                       ArrayRef<int64_t> shape, Type elemTy, bool descending,
                       int64_t dim, SmallVector<Value> &outerIndices) const {
    int64_t rank = static_cast<int64_t>(shape.size());

    // Base case: Sort along the last axis when all outer loops are built.
    if (dim == rank - 1)
      return emitSortOnLastAxis(rewriter, loc, tensor, outerIndices,
                                shape.back(), elemTy, descending);
                                
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value upper = rewriter.create<arith::ConstantIndexOp>(loc, shape[dim]);

    // Create a loop for the current dimension.
    auto outerFor = rewriter.create<scf::ForOp>(loc, c0, upper, c1,
                                                ValueRange{tensor});

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());

      Value iv = outerFor.getInductionVar();
      Value curTensor = outerFor.getRegionIterArgs()[0];

      // Store induction variable to maintain the coordinate stack.
      outerIndices.push_back(iv);
      Value updated = emitOuterLoops(rewriter, loc, curTensor, shape, elemTy,
                                     descending, dim + 1, outerIndices);
      outerIndices.pop_back();

      rewriter.create<scf::YieldOp>(loc, updated);
    }

    return outerFor.getResult(0);
  }

public:
  // Main entry for Sort operator decomposition.
  FailureOr<SmallVector<Value>> decomposeOperation(Operation *op,
                                                   OpBuilder &rewriter) const {
    auto sortOp = dyn_cast<hfusion::SortOp>(op);
    if (!sortOp)
      return failure();

    Location loc = op->getLoc();
    Value src = sortOp.getSrc();

    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTy)
      return failure();

    if (sortOp->getNumResults() != 1)
      return failure();

    int64_t rank = srcTy.getRank();
    if (rank <= 0)
      return failure();

    int64_t axis = sortOp.getSignedSortAxis();
    bool descending = sortOp.getDescending();

    // Constraint: Currently only support sorting on the last axis.
    if (axis < 0 || axis != rank - 1)
      return failure();

    ArrayRef<int64_t> shape = srcTy.getShape();

    // Verify static shapes and valid element types.
    for (int64_t d : shape) {
      if (d == ShapedType::kDynamic)
        return failure();
    }

    int64_t innerDim = shape.back();
    if (innerDim <= 0)
      return failure();

    Type elemTy = srcTy.getElementType();
    if (!mlir::isa<FloatType>(elemTy) && !mlir::isa<IntegerType>(elemTy))
      return failure();

    // Handle length-1 dimension as a no-op.
    if (innerDim == 1)
      return SmallVector<Value>{src};

    // Start recursive loop generation.
    SmallVector<Value> outerIndices;
    Value result =
        emitOuterLoops(rewriter, loc, src, shape, elemTy, descending,
                       /*dim=*/0, outerIndices);

    return SmallVector<Value>{result};
  }

  bishengir::DecomposePhase getDecomposePhase(Operation *op) const {
    return bishengir::DecomposePhase::BEFORE_LOWER_TO_LOOPS;
  }
};


} // namespace

void mlir::hfusion::registerDecomposeInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::TransposeOp::attachInterface<TransposeDecomposeInterface>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, hfusion::HFusionDialect *dialect) {
    hfusion::SortOp::attachInterface<SortDecomposeInterface>(*ctx);
  });

}
