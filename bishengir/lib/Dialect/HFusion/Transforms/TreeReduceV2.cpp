//===----------------------- TreeReduceV2.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Transforms.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <numeric>

namespace mlir {
#define GEN_PASS_DEF_TREEREDUCEV2
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "tree-reduce-v2"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

template <typename IntOp, typename FloatOp>
bool isValidTreeReducePattern(linalg::GenericOp op,
                              SmallVector<int> &ReductionIndex, Block &body,
                              bool isNotMulti) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
    return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.empty())
    return false;

  for (size_t i = 0; i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::reduction)
      ReductionIndex.push_back(i);
  }
  if (ReductionIndex.empty() || ((ReductionIndex.size() == 1) ^ isNotMulti))
    return false;
  if (!isa<IntOp, FloatOp>(body.front()) || body.getOperations().size() > 2)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!inputType || !all_of(inputType.getShape(), [](int64_t sz) {
        return !ShapedType::isDynamic(sz);
      }))
    return false;
  if (inputType.getElementType().getIntOrFloatBitWidth() == 1)
    return false;
  return true;
}

TypedAttr createFillAttr(Operation &frontOp, RankedTensorType inputType,
                         PatternRewriter &rewriter, int addType) {
  auto elemType = inputType.getElementType();
  if (addType == 0) {
    if (isa<arith::AddIOp>(frontOp))
      return rewriter.getIntegerAttr(elemType, 0);
    if (isa<arith::MaxSIOp>(frontOp))
      return rewriter.getIntegerAttr(
          elemType,
          llvm::APInt::getSignedMinValue(elemType.getIntOrFloatBitWidth()));
    if (isa<arith::MinSIOp>(frontOp))
      return rewriter.getIntegerAttr(
          elemType,
          llvm::APInt::getSignedMaxValue(elemType.getIntOrFloatBitWidth()));
  } else {
    auto &sem = dyn_cast<mlir::FloatType>(elemType).getFloatSemantics();
    if (isa<arith::AddFOp>(frontOp))
      return rewriter.getFloatAttr(elemType, 0.0);
    if (isa<arith::MaxNumFOp>(frontOp))
      return rewriter.getFloatAttr(elemType, llvm::APFloat::getInf(sem, true));
    if (isa<arith::MinNumFOp>(frontOp))
      return rewriter.getFloatAttr(elemType, llvm::APFloat::getInf(sem, false));
  }
  llvm::report_fatal_error("not complete yet");
}

int calculateCurrentReductionDim(int reductionDim,
                                 SmallVector<int> &beenReductionDim) {
  int currentDim = reductionDim;
  for (int prev : beenReductionDim) {
    if (prev < reductionDim)
      currentDim--;
  }
  return currentDim;
}

template <typename IntOp, typename FloatOp>
void splitMultiReduction(PatternRewriter &rewriter, linalg::GenericOp op,
                         RankedTensorType &currentType, Value &currentResult,
                         SmallVector<int> &ReductionIndex,
                         SmallVector<int> &beenReductionDim, Value output,
                         Block &body, int addType) {
  for (size_t idx = 0; idx < ReductionIndex.size(); ++idx) {
    int reductionDim = ReductionIndex[idx];
    int currentReductionDim =
        calculateCurrentReductionDim(reductionDim, beenReductionDim);

    SmallVector<int64_t> intermediateShape(currentType.getShape().begin(),
                                           currentType.getShape().end());
    intermediateShape.erase(intermediateShape.begin() + currentReductionDim);
    auto intermediateType =
        RankedTensorType::get(intermediateShape, currentType.getElementType());
    Value initTensor;

    if (idx == ReductionIndex.size() - 1) {
      initTensor = output;
      if (currentType.getShape().back() == 1) {
        SmallVector<ReassociationExprs, 4> reassociationMap(
            currentType.getRank() - 1);
        for (int i = 0; i < currentType.getRank() - 1; i++)
          reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
        reassociationMap.back().push_back(
            rewriter.getAffineDimExpr(currentType.getRank() - 1));
        currentResult = rewriter.create<tensor::CollapseShapeOp>(
            op.getLoc(), currentResult, reassociationMap);
        currentType = intermediateType;
        beenReductionDim.push_back(reductionDim);
        break;
      }
    } else {
      initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), intermediateShape, currentType.getElementType());
      TypedAttr zeroAttr =
          createFillAttr(body.front(), currentType, rewriter, addType);
      Value zeroVal = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
      initTensor = rewriter
                       .create<linalg::FillOp>(op.getLoc(), ValueRange{zeroVal},
                                               ValueRange{initTensor})
                       .getResult(0);
    }

    SmallVector<AffineMap> indexingMaps = {AffineMap::getMultiDimIdentityMap(
        currentType.getRank(), rewriter.getContext())};

    SmallVector<AffineExpr> outputExprs;
    for (int64_t i = 0; i < currentType.getRank(); ++i) {
      if (i != currentReductionDim)
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    indexingMaps.push_back(AffineMap::get(currentType.getRank(), 0, outputExprs,
                                          rewriter.getContext()));

    SmallVector<utils::IteratorType> iterTypes(currentType.getRank(),
                                               utils::IteratorType::parallel);
    iterTypes[currentReductionDim] = utils::IteratorType::reduction;

    auto reduceOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), intermediateType, ValueRange{currentResult},
        ValueRange{initTensor}, indexingMaps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result =
              addType == 0
                  ? b.create<IntOp>(loc, args[0], args[1]).getResult()
                  : b.create<FloatOp>(loc, args[0], args[1]).getResult();
          b.create<linalg::YieldOp>(loc, result);
        });

    currentResult = reduceOp.getResult(0);
    currentType = intermediateType;
    beenReductionDim.push_back(reductionDim);
  }
}

template <typename IntOp, typename FloatOp>
void convertStridedReduceToMultiReduce(func::FuncOp func,
                                       PatternRewriter &rewriter) {
  auto originalInsertPoint = rewriter.saveInsertionPoint();
  SmallVector<linalg::GenericOp> opList;
  func.walk([&](linalg::GenericOp genericOp) { opList.push_back(genericOp); });

  for (auto op : opList) {
    rewriter.setInsertionPoint(op);
    SmallVector<int> ReductionIndex;
    Block &body = op.getRegion().front();
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body,
                                                  false))
      continue;

    int addType = isa<FloatOp>(body.front());
    Value currentResult = op.getInputs()[0];
    auto currentType = cast<RankedTensorType>(currentResult.getType());
    llvm::sort(ReductionIndex);

    SmallVector<int> beenReductionDim;
    rewriter.setInsertionPointAfter(op);
    splitMultiReduction<IntOp, FloatOp>(
        rewriter, op, currentType, currentResult, ReductionIndex,
        beenReductionDim, op.getOutputs()[0], body, addType);
    op->getResult(0).replaceAllUsesWith(currentResult);
    op->erase();
  }
  rewriter.restoreInsertionPoint(originalInsertPoint);
}

Value padReductionInput(PatternRewriter &rewriter, Location loc, Value input,
                        int64_t reductionDim, bool addType,
                        Operation *reductionOp) {
  RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> newShape(inputType.getShape().begin(),
                                inputType.getShape().end());
  newShape[reductionDim] += 1;

  Value paddedInput = rewriter.create<tensor::EmptyOp>(
      loc, newShape, inputType.getElementType());
  TypedAttr zeroAttr =
      createFillAttr(*reductionOp, inputType, rewriter, addType);
  Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  paddedInput = rewriter
                    .create<linalg::FillOp>(loc, ValueRange{zeroVal},
                                            ValueRange{paddedInput})
                    .getResult(0);

  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    rewriter.getIndexAttr(1));
  for (int64_t dimSize : inputType.getShape())
    sizes.push_back(rewriter.getIndexAttr(dimSize));

  Value inputSubview = rewriter.create<tensor::ExtractSliceOp>(
      loc, input, offsets, sizes, strides);
  return rewriter.create<tensor::InsertSliceOp>(loc, inputSubview, paddedInput,
                                                offsets, sizes, strides);
}

template <typename IntOp, typename FloatOp>
Value handleLargeChunkCase(
    PatternRewriter &rewriter, Location loc, Value input,
    SmallVector<OpFoldResult> offsets, SmallVector<OpFoldResult> sizes,
    SmallVector<OpFoldResult> strides, Value acc, Value corresponding_chunk,
    SmallVector<AffineMap> indexingMaps_reduce,
    SmallVector<utils::IteratorType> iteratorTypes_reduce,
    SmallVector<utils::IteratorType> iteratorTypes, RankedTensorType inputType,
    int addType, bool needMerge, Operation *reductionOp) {
  auto tmpType = dyn_cast<RankedTensorType>(corresponding_chunk.getType());
  auto accType = dyn_cast<RankedTensorType>(acc.getType());
  Value initTensor;
  int count = 0;
  bool needInsert = false;

  if (accType.getRank() == 0) {
    Value empty = rewriter.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{},
                                                   tmpType.getElementType());
    TypedAttr zeroAttr =
        createFillAttr(*reductionOp, tmpType, rewriter, addType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    initTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{empty})
            .getResult(0);
  } else {
    SmallVector<int64_t> newShape(tmpType.getShape().begin(),
                                  tmpType.getShape().end());
    for (auto iteratorType : iteratorTypes_reduce) {
      if (iteratorType == utils::IteratorType::reduction) {
        newShape.erase(newShape.begin() + count);
        break;
      }
      count++;
    }
    needInsert = (newShape != accType.getShape());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, newShape,
                                                   tmpType.getElementType());
    TypedAttr zeroAttr =
        createFillAttr(*reductionOp, tmpType, rewriter, addType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    initTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{empty})
            .getResult(0);
  }

  auto current_chunk = rewriter.create<tensor::ExtractSliceOp>(
      loc, input, offsets, sizes, strides);
  SmallVector<AffineMap> indexingMaps(
      3, rewriter.getMultiDimIdentityMap(inputType.getRank()));

  auto added = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{corresponding_chunk.getType()},
      ValueRange{current_chunk.getResult(), corresponding_chunk},
      ValueRange{current_chunk.getResult()}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value sum =
            addType == 0
                ? nestedBuilder
                      .create<IntOp>(nestedLoc, blockArgs[0], blockArgs[1])
                      .getResult()
                : nestedBuilder
                      .create<FloatOp>(nestedLoc, blockArgs[0], blockArgs[1])
                      .getResult();
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });

  if (needMerge) {
    Value targetAcc = needInsert ? initTensor : acc;
    added = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{targetAcc.getType()}, ValueRange{added.getResult(0)},
        ValueRange{targetAcc}, indexingMaps_reduce, iteratorTypes_reduce,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value sum =
              addType == 0
                  ? nestedBuilder
                        .create<IntOp>(nestedLoc, blockArgs[0], blockArgs[1])
                        .getResult()
                  : nestedBuilder
                        .create<FloatOp>(nestedLoc, blockArgs[0], blockArgs[1])
                        .getResult();
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
        });

    if (needInsert) {
      offsets.erase(offsets.begin() + count);
      sizes.erase(sizes.begin() + count);
      strides.erase(strides.begin() + count);
      return rewriter
          .create<tensor::InsertSliceOp>(loc, added.getResult(0), acc, offsets,
                                         sizes, strides)
          .getResult();
    }
  }
  return added.getResult(0);
}

template <typename IntOp, typename FloatOp>
void convertLinalgToTreeReduce(func::FuncOp func, PatternRewriter &rewriter) {
  auto originalInsertPoint = rewriter.saveInsertionPoint();
  SmallVector<linalg::GenericOp> opList;
  func.walk([&](linalg::GenericOp op) { opList.push_back(op); });

  for (auto op : opList) {
    rewriter.setInsertionPoint(op);
    SmallVector<int> ReductionIndex;
    Block &body = op.getRegion().front();
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body,
                                                  true))
      continue;

    int addType = isa<FloatOp>(body.front());
    Value input = op.getInputs()[0];
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t vectorSize =
        inputType.getElementType().getIntOrFloatBitWidth() == 64
            ? 64
            : 64 * 32 /
                  static_cast<int64_t>(
                      inputType.getElementType().getIntOrFloatBitWidth());
    int64_t dim0Value = inputType.getShape()[ReductionIndex[0]];

    if (dim0Value <= vectorSize)
      continue;

    if (inputType.getRank() - 1 == ReductionIndex[0]) {
      auto insertPoint = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfterValue(input);
      int remaining_value =
          2 * 32 / (inputType.getElementType().getIntOrFloatBitWidth() / 8);
      if (inputType.getShape().back() % remaining_value) {
        SmallVector<int64_t> newShape(inputType.getShape().begin(),
                                      inputType.getShape().end());
        newShape.back() = (newShape.back() + remaining_value - 1) /
                          remaining_value * remaining_value;
        auto resultType =
            RankedTensorType::get(newShape, inputType.getElementType());

        Value empty = rewriter.create<tensor::EmptyOp>(
            op.getLoc(), resultType.getShape(), resultType.getElementType());
        TypedAttr zeroAttr =
            createFillAttr(body.front(), inputType, rewriter, addType);
        Value zeroVal =
            rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
        Value initTensor =
            rewriter
                .create<linalg::FillOp>(op.getLoc(), ValueRange{zeroVal},
                                        ValueRange{empty})
                .getResult(0);

        SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(inputType.getRank(),
                                          rewriter.getIndexAttr(1));
        for (int64_t dimSize : inputType.getShape())
          sizes.push_back(rewriter.getIndexAttr(dimSize));

        Value paddedInput = rewriter.create<tensor::InsertSliceOp>(
            op.getLoc(), input, initTensor, offsets, sizes, strides);
        rewriter.replaceUsesWithIf(
            input, paddedInput, [&](OpOperand &opOperand) {
              return opOperand.getOwner() != paddedInput.getDefiningOp();
            });

        input = paddedInput;
        inputType = resultType;
        dim0Value = inputType.getShape()[ReductionIndex[0]];
      }
      rewriter.restoreInsertionPoint(insertPoint);
    }

    Location loc = op.getLoc();
    if (dim0Value % 2 != 0) {
      input = padReductionInput(rewriter, loc, input, ReductionIndex[0],
                                addType, &body.front());
      inputType = cast<RankedTensorType>(input.getType());
      dim0Value = inputType.getShape()[ReductionIndex[0]];
    }

    int64_t dimXValue = inputType.getShape().back();
    int64_t half_point_value = inputType.getRank() - 1 == ReductionIndex[0]
                                   ? (dimXValue + 1) / 2
                                   : dimXValue;
    Value acc = op.getOutputs()[0];

    SmallVector<OpFoldResult> offsets, offsets2, sizes, sizes2, strides;
    SmallVector<utils::IteratorType> iteratorTypes, iteratorTypes_reduce;

    for (int64_t dim = 0; dim < inputType.getRank(); dim++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
      iteratorTypes.push_back(utils::IteratorType::parallel);

      if (dim == ReductionIndex[0]) {
        sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[dim] / 2));
        sizes2.push_back(rewriter.getIndexAttr(inputType.getShape()[dim] / 2));
        offsets2.push_back(
            rewriter.getIndexAttr(inputType.getShape()[dim] / 2));
        iteratorTypes_reduce.push_back(utils::IteratorType::reduction);
      } else {
        sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[dim]));
        sizes2.push_back(rewriter.getIndexAttr(inputType.getShape()[dim]));
        offsets2.push_back(rewriter.getIndexAttr(0));
        iteratorTypes_reduce.push_back(utils::IteratorType::parallel);
      }
    }

    for (int64_t i = 0; i <= 1; i++) {
      int64_t current_start_value =
          (i == 0 ? 0
           : half_point_value < vectorSize
               ? half_point_value
               : half_point_value / vectorSize * vectorSize);
      int64_t remaining_value = half_point_value - current_start_value;
      if (!remaining_value)
        break;

      int64_t chunk_size_value = std::min(
          remaining_value,
          std::max(vectorSize, half_point_value / vectorSize * vectorSize));
      int64_t corresponding_start_value =
          current_start_value +
          (inputType.getRank() - 1 == ReductionIndex[0] ? half_point_value : 0);
      int64_t corresponding_size_value =
          std::max((int64_t)0, std::min(chunk_size_value,
                                        dimXValue - corresponding_start_value));

      offsets.back() = rewriter.getIndexAttr(current_start_value);
      offsets2.back() = rewriter.getIndexAttr(corresponding_start_value);
      sizes.back() = rewriter.getIndexAttr(corresponding_size_value);

      SmallVector<AffineMap> indexingMaps_reduce = {
          rewriter.getMultiDimIdentityMap(inputType.getRank()),
          AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                            rewriter.getContext())
              .dropResult(ReductionIndex[0])};

      Value corresponding_chunk = rewriter.create<tensor::ExtractSliceOp>(
          loc, input, offsets2, sizes, strides);
      bool needMerge = (cast<RankedTensorType>(corresponding_chunk.getType())
                            .getShape()[ReductionIndex[0]] != 0);

      acc = handleLargeChunkCase<IntOp, FloatOp>(
          rewriter, loc, input, offsets, sizes, strides, acc,
          corresponding_chunk, indexingMaps_reduce, iteratorTypes_reduce,
          iteratorTypes, inputType, addType, needMerge, &body.front());
    }
    op->getResult(0).replaceAllUsesWith(acc);
    op->erase();
  }
  rewriter.restoreInsertionPoint(originalInsertPoint);
}

struct TreeReducePattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
    if (!hacc::utils::isDevice(func) ||
        (fusionKind.has_value() &&
         (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
          fusionKind.value() == mlir::hfusion::FusionKind::SingleCube))) {
      return failure();
    }
    convertStridedReduceToMultiReduce<arith::AddIOp, arith::AddFOp>(func,
                                                                    rewriter);
    convertStridedReduceToMultiReduce<arith::MaxSIOp, arith::MaxNumFOp>(
        func, rewriter);
    convertStridedReduceToMultiReduce<arith::MinSIOp, arith::MinNumFOp>(
        func, rewriter);
    convertLinalgToTreeReduce<arith::AddIOp, arith::AddFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MaxSIOp, arith::MaxNumFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MinSIOp, arith::MinNumFOp>(func, rewriter);
    return success();
  }
};

void collectOriginalTensorCandidates(Value value, llvm::DenseSet<Value> &seen,
                                     SmallVectorImpl<Value> &candidates) {
  if (!seen.insert(value).second)
    return;

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    if (isa<RankedTensorType>(value.getType()))
      candidates.push_back(value);
    return;
  }

  if (auto maskOp = dyn_cast<vector::MaskOp>(defOp)) {
    Operation *maskedOp = maskOp.getMaskableOp();
    if (auto readOp = dyn_cast_or_null<vector::TransferReadOp>(maskedOp)) {
      collectOriginalTensorCandidates(readOp.getSource(), seen, candidates);
      return;
    }
    if (auto writeOp = dyn_cast_or_null<vector::TransferWriteOp>(maskedOp)) {
      collectOriginalTensorCandidates(writeOp.getVector(), seen, candidates);
      return;
    }
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(defOp)) {
    collectOriginalTensorCandidates(readOp.getSource(), seen, candidates);
    return;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(defOp)) {
    collectOriginalTensorCandidates(writeOp.getVector(), seen, candidates);
    return;
  }
  if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
    collectOriginalTensorCandidates(extractOp.getSource(), seen, candidates);
    return;
  }
  if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(defOp)) {
    collectOriginalTensorCandidates(insertOp.getSource(), seen, candidates);
    return;
  }
  if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(defOp)) {
    collectOriginalTensorCandidates(shapeCastOp.getSource(), seen, candidates);
    return;
  }
  if (auto extfOp = dyn_cast<arith::ExtFOp>(defOp)) {
    collectOriginalTensorCandidates(extfOp.getIn(), seen, candidates);
    return;
  }
  if (auto truncfOp = dyn_cast<arith::TruncFOp>(defOp)) {
    collectOriginalTensorCandidates(truncfOp.getIn(), seen, candidates);
    return;
  }
  if (auto extiOp = dyn_cast<arith::ExtSIOp>(defOp)) {
    collectOriginalTensorCandidates(extiOp.getIn(), seen, candidates);
    return;
  }
  if (auto trunciOp = dyn_cast<arith::TruncIOp>(defOp)) {
    collectOriginalTensorCandidates(trunciOp.getIn(), seen, candidates);
    return;
  }
  if (auto fptosiOp = dyn_cast<arith::FPToSIOp>(defOp)) {
    collectOriginalTensorCandidates(fptosiOp.getIn(), seen, candidates);
    return;
  }
  if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(defOp)) {
    collectOriginalTensorCandidates(sitofpOp.getIn(), seen, candidates);
    return;
  }

  if (auto addfOp = dyn_cast<arith::AddFOp>(defOp)) {
    collectOriginalTensorCandidates(addfOp.getLhs(), seen, candidates);
    collectOriginalTensorCandidates(addfOp.getRhs(), seen, candidates);
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    if (!forOp->hasAttr("reductionLoop"))
      return;
    (void)forOp.walk([&](tensor::ExtractSliceOp sliceOp) {
      Value source = sliceOp.getSource();
      if (llvm::is_contained(forOp.getRegionIterArgs(), source))
        return WalkResult::advance();
      collectOriginalTensorCandidates(source, seen, candidates);
      return WalkResult::interrupt();
    });
    return;
  }

  if (isa<RankedTensorType>(value.getType()))
    candidates.push_back(value);
  LDBG("collectOriginalTensorCandidates: stopping at unrecognized op: "
       << defOp->getName());
}

// The reduction dimensions belong to the vector source. Only project them
// onto a tensor candidate when the read path preserves every axis in order.
static bool hasIdentityAxisMapping(Value value, Value candidate,
                                   llvm::DenseSet<Value> &seen) {
  if (value == candidate)
    return true;
  if (!seen.insert(value).second)
    return false;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  if (auto maskOp = dyn_cast<vector::MaskOp>(defOp)) {
    if (auto readOp =
            dyn_cast_or_null<vector::TransferReadOp>(maskOp.getMaskableOp()))
      return hasIdentityAxisMapping(readOp.getResult(), candidate, seen);
    return false;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(defOp)) {
    auto sourceType = dyn_cast<RankedTensorType>(readOp.getSource().getType());
    auto vectorType = dyn_cast<VectorType>(readOp.getType());
    return sourceType && vectorType &&
           sourceType.getRank() == vectorType.getRank() &&
           readOp.getPermutationMap().isIdentity() &&
           hasIdentityAxisMapping(readOp.getSource(), candidate, seen);
  }
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
    if (sliceOp.getDroppedDims().any())
      return false;
    return hasIdentityAxisMapping(sliceOp.getSource(), candidate, seen);
  }
  if (auto castOp = dyn_cast<vector::ShapeCastOp>(defOp)) {
    if (castOp.getSource().getType() != castOp.getType())
      return false;
    return hasIdentityAxisMapping(castOp.getSource(), candidate, seen);
  }
  if (auto extOp = dyn_cast<arith::ExtFOp>(defOp))
    return hasIdentityAxisMapping(extOp.getIn(), candidate, seen);
  if (auto truncOp = dyn_cast<arith::TruncFOp>(defOp))
    return hasIdentityAxisMapping(truncOp.getIn(), candidate, seen);
  if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
    return hasIdentityAxisMapping(extOp.getIn(), candidate, seen);
  if (auto truncOp = dyn_cast<arith::TruncIOp>(defOp))
    return hasIdentityAxisMapping(truncOp.getIn(), candidate, seen);
  if (auto addOp = dyn_cast<arith::AddFOp>(defOp))
    return hasIdentityAxisMapping(addOp.getLhs(), candidate, seen) ||
           hasIdentityAxisMapping(addOp.getRhs(), candidate, seen);
  return false;
}

struct LogicalReductionShape {
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> reductionDims;
  SmallVector<int64_t> parallelDims;
  SmallVector<int64_t> reductionShape;
  SmallVector<int64_t> parallelShape;
  // Stable grouping of reduction axes before parallel axes. This describes a
  // logical RA view; interleaved axes require a permutation before collapse.
  SmallVector<int64_t> raPermutation;
  int64_t dimR = 1;
  int64_t dimA = 1;
};

struct ReductionStagePlan {
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> outputShape;
  SmallVector<int64_t> reductionDims;
  int64_t reductionBegin = 0;
  int64_t reductionRank = 0;
  int64_t outer = 1;
  int64_t dimR = 1;
  int64_t inner = 1;
};

// Physical layout used to feed one reduction stage. `AR2D` keeps the
// reduction axis last ([outer, R]); `RA2D` and `BatchedRA3D` keep the
// remaining parallel axis last ([R, inner] and [outer, R, inner]), so vector
// lanes always map to consecutive A elements.
enum class ReductionStageLayout {
  AR2D,
  RA2D,
  BatchedRA3D,
};

struct ReductionStageView {
  Value tensor;
  ReductionStageLayout layout;
  ReductionStagePlan plan;
};

// Deterministic radix-16 fold schedule shared by the Ascend-style AR and RA
// lowerings. `unit` is the number of reduction elements covered by one
// register: one row for RA and one vector for AR. `mainR` is the largest
// multiple of `unit` made of power-of-two units that does not exceed `dimR`,
// so every stage divides evenly and only the tail needs folding into the head.
struct ReductionFoldPlan {
  int64_t mainR = 0;
  int64_t tailR = 0;
  int64_t folds = 0;
  int64_t mainTimes = 0;
  int64_t tailFolds = 0;
  int64_t numLeaves = 0;
  int64_t tailUnits = 0;
};

static ReductionFoldPlan computeReductionFoldPlan(int64_t dimR, int64_t unit) {
  assert(dimR > 0 && unit > 0 && "fold schedule needs positive extents");
  int64_t mainR = unit;
  while (mainR <= dimR / 2)
    mainR *= 2;
  int64_t tailR = dimR - mainR;
  int64_t units = mainR / unit;
  int64_t folds = llvm::Log2_64(units);
  int64_t mainTimes = folds / 4;
  int64_t tailFolds = folds % 4;
  int64_t numLeaves = int64_t{1} << tailFolds;
  int64_t tailUnits = (tailR + unit - 1) / unit;
  return {mainR, tailR, folds, mainTimes, tailFolds, numLeaves, tailUnits};
}

static FailureOr<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || ShapedType::isDynamic(dim) ||
        elements > std::numeric_limits<int64_t>::max() / dim)
      return failure();
    elements *= dim;
  }
  return elements;
}

static FailureOr<SmallVector<ReductionStagePlan>>
buildReductionStagePlan(const LogicalReductionShape &logicalShape) {
  SmallVector<SmallVector<int64_t>> groups;
  for (int64_t dim : logicalShape.reductionDims) {
    if (groups.empty() || groups.back().back() + 1 != dim)
      groups.emplace_back();
    groups.back().push_back(dim);
  }

  SmallVector<int64_t> currentShape = logicalShape.inputShape;
  SmallVector<int64_t> currentDims(currentShape.size());
  std::iota(currentDims.begin(), currentDims.end(), 0);
  SmallVector<ReductionStagePlan> stages;
  stages.reserve(groups.size());

  // Process inner groups first. Removing a suffix-side group does not change
  // the current positions of any outer group.
  for (auto groupIt = groups.rbegin(); groupIt != groups.rend(); ++groupIt) {
    ArrayRef<int64_t> group = *groupIt;
    auto beginIt = llvm::find(currentDims, group.front());
    if (beginIt == currentDims.end())
      return failure();
    int64_t begin = std::distance(currentDims.begin(), beginIt);
    int64_t end = begin + group.size();
    if (end > static_cast<int64_t>(currentDims.size()) ||
        !std::equal(group.begin(), group.end(), currentDims.begin() + begin))
      return failure();

    ReductionStagePlan stage;
    stage.inputShape = currentShape;
    stage.reductionDims.append(group.begin(), group.end());
    auto outer = getStaticNumElements(ArrayRef(currentShape).take_front(begin));
    auto dimR =
        getStaticNumElements(ArrayRef(currentShape).slice(begin, group.size()));
    auto inner = getStaticNumElements(ArrayRef(currentShape).drop_front(end));
    if (failed(outer) || failed(dimR) || failed(inner))
      return failure();
    stage.outer = *outer;
    stage.dimR = *dimR;
    stage.inner = *inner;
    stage.reductionBegin = begin;
    stage.reductionRank = group.size();

    currentShape.erase(currentShape.begin() + begin,
                       currentShape.begin() + end);
    currentDims.erase(currentDims.begin() + begin, currentDims.begin() + end);
    stage.outputShape = currentShape;
    stages.push_back(std::move(stage));
  }

  if (currentShape != logicalShape.parallelShape)
    return failure();
  return stages;
}

static FailureOr<LogicalReductionShape>
inferLogicalReductionShape(vector::MultiDimReductionOp reduceOp,
                           RankedTensorType inputType,
                           RankedTensorType accumulatorType) {
  int64_t rank = inputType.getRank();
  if (rank == 0 || reduceOp.getReductionDims().empty())
    return failure();

  LogicalReductionShape shape;
  shape.inputShape.append(inputType.getShape().begin(),
                          inputType.getShape().end());
  SmallVector<char> isReductionDim(rank, false);
  for (Attribute attr : reduceOp.getReductionDims()) {
    auto dimAttr = dyn_cast<IntegerAttr>(attr);
    if (!dimAttr)
      return failure();
    int64_t dim = dimAttr.getInt();
    if (dim < 0 || dim >= rank || isReductionDim[dim])
      return failure();
    isReductionDim[dim] = true;
  }

  auto appendDimension =
      [](int64_t size, int64_t dim, SmallVectorImpl<int64_t> &dims,
         SmallVectorImpl<int64_t> &sizes, int64_t &product) -> LogicalResult {
    if (size <= 0 || ShapedType::isDynamic(size) ||
        product > std::numeric_limits<int64_t>::max() / size)
      return failure();
    dims.push_back(dim);
    sizes.push_back(size);
    product *= size;
    return success();
  };

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t size = shape.inputShape[dim];
    if (isReductionDim[dim]) {
      if (failed(appendDimension(size, dim, shape.reductionDims,
                                 shape.reductionShape, shape.dimR)))
        return failure();
    } else if (failed(appendDimension(size, dim, shape.parallelDims,
                                      shape.parallelShape, shape.dimA))) {
      return failure();
    }
  }

  if (shape.dimR > std::numeric_limits<int64_t>::max() / shape.dimA)
    return failure();

  shape.raPermutation.append(shape.reductionDims.begin(),
                             shape.reductionDims.end());
  shape.raPermutation.append(shape.parallelDims.begin(),
                             shape.parallelDims.end());

  // The accumulator may use a rank-1 singleton to represent a scalar in the
  // existing 2D path. Otherwise its shape must be the exact unreduced shape.
  ArrayRef<int64_t> accShape = accumulatorType.getShape();
  if (accShape != ArrayRef<int64_t>(shape.parallelShape) &&
      !(shape.parallelShape.empty() && accShape.size() == 1 &&
        accShape.front() == 1))
    return failure();
  return shape;
}

class TreeReductionBuilder {
public:
  IRRewriter &rewriter;
  Location loc;
  Type inElemTy, accElemTy, computeElemTy;
  VectorType vec1DInTy, vec2DInTy, vec1DAccTy, vec1DComputeTy;
  Value c0, c1, c16, c64, dimAVal, inputCstZero, accCstZero, computeCstZero;
  int64_t dimA;
  int64_t vectorLength;

  TreeReductionBuilder(IRRewriter &rewriter, Location loc, Type inTy,
                       Type accTy, int64_t dimA, int64_t vLen)
      : rewriter(rewriter), loc(loc), inElemTy(inTy), accElemTy(accTy),
        dimA(dimA), vectorLength(vLen) {
    c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    c64 = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    dimAVal = rewriter.create<arith::ConstantIndexOp>(loc, dimA);

    computeElemTy = inElemTy;
    if (isa<FloatType>(inElemTy) && inElemTy.getIntOrFloatBitWidth() < 32)
      computeElemTy = rewriter.getF32Type();

    vec1DInTy = VectorType::get({vectorLength}, inElemTy);
    vec2DInTy = VectorType::get({1, vectorLength}, inElemTy);
    vec1DAccTy = VectorType::get({vectorLength}, accElemTy);
    vec1DComputeTy = VectorType::get({vectorLength}, computeElemTy);

    inputCstZero = rewriter.create<arith::ConstantOp>(
        loc, inElemTy, rewriter.getZeroAttr(inElemTy));
    accCstZero = rewriter.create<arith::ConstantOp>(
        loc, accElemTy, rewriter.getZeroAttr(accElemTy));
    computeCstZero = rewriter.create<arith::ConstantOp>(
        loc, computeElemTy, rewriter.getZeroAttr(computeElemTy));
  }

  Value castToComputeTy(Value vec) {
    auto vecTy = dyn_cast<VectorType>(vec.getType());
    if (!vecTy)
      return vec;
    Type vecElemTy = vecTy.getElementType();
    if (vecElemTy == computeElemTy) {
      return vec;
    }
    auto targetVecTy = VectorType::get(vecTy.getShape(), computeElemTy);
    if (isa<FloatType>(vecElemTy)) {
      return rewriter.create<arith::ExtFOp>(loc, targetVecTy, vec);
    }
    return rewriter.create<arith::ExtSIOp>(loc, targetVecTy, vec);
  }

  Value createAdd(Value lhs, Value rhs) {
    Type ty = lhs.getType();
    if (auto vecTy = dyn_cast<VectorType>(ty))
      ty = vecTy.getElementType();

    if (isa<FloatType>(ty))
      return rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
  }

  Value buildTreeReduction(SmallVector<Value> &currentLevel) {
    while (currentLevel.size() > 1) {
      SmallVector<Value> nextLevel;
      int halfSize = currentLevel.size() / 2;
      nextLevel.reserve(halfSize + (currentLevel.size() % 2));
      for (int i = 0; i < halfSize; ++i)
        nextLevel.push_back(
            createAdd(currentLevel[i], currentLevel[i + halfSize]));
      if (currentLevel.size() % 2 != 0)
        nextLevel.push_back(currentLevel.back());
      currentLevel = std::move(nextLevel);
    }
    return currentLevel.front();
  }

  Value createLoopMask1D(Value minBoundary) {
    return rewriter.create<vector::CreateMaskOp>(
        loc, VectorType::get({vectorLength}, rewriter.getI1Type()),
        ValueRange{minBoundary});
  }

  Value readSliceMaskedRA(Value tensor, Value rIdx, Value aIdx, Value mask1D,
                          VectorType vecTy, Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, static_cast<Operation *>(nullptr),
        [this, vecTy, tensor, rIdx, aIdx, cstZero](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
                              loc, vecTy, tensor, ValueRange{rIdx, aIdx},
                              cstZero, ArrayRef<bool>{false})
                             .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }

  Value read1DAccMasked(Value tensor, Value idx, Value mask1D, VectorType vecTy,
                        Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, static_cast<Operation *>(nullptr),
        [this, vecTy, tensor, idx, cstZero](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
                              loc, vecTy, tensor, ValueRange{idx}, cstZero,
                              ArrayRef<bool>{false})
                             .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }

  Value readSliceMaskedAR(Value tensor, Value aIdx, Value rIdx, Value mask1D,
                          VectorType vecTy, Value cstZero) {
    if (mask1D) {
      auto maskOp = rewriter.create<vector::MaskOp>(
          loc, TypeRange{vecTy}, mask1D, static_cast<Operation *>(nullptr),
          [this, vecTy, tensor, aIdx, rIdx, cstZero](OpBuilder &b,
                                                     Operation *) {
            Value readOp = b.create<vector::TransferReadOp>(
                                loc, vecTy, tensor, ValueRange{aIdx, rIdx},
                                cstZero, ArrayRef<bool>{false})
                               .getResult();
            b.create<vector::YieldOp>(loc, readOp);
          });
      return maskOp.getResult(0);
    } else {
      return rewriter
          .create<vector::TransferReadOp>(loc, vecTy, tensor,
                                          ValueRange{aIdx, rIdx}, cstZero,
                                          ArrayRef<bool>{false})
          .getResult();
    }
  }

  Value write1DToSlice(Value vec1D, Value tensor, Value rIdx, Value aIdx,
                       Value minA, Value mask1D) {
    auto elemTy = cast<RankedTensorType>(tensor.getType()).getElementType();
    auto sliceTy = RankedTensorType::get({ShapedType::kDynamic}, elemTy);
    SmallVector<OpFoldResult> offsets = {rIdx, aIdx};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, tensor, offsets, sizes, strides);

    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{sliceTy}, mask1D, static_cast<Operation *>(nullptr),
        [this, vec1D, slice](OpBuilder &b, Operation *) {
          Value writeOp =
              b.create<vector::TransferWriteOp>(
                   loc, vec1D, slice, ValueRange{c0}, ArrayRef<bool>{false})
                  .getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    return rewriter
        .create<tensor::InsertSliceOp>(loc, maskOp.getResult(0), tensor,
                                       offsets, sizes, strides)
        .getResult();
  }

  Value promoteToComputeType(Value inputTensor, int64_t dimR,
                             AffineMap minMap) {
    if (inElemTy == computeElemTy)
      return inputTensor;
    LDBG("Promoting to compute type: " << inElemTy << " -> " << computeElemTy);
    auto emptyTy = RankedTensorType::get({dimR, dimA}, computeElemTy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, emptyTy.getShape(), computeElemTy);
    Value dimRVal = rewriter.create<arith::ConstantIndexOp>(loc, dimR);

    auto loopR = rewriter.create<scf::ForOp>(loc, c0, dimRVal, c1,
                                             ValueRange{emptyTensor});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value rAcc = loopR.getRegionIterArg(0);
    Value ivR = loopR.getInductionVar();

    auto loopA =
        rewriter.create<scf::ForOp>(loc, c0, dimAVal, c64, ValueRange{rAcc});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value aAcc = loopA.getRegionIterArg(0);
    Value ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap,
                                                      ValueRange{ivA, dimAVal});
    Value mask1D = createLoopMask1D(minA);

    Value rawRead = readSliceMaskedRA(inputTensor, ivR, ivA, mask1D, vec1DInTy,
                                      inputCstZero);
    Value castedRead = castToComputeTy(rawRead);
    Value insertOp = write1DToSlice(castedRead, aAcc, ivR, ivA, minA, mask1D);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertOp});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopA.getResult(0)});
    rewriter.setInsertionPointAfter(loopR);
    return loopR.getResult(0);
  }
};

static bool hasUseOutsideScope(Value value, Operation *scope);

struct TreeReduceV2Pass : public impl::TreeReduceV2Base<TreeReduceV2Pass> {
  using TreeReduceV2Base<TreeReduceV2Pass>::TreeReduceV2Base;
  void runOnOperation() override;

private:
  bool isValid2DReduction(vector::MultiDimReductionOp reduceOp,
                          Value inputTensor, int64_t &dimR, int64_t &dimA,
                          bool &isAR) {
    if (reduceOp.getKind() != vector::CombiningKind::ADD)
      return false;
    auto reduceDims = reduceOp.getReductionDims();
    if (reduceDims.empty() || reduceDims.size() != 1)
      return false;

    int64_t reducedDim = cast<mlir::IntegerAttr>(reduceDims[0]).getInt();
    auto rankedTy = dyn_cast_or_null<RankedTensorType>(inputTensor.getType());
    if (!rankedTy || rankedTy.getRank() != 2)
      return false;

    if (reducedDim == 0) {
      isAR = false;
      dimR = rankedTy.getShape()[0];
      dimA = rankedTy.getShape()[1];
    } else if (reducedDim == 1) {
      isAR = true;
      dimA = rankedTy.getShape()[0];
      dimR = rankedTy.getShape()[1];
    } else {
      return false;
    }
    return dimR > 0 && dimA > 0;
  }
  int64_t getVectorLength(Type elemTy) const {
    int64_t bytesPerElement = elemTy.getIntOrFloatBitWidth() / 8;
    if (bytesPerElement == 0)
      bytesPerElement = 1;
    return vectorBytes / bytesPerElement;
  }
  int64_t getBlockDataLen(Type elemTy) const {
    int64_t bytesPerElement = elemTy.getIntOrFloatBitWidth() / 8;
    if (bytesPerElement == 0)
      bytesPerElement = 1;
    return blockBytes / bytesPerElement;
  }
  static constexpr int64_t vectorBytes = 256;
  static constexpr int64_t blockBytes = 32;

  enum class ReduceType { IS_REDUCE_SUM, IS_REDUCE_MAX, IS_REDUCE_MIN, OTHERS };

  void buildAscendCopyOut(IRRewriter &rewriter, Location loc,
                          TreeReductionBuilder &builder, scf::ForOp targetForOp,
                          Value inputTensor, int64_t dimR, int64_t dimA) const;

  void buildAscendGroupReduce(IRRewriter &rewriter, Location loc,
                              TreeReductionBuilder &builder,
                              scf::ForOp targetForOp, Value inputTensor,
                              int64_t dimR, int64_t dimA, bool needFoldR) const;

  void buildAscendShortReduceAR(IRRewriter &rewriter, Location loc,
                                TreeReductionBuilder &builder,
                                scf::ForOp targetForOp, Value inputTensor,
                                int64_t dimR, int64_t dimA) const;

  void buildRA(IRRewriter &rewriter, Location loc,
               TreeReductionBuilder &builder, scf::ForOp targetForOp,
               Value inputTensor, int64_t dimR, int64_t dimA,
               RankedTensorType accTensorType) const;
  void buildDirectRegisterRA(IRRewriter &rewriter, Location loc,
                             TreeReductionBuilder &builder,
                             scf::ForOp targetForOp, Value inputTensor,
                             int64_t dimR, int64_t dimA,
                             RankedTensorType accTensorType) const;
  void buildAR(IRRewriter &rewriter, Location loc,
               TreeReductionBuilder &builder, scf::ForOp targetForOp,
               Value inputTensor, int64_t dimR, int64_t dimA,
               RankedTensorType accTensorType) const;
  // AscendC-style AR folding, mirroring `ReduceAR` in
  // op_common/atvoss/reduce/reduce_operator.h: `mainR` is the largest power of
  // two multiple of the vector length not exceeding dimR, the tail
  // [mainR, dimR) is folded into the head first, and each stage then folds 16
  // vectors into one until fewer than 16 remain. Shared by single-axis AR and
  // by suffix multi-axis stages on an AR2D [outer, R] view.
  Value buildARReductionStage(IRRewriter &rewriter, Location loc,
                              Value stageInput, Type inputElemTy,
                              Type computeElemTy, int64_t vectorLength,
                              const ReductionStagePlan &stage,
                              bool canReuseInput, Value finalAccumulator) const;
  // Single-axis AR entry point. Wraps the rank-2 [dimA, dimR] input as an AR2D
  // view and runs the shared Ascend-style AR stage.
  void rewriteSingleAxisAR(IRRewriter &rewriter, Location loc,
                           TreeReductionBuilder &builder,
                           scf::ForOp targetForOp, Value inputTensor,
                           int64_t dimR, int64_t dimA,
                           RankedTensorType accTensorType) const;
  // Single-axis RA entry point. Builds the RA2D view for a rank-2 [dimR, dimA]
  // tensor and runs the shared batched RA stage with the loop accumulator as
  // its final merge target.
  void rewriteSingleAxisRA(IRRewriter &rewriter, Location loc,
                           TreeReductionBuilder &builder,
                           scf::ForOp targetForOp, Value inputTensor,
                           int64_t dimR, int64_t dimA,
                           RankedTensorType accTensorType) const;
  Value buildSuffixMultiAxisStage(IRRewriter &rewriter, Location loc,
                                  Value stageInput, Type inputElemTy,
                                  Type computeElemTy, int64_t vectorLength,
                                  const ReductionStagePlan &stage,
                                  Value finalAccumulator) const;
  // AscendC-style RA folding. `mainR` is the largest power of two not
  // exceeding dimR, the tail [mainR, dimR) is folded into the head, and each
  // stage then folds 16 rows into one. Handles both RA2D (outer == 1) and
  // BatchedRA3D (outer > 1) layouts; the final vector keeps every A lane and
  // merges directly with the accumulator, without a horizontal reduction.
  Value buildBatchedRAReductionStage(IRRewriter &rewriter, Location loc,
                                     Value stageInput, Type inputElemTy,
                                     Type computeElemTy, int64_t vectorLength,
                                     const ReductionStagePlan &stage,
                                     ReductionStageLayout layout,
                                     bool allowInputReuse,
                                     Value finalAccumulator) const;
  Value buildMultiAxisStage(IRRewriter &rewriter, Location loc,
                            Value stageInput, Type inputElemTy,
                            Type computeElemTy, int64_t vectorLength,
                            const ReductionStagePlan &stage,
                            ReductionStageLayout layout, bool canReuseInput,
                            Value finalAccumulator) const;
  Value mergeMultiAxisVectorIntoAccumulator(IRRewriter &rewriter, Location loc,
                                            Value reduced,
                                            Value flatAccumulator, Value offset,
                                            Value mask,
                                            Type computeElemTy) const;
  Value mergeMultiAxisScalarIntoAccumulator(IRRewriter &rewriter, Location loc,
                                            Value reduced,
                                            Value scalarAccumulator,
                                            Type computeElemTy) const;
  void buildMultiAxisReduction(IRRewriter &rewriter, Location loc,
                               scf::ForOp targetForOp, Value inputTensor,
                               RankedTensorType accTensorType,
                               const LogicalReductionShape &logicalShape,
                               Type computeElemTy, int64_t vectorLength) const;
};

static Value collapseToFlatTensor(IRRewriter &rewriter, Location loc,
                                  Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  if (tensorType.getRank() == 1)
    return tensor;

  ReassociationIndices indices;
  indices.reserve(tensorType.getRank());
  for (int64_t dim = 0; dim < tensorType.getRank(); ++dim)
    indices.push_back(dim);
  SmallVector<ReassociationIndices> reassociation = {std::move(indices)};
  auto flatType = RankedTensorType::get({tensorType.getNumElements()},
                                        tensorType.getElementType());
  return rewriter.create<tensor::CollapseShapeOp>(loc, flatType, tensor,
                                                  reassociation);
}

// Collapse `tensor` into the given ordered dim groups. Every group must be
// non-empty and the groups must cover the source shape in order, so the
// operation is a pure zero-copy view that preserves the element order.
static Value collapseToDimGroups(IRRewriter &rewriter, Location loc,
                                 Value tensor,
                                 ArrayRef<ReassociationIndices> groups) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> shape;
  shape.reserve(groups.size());
  int64_t nextDim = 0;
  for (const ReassociationIndices &group : groups) {
    assert(!group.empty() && "collapse groups must be non-empty");
    int64_t extent = 1;
    for (int64_t dim : group) {
      assert(dim == nextDim && "groups must cover the shape in order");
      extent *= tensorType.getShape()[dim];
      ++nextDim;
    }
    shape.push_back(extent);
  }
  assert(nextDim == tensorType.getRank() && "groups must cover every dim");
  auto resultType = RankedTensorType::get(shape, tensorType.getElementType());
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultType, tensor,
                                                  groups);
}

// Expand a flat tensor to a stage output shape without moving elements.
static Value expandFlatTensorToShape(IRRewriter &rewriter, Location loc,
                                     Value flat, ArrayRef<int64_t> shape) {
  auto flatType = cast<RankedTensorType>(flat.getType());
  if (flatType.getShape() == shape)
    return flat;
  assert(flatType.getRank() == 1 && !shape.empty() &&
         "only a flat tensor can be expanded to a stage output shape");
  int64_t elementCount = 1;
  for (int64_t extent : shape)
    elementCount *= extent;
  assert(flatType.getNumElements() == elementCount &&
         "expanded shape must preserve the element count");
  ReassociationIndices indices;
  indices.reserve(shape.size());
  for (int64_t dim = 0; dim < static_cast<int64_t>(shape.size()); ++dim)
    indices.push_back(dim);
  SmallVector<ReassociationIndices> reassociation = {std::move(indices)};
  auto resultType = RankedTensorType::get(shape, flatType.getElementType());
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, flat,
                                                reassociation);
}

// Materialize the zero-copy AR2D/RA2D/BatchedRA3D view for `stage`. The input
// must already carry `stage.inputShape`; only collapse_shape and expand_shape
// are used, so the element order never changes.
static ReductionStageView
getReductionStageView(IRRewriter &rewriter, Location loc, Value tensor,
                      const ReductionStagePlan &stage) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  assert(tensorType.getShape() == ArrayRef<int64_t>(stage.inputShape) &&
         "stage input must carry the expected logical shape");

  SmallVector<int64_t> targetShape;
  ReductionStageLayout layout;
  if (stage.inner == 1) {
    layout = ReductionStageLayout::AR2D;
    targetShape = {stage.outer, stage.dimR};
  } else if (stage.outer == 1) {
    layout = ReductionStageLayout::RA2D;
    targetShape = {stage.dimR, stage.inner};
  } else {
    layout = ReductionStageLayout::BatchedRA3D;
    targetShape = {stage.outer, stage.dimR, stage.inner};
  }

  if (tensorType.getShape() == ArrayRef<int64_t>(targetShape))
    return {tensor, layout, stage};

  int64_t rank = tensorType.getRank();
  int64_t begin = stage.reductionBegin;
  int64_t end = begin + stage.reductionRank;

  // AR2D with a unit outer extent has no prefix dims to collapse into a
  // leading one, so flatten first and expand back to [1, R].
  if (layout == ReductionStageLayout::AR2D && stage.outer == 1) {
    Value flat = collapseToFlatTensor(rewriter, loc, tensor);
    return {expandFlatTensorToShape(rewriter, loc, flat, targetShape), layout,
            stage};
  }

  // inner == 1 means every dim after the reduction group is a unit extent;
  // fold them into the group so the collapse has exactly two groups. The
  // symmetric case applies to RA2D with a unit outer extent.
  if (layout == ReductionStageLayout::AR2D)
    end = rank;
  else if (layout == ReductionStageLayout::RA2D)
    begin = 0;
  else
    assert(begin > 0 && end < rank &&
           "batched RA needs non-unit outer and inner extents");

  SmallVector<ReassociationIndices> groups;
  auto appendGroup = [&](int64_t first, int64_t last) {
    ReassociationIndices group;
    for (int64_t dim = first; dim < last; ++dim)
      group.push_back(dim);
    groups.push_back(std::move(group));
  };
  if (begin > 0)
    appendGroup(0, begin);
  appendGroup(begin, end);
  if (end < rank)
    appendGroup(end, rank);
  assert(groups.size() == targetShape.size() &&
         "stage view must collapse into the expected number of groups");
  Value view = collapseToDimGroups(rewriter, loc, tensor, groups);
  return {view, layout, stage};
}

Value TreeReduceV2Pass::mergeMultiAxisVectorIntoAccumulator(
    IRRewriter &rewriter, Location loc, Value reduced, Value flatAccumulator,
    Value offset, Value mask, Type computeElemTy) const {
  auto accumulatorType = cast<RankedTensorType>(flatAccumulator.getType());
  assert(accumulatorType.getRank() == 1 &&
         "vector accumulator merge expects a flat tensor");
  Type accumulatorElemTy = accumulatorType.getElementType();
  auto reducedType = cast<VectorType>(reduced.getType());
  auto accumulatorVectorType =
      VectorType::get(reducedType.getShape(), accumulatorElemTy);
  Value accumulatorZero = rewriter.create<arith::ConstantOp>(
      loc, accumulatorElemTy, rewriter.getZeroAttr(accumulatorElemTy));
  auto readMask = rewriter.create<vector::MaskOp>(
      loc, TypeRange{accumulatorVectorType}, mask,
      static_cast<Operation *>(nullptr),
      [loc, flatAccumulator, offset, accumulatorZero,
       accumulatorVectorType](OpBuilder &builder, Operation *) {
        Value read = builder.create<vector::TransferReadOp>(
            loc, accumulatorVectorType, flatAccumulator, ValueRange{offset},
            accumulatorZero, ArrayRef<bool>{true});
        builder.create<vector::YieldOp>(loc, read);
      });

  Value oldAccumulator = readMask.getResult(0);
  if (accumulatorElemTy != computeElemTy)
    oldAccumulator =
        rewriter.create<arith::ExtFOp>(loc, reducedType, oldAccumulator);
  Value merged = rewriter.create<arith::AddFOp>(loc, reduced, oldAccumulator);
  Value result = merged;
  if (accumulatorElemTy != computeElemTy)
    result =
        rewriter.create<arith::TruncFOp>(loc, accumulatorVectorType, merged);

  auto writeMask = rewriter.create<vector::MaskOp>(
      loc, TypeRange{accumulatorType}, mask, static_cast<Operation *>(nullptr),
      [loc, result, flatAccumulator, offset](OpBuilder &builder, Operation *) {
        Value written = builder
                            .create<vector::TransferWriteOp>(
                                loc, result, flatAccumulator,
                                ValueRange{offset}, ArrayRef<bool>{true})
                            .getResult();
        builder.create<vector::YieldOp>(loc, written);
      });
  return writeMask.getResult(0);
}

Value TreeReduceV2Pass::mergeMultiAxisScalarIntoAccumulator(
    IRRewriter &rewriter, Location loc, Value reduced, Value scalarAccumulator,
    Type computeElemTy) const {
  auto accumulatorType = cast<RankedTensorType>(scalarAccumulator.getType());
  assert(accumulatorType.getRank() == 0 &&
         "scalar accumulator merge expects a rank-0 tensor");
  Type accumulatorElemTy = accumulatorType.getElementType();
  Value oldAccumulator =
      rewriter.create<tensor::ExtractOp>(loc, scalarAccumulator, ValueRange{});
  if (accumulatorElemTy != computeElemTy)
    oldAccumulator =
        rewriter.create<arith::ExtFOp>(loc, computeElemTy, oldAccumulator);
  Value merged = rewriter.create<arith::AddFOp>(loc, reduced, oldAccumulator);
  Value result = merged;
  if (accumulatorElemTy != computeElemTy)
    result = rewriter.create<arith::TruncFOp>(loc, accumulatorElemTy, merged);
  return rewriter.create<tensor::InsertOp>(loc, result, scalarAccumulator,
                                           ValueRange{});
}

Value TreeReduceV2Pass::buildSuffixMultiAxisStage(
    IRRewriter &rewriter, Location loc, Value stageInput, Type inputElemTy,
    Type computeElemTy, int64_t vectorLength, const ReductionStagePlan &stage,
    Value finalAccumulator) const {
  assert(stage.inner == 1 && "suffix reduction must have unit inner size");
  auto stageInputType = cast<RankedTensorType>(stageInput.getType());
  assert(stageInputType.getRank() == 2 &&
         stageInputType.getDimSize(0) == stage.outer &&
         stageInputType.getDimSize(1) == stage.dimR &&
         "suffix reduction expects an AR2D view");

  LDBG("Building suffix multi-axis stage as AR: outer="
       << stage.outer << " dimR=" << stage.dimR);
  TreeReductionBuilder builder(rewriter, loc, inputElemTy, computeElemTy,
                               stage.outer, vectorLength);
  assert(builder.computeElemTy == computeElemTy &&
         "suffix reduction compute type must match the stage type");

  Value output;
  if (finalAccumulator) {
    auto accumulatorType = cast<RankedTensorType>(finalAccumulator.getType());
    assert(accumulatorType.getNumElements() == stage.outer &&
           "suffix result must match the final accumulator");
    output = accumulatorType.getRank() == 0
                 ? finalAccumulator
                 : collapseToFlatTensor(rewriter, loc, finalAccumulator);
  } else {
    auto outputType = RankedTensorType::get({stage.outer}, computeElemTy);
    output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(),
                                              outputType.getElementType());
  }
  Value outerBound = rewriter.create<arith::ConstantIndexOp>(loc, stage.outer);
  auto outerLoop = rewriter.create<scf::ForOp>(loc, builder.c0, outerBound,
                                               builder.c1, ValueRange{output});
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  Value outerIndex = outerLoop.getInductionVar();
  Value loopOutput = outerLoop.getRegionIterArg(0);

  auto readVector = [&](Value offset) -> Value {
    Value raw = rewriter.create<vector::TransferReadOp>(
        loc, builder.vec1DInTy, stageInput, ValueRange{outerIndex, offset},
        builder.inputCstZero, ArrayRef<bool>{true});
    return builder.castToComputeTy(raw);
  };

  auto readTailVector = [&](Value offset, int64_t validElements) -> Value {
    Value valid = rewriter.create<arith::ConstantIndexOp>(loc, validElements);
    Value mask = builder.createLoopMask1D(valid);
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{builder.vec1DInTy}, mask,
        static_cast<Operation *>(nullptr),
        [loc, stageInput, outerIndex, offset, &builder](OpBuilder &b,
                                                        Operation *) {
          Value read = b.create<vector::TransferReadOp>(
                            loc, builder.vec1DInTy, stageInput,
                            ValueRange{outerIndex, offset},
                            builder.inputCstZero, ArrayRef<bool>{true})
                           .getResult();
          b.create<vector::YieldOp>(loc, read);
        });
    Value computeRead = builder.castToComputeTy(maskOp.getResult(0));
    Value zeroVector = rewriter.create<vector::BroadcastOp>(
        loc, builder.vec1DComputeTy, builder.computeCstZero);
    return rewriter.create<arith::SelectOp>(loc, mask, computeRead, zeroVector);
  };

  static constexpr int64_t numVectorRegisters = 16;
  int64_t fullVectors = stage.dimR / vectorLength;
  int64_t fullGroups = fullVectors / numVectorRegisters;
  int64_t remainingVectors = fullVectors % numVectorRegisters;
  int64_t tailElements = stage.dimR % vectorLength;
  Value accumulated = rewriter.create<vector::BroadcastOp>(
      loc, builder.vec1DComputeTy, builder.computeCstZero);

  if (fullGroups > 0) {
    Value groupBound = rewriter.create<arith::ConstantIndexOp>(loc, fullGroups);
    auto groupLoop = rewriter.create<scf::ForOp>(
        loc, builder.c0, groupBound, builder.c1, ValueRange{accumulated});
    rewriter.setInsertionPointToStart(groupLoop.getBody());
    Value groupIndex = groupLoop.getInductionVar();
    Value groupAccumulator = groupLoop.getRegionIterArg(0);
    Value groupStride = rewriter.create<arith::ConstantIndexOp>(
        loc, numVectorRegisters * vectorLength);
    Value groupBase =
        rewriter.create<arith::MulIOp>(loc, groupIndex, groupStride);

    SmallVector<Value> registers;
    registers.reserve(numVectorRegisters);
    for (int64_t registerIndex = 0; registerIndex < numVectorRegisters;
         ++registerIndex) {
      Value registerOffset = rewriter.create<arith::ConstantIndexOp>(
          loc, registerIndex * vectorLength);
      Value offset =
          rewriter.create<arith::AddIOp>(loc, groupBase, registerOffset);
      registers.push_back(readVector(offset));
    }
    Value groupSum = builder.buildTreeReduction(registers);
    Value nextAccumulator = builder.createAdd(groupAccumulator, groupSum);
    rewriter.create<scf::YieldOp>(loc, nextAccumulator);
    rewriter.setInsertionPointAfter(groupLoop);
    accumulated = groupLoop.getResult(0);
  }

  if (remainingVectors > 0) {
    SmallVector<Value> registers;
    registers.reserve(remainingVectors);
    int64_t baseVector = fullGroups * numVectorRegisters;
    for (int64_t vectorIndex = 0; vectorIndex < remainingVectors;
         ++vectorIndex) {
      Value offset = rewriter.create<arith::ConstantIndexOp>(
          loc, (baseVector + vectorIndex) * vectorLength);
      registers.push_back(readVector(offset));
    }
    accumulated =
        builder.createAdd(accumulated, builder.buildTreeReduction(registers));
  }

  if (tailElements > 0) {
    Value offset = rewriter.create<arith::ConstantIndexOp>(
        loc, fullVectors * vectorLength);
    accumulated =
        builder.createAdd(accumulated, readTailVector(offset, tailElements));
  }

  auto computeVec2DTy = VectorType::get({1, vectorLength}, computeElemTy);
  Value accumulated2D =
      rewriter.create<vector::ShapeCastOp>(loc, computeVec2DTy, accumulated);
  auto scalarVecTy = VectorType::get({1}, computeElemTy);
  Value zeroScalarVector = rewriter.create<vector::BroadcastOp>(
      loc, scalarVecTy, builder.computeCstZero);
  auto horizontalReduction = rewriter.create<vector::MultiDimReductionOp>(
      loc, vector::CombiningKind::ADD, accumulated2D, zeroScalarVector,
      rewriter.getI64ArrayAttr({1}));
  horizontalReduction->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
  Value nextOutput;
  if (!finalAccumulator) {
    nextOutput = rewriter
                     .create<vector::TransferWriteOp>(
                         loc, horizontalReduction, loopOutput,
                         ValueRange{outerIndex}, ArrayRef<bool>{true})
                     .getResult();
  } else if (cast<RankedTensorType>(loopOutput.getType()).getRank() == 0) {
    Value reduced = rewriter.create<vector::ExtractOp>(
        loc, horizontalReduction.getResult(), 0);
    nextOutput = mergeMultiAxisScalarIntoAccumulator(rewriter, loc, reduced,
                                                     loopOutput, computeElemTy);
  } else {
    auto scalarMaskType = VectorType::get({1}, rewriter.getI1Type());
    Value scalarMask = rewriter.create<vector::CreateMaskOp>(
        loc, scalarMaskType, ValueRange{builder.c1});
    nextOutput = mergeMultiAxisVectorIntoAccumulator(
        rewriter, loc, horizontalReduction.getResult(), loopOutput, outerIndex,
        scalarMask, computeElemTy);
  }
  rewriter.create<scf::YieldOp>(loc, nextOutput);
  rewriter.setInsertionPointAfter(outerLoop);
  return outerLoop.getResult(0);
}

Value TreeReduceV2Pass::buildBatchedRAReductionStage(
    IRRewriter &rewriter, Location loc, Value stageInput, Type inputElemTy,
    Type computeElemTy, int64_t vectorLength, const ReductionStagePlan &stage,
    ReductionStageLayout layout, bool allowInputReuse,
    Value finalAccumulator) const {
  assert(stage.inner > 1 && "batched RA expects a non-unit inner size");
  assert(stage.outer > 0 && stage.dimR > 0 && stage.inner > 0 &&
         "batched RA expects a positive static shape");
  assert((layout == ReductionStageLayout::RA2D ||
          layout == ReductionStageLayout::BatchedRA3D) &&
         "batched RA expects an RA2D or BatchedRA3D view");

  auto stageInputType = cast<RankedTensorType>(stageInput.getType());
  bool isBatched = layout == ReductionStageLayout::BatchedRA3D;
  if (isBatched) {
    assert(stageInputType.getRank() == 3 &&
           stageInputType.getDimSize(0) == stage.outer &&
           stageInputType.getDimSize(1) == stage.dimR &&
           stageInputType.getDimSize(2) == stage.inner &&
           "batched RA expects a BatchedRA3D view");
  } else {
    assert(stage.outer == 1 && stageInputType.getRank() == 2 &&
           stageInputType.getDimSize(0) == stage.dimR &&
           stageInputType.getDimSize(1) == stage.inner &&
           "RA2D expects a [R, inner] view");
  }
  assert(stageInputType.getElementType() == inputElemTy &&
         "stage view element type must match the stage input type");

  // RA registers cover one row each. `mainR` is the largest power of two not
  // exceeding `dimR`; [mainR, dimR) is folded into the head before the
  // radix-16 stages run, so every stage divides evenly.
  ReductionFoldPlan fold = computeReductionFoldPlan(stage.dimR, /*unit=*/1);
  int64_t mainR = fold.mainR;
  int64_t tailR = fold.tailR;
  int64_t mainTimes = fold.mainTimes;
  int64_t numLeaves = fold.numLeaves;
  bool needsWorkingTensor = tailR > 0 || mainTimes > 0;
  // In-place folding is only legal when the caller proved that the input is
  // dead after the reduction and already carries the compute type.
  bool useInputAsWorking =
      needsWorkingTensor && allowInputReuse && inputElemTy == computeElemTy;
  LDBG("buildBatchedRAReductionStage: outer="
       << stage.outer << " dimR=" << stage.dimR << " inner=" << stage.inner
       << " mainR=" << mainR << " tailR=" << tailR << " folds=" << fold.folds
       << " mainTimes=" << mainTimes << " tailFolds=" << fold.tailFolds
       << " numLeaves=" << numLeaves
       << " layout=" << (isBatched ? "BatchedRA3D" : "RA2D")
       << " reuseInput=" << useInputAsWorking);

  TreeReductionBuilder builder(rewriter, loc, inputElemTy, computeElemTy,
                               stage.inner, vectorLength);
  assert(builder.computeElemTy == computeElemTy &&
         "batched RA compute type must match the stage type");

  bool mergeIntoAccumulator = static_cast<bool>(finalAccumulator);
  Value result;
  if (mergeIntoAccumulator) {
    auto accumulatorType = cast<RankedTensorType>(finalAccumulator.getType());
    assert(accumulatorType.getRank() > 0 &&
           "batched RA cannot merge into a scalar accumulator");
    assert(accumulatorType.getNumElements() == stage.outer * stage.inner &&
           "batched RA result must match the final accumulator");
    result = accumulatorType.getRank() == 1
                 ? finalAccumulator
                 : collapseToFlatTensor(rewriter, loc, finalAccumulator);
  } else {
    result = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{stage.outer * stage.inner}, computeElemTy);
  }

  Value workingTensor;
  if (useInputAsWorking) {
    workingTensor = stageInput;
  } else if (needsWorkingTensor) {
    SmallVector<int64_t> workingShape;
    if (isBatched)
      workingShape = {stage.outer, mainR, stage.inner};
    else
      workingShape = {mainR, stage.inner};
    workingTensor =
        rewriter.create<tensor::EmptyOp>(loc, workingShape, computeElemTy);
  }

  Value c0 = builder.c0;
  Value c1 = builder.c1;
  Value outerBound = rewriter.create<arith::ConstantIndexOp>(loc, stage.outer);
  Value innerBound = rewriter.create<arith::ConstantIndexOp>(loc, stage.inner);
  Value cMainR = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
  Value cInner = rewriter.create<arith::ConstantIndexOp>(loc, stage.inner);

  AffineMap minMap = AffineMap::get(
      1, 1,
      {rewriter.getAffineConstantExpr(vectorLength),
       rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)},
      rewriter.getContext());

  // RA2D indexes [r, a]; BatchedRA3D keeps the leading outer index.
  auto stageOffsets = [&](Value outerIndex, Value rowIndex,
                          Value columnIndex) -> SmallVector<Value> {
    SmallVector<Value> offsets;
    if (isBatched)
      offsets.push_back(outerIndex);
    offsets.push_back(rowIndex);
    offsets.push_back(columnIndex);
    return offsets;
  };

  auto readVector = [&](Value tensor, ArrayRef<Value> offsets,
                        Value mask) -> Value {
    Type elemTy = cast<RankedTensorType>(tensor.getType()).getElementType();
    bool isComputeType = elemTy == computeElemTy;
    auto vectorType = VectorType::get({vectorLength}, elemTy);
    Value zero = isComputeType ? builder.computeCstZero : builder.inputCstZero;
    SmallVector<Value> offsetValues(offsets.begin(), offsets.end());
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vectorType}, mask, static_cast<Operation *>(nullptr),
        [loc, tensor, offsetValues, vectorType, zero](OpBuilder &b,
                                                      Operation *) {
          Value read = b.create<vector::TransferReadOp>(loc, vectorType, tensor,
                                                        offsetValues, zero,
                                                        ArrayRef<bool>{true})
                           .getResult();
          b.create<vector::YieldOp>(loc, read);
        });
    Value raw = maskOp.getResult(0);
    return isComputeType ? raw : builder.castToComputeTy(raw);
  };

  auto writeVector = [&](Value tensor, ArrayRef<Value> offsets, Value vector,
                         Value mask) -> Value {
    auto tensorType = cast<RankedTensorType>(tensor.getType());
    assert(tensorType.getElementType() == computeElemTy &&
           "RA fold writes must target a compute-type tensor");
    SmallVector<Value> offsetValues(offsets.begin(), offsets.end());
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{tensorType}, mask, static_cast<Operation *>(nullptr),
        [loc, vector, tensor, offsetValues](OpBuilder &b, Operation *) {
          Value written =
              b.create<vector::TransferWriteOp>(
                   loc, vector, tensor, offsetValues, ArrayRef<bool>{true})
                  .getResult();
          b.create<vector::YieldOp>(loc, written);
        });
    return maskOp.getResult(0);
  };

  auto mergeResult = [&](Value accumulator, Value vector, Value offset,
                         Value mask) -> Value {
    if (!mergeIntoAccumulator)
      return writeVector(accumulator, offset, vector, mask);
    return mergeMultiAxisVectorIntoAccumulator(
        rewriter, loc, vector, accumulator, offset, mask, computeElemTy);
  };

  SmallVector<Value> outerInitArgs{result};
  if (needsWorkingTensor)
    outerInitArgs.push_back(workingTensor);
  auto outerLoop =
      rewriter.create<scf::ForOp>(loc, c0, outerBound, c1, outerInitArgs);
  rewriter.setInsertionPointToStart(outerLoop.getBody());
  Value outerIndex = outerLoop.getInductionVar();
  Value outerResult = outerLoop.getRegionIterArg(0);
  Value outerWorking =
      needsWorkingTensor ? outerLoop.getRegionIterArg(1) : Value{};

  SmallVector<Value> tileInitArgs{outerResult};
  if (needsWorkingTensor)
    tileInitArgs.push_back(outerWorking);
  auto tileLoop = rewriter.create<scf::ForOp>(loc, c0, innerBound, builder.c64,
                                              tileInitArgs);
  rewriter.setInsertionPointToStart(tileLoop.getBody());
  Value innerIndex = tileLoop.getInductionVar();
  Value tileResult = tileLoop.getRegionIterArg(0);
  Value tileWorking =
      needsWorkingTensor ? tileLoop.getRegionIterArg(1) : Value{};
  Value validInner = rewriter.create<affine::AffineMinOp>(
      loc, minMap, ValueRange{innerIndex, innerBound});
  Value mask = builder.createLoopMask1D(validInner);

  // Stage 0a: fold the tail rows [mainR, dimR) into the head, one row at a
  // time. Every target row is below `tailR < mainR`, and every source row is
  // at least `mainR`, so no live value is overwritten.
  Value currentWorking = tileWorking;
  if (tailR > 0) {
    Value tailBound = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    auto loopR = rewriter.create<scf::ForOp>(loc, c0, tailBound, c1,
                                             ValueRange{currentWorking});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value reductionIndex = loopR.getInductionVar();
    Value working = loopR.getRegionIterArg(0);
    Value tailIndex =
        rewriter.create<arith::AddIOp>(loc, reductionIndex, cMainR);
    // When folding in place, read through the loop-carried value so the write
    // chain stays ordered under bufferization.
    Value readTensor = useInputAsWorking ? working : stageInput;
    Value head = readVector(
        readTensor, stageOffsets(outerIndex, reductionIndex, innerIndex), mask);
    Value tail = readVector(
        readTensor, stageOffsets(outerIndex, tailIndex, innerIndex), mask);
    Value folded = builder.createAdd(head, tail);
    Value next = writeVector(
        working, stageOffsets(outerIndex, reductionIndex, innerIndex), folded,
        mask);
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(loopR);
    currentWorking = loopR.getResult(0);
  }

  // Stage 0b: the main tree reads all rows in [0, mainR), but the tail fold
  // only materialized [0, tailR). Forward the remaining rows into the
  // compute-type working tensor instead of converting the whole input first.
  if (!useInputAsWorking && mainTimes > 0 && tailR < mainR) {
    Value tailBound = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    auto loopR = rewriter.create<scf::ForOp>(loc, tailBound, cMainR, c1,
                                             ValueRange{currentWorking});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value reductionIndex = loopR.getInductionVar();
    Value working = loopR.getRegionIterArg(0);
    Value forwarded = readVector(
        stageInput, stageOffsets(outerIndex, reductionIndex, innerIndex), mask);
    Value next = writeVector(
        working, stageOffsets(outerIndex, reductionIndex, innerIndex),
        forwarded, mask);
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(loopR);
    currentWorking = loopR.getResult(0);
  }

  // Main stages: each one folds 16 registers into one, so `currentR` shrinks
  // by a factor of 16 per stage.
  int64_t currentR = mainR;
  for (int64_t stageIndex = 0; stageIndex < mainTimes; ++stageIndex) {
    int64_t nextR = currentR / 16;
    LDBG("buildBatchedRAReductionStage: stage " << stageIndex << ": "
                                                << currentR << " -> " << nextR);
    Value nextBound = rewriter.create<arith::ConstantIndexOp>(loc, nextR);
    Value rowStride = rewriter.create<arith::ConstantIndexOp>(loc, nextR);
    auto loopR = rewriter.create<scf::ForOp>(loc, c0, nextBound, c1,
                                             ValueRange{currentWorking});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value reductionIndex = loopR.getInductionVar();
    Value working = loopR.getRegionIterArg(0);
    SmallVector<Value> registers;
    registers.reserve(16);
    for (int registerIndex = 0; registerIndex < 16; ++registerIndex) {
      Value strideIndex =
          rewriter.create<arith::ConstantIndexOp>(loc, registerIndex);
      Value stride =
          rewriter.create<arith::MulIOp>(loc, strideIndex, rowStride);
      Value row = rewriter.create<arith::AddIOp>(loc, reductionIndex, stride);
      registers.push_back(
          readVector(working, stageOffsets(outerIndex, row, innerIndex), mask));
    }
    Value reduced = builder.buildTreeReduction(registers);
    Value next = writeVector(
        working, stageOffsets(outerIndex, reductionIndex, innerIndex), reduced,
        mask);
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(loopR);
    currentWorking = loopR.getResult(0);
    currentR = nextR;
  }

  // Final register tree: `numLeaves` survivors fold to one vector that is
  // merged with the accumulator directly, keeping every A lane.
  SmallVector<Value> leaves;
  leaves.reserve(numLeaves);
  for (int64_t leafIndex = 0; leafIndex < numLeaves; ++leafIndex) {
    Value rowIndex = rewriter.create<arith::ConstantIndexOp>(loc, leafIndex);
    Value tensor = currentWorking;
    if (!needsWorkingTensor ||
        (!useInputAsWorking && mainTimes == 0 && leafIndex >= tailR))
      tensor = stageInput;
    leaves.push_back(readVector(
        tensor, stageOffsets(outerIndex, rowIndex, innerIndex), mask));
  }
  Value reduced = builder.buildTreeReduction(leaves);

  Value outputOffset = rewriter.create<arith::MulIOp>(loc, outerIndex, cInner);
  outputOffset = rewriter.create<arith::AddIOp>(loc, outputOffset, innerIndex);
  Value nextResult = mergeResult(tileResult, reduced, outputOffset, mask);

  SmallVector<Value> tileYields{nextResult};
  if (needsWorkingTensor)
    tileYields.push_back(currentWorking);
  rewriter.create<scf::YieldOp>(loc, tileYields);
  rewriter.setInsertionPointAfter(tileLoop);
  SmallVector<Value> outerYields{tileLoop.getResult(0)};
  if (needsWorkingTensor)
    outerYields.push_back(tileLoop.getResult(1));
  rewriter.create<scf::YieldOp>(loc, outerYields);
  rewriter.setInsertionPointAfter(outerLoop);
  return outerLoop.getResult(0);
}

Value TreeReduceV2Pass::buildARReductionStage(
    IRRewriter &rewriter, Location loc, Value stageInput, Type inputElemTy,
    Type computeElemTy, int64_t vectorLength, const ReductionStagePlan &stage,
    bool canReuseInput, Value finalAccumulator) const {
  assert(stage.inner == 1 && "AR expects a unit inner extent");
  auto stageInputType = cast<RankedTensorType>(stageInput.getType());
  assert(stageInputType.getRank() == 2 &&
         stageInputType.getDimSize(0) == stage.outer &&
         stageInputType.getDimSize(1) == stage.dimR &&
         "AR expects an AR2D view");
  assert(stageInputType.getElementType() == inputElemTy &&
         "stage view element type must match the stage input type");

  // AR registers cover one vector each, so the schedule is expressed in units
  // of `vectorLength` reduction elements.
  ReductionFoldPlan fold = computeReductionFoldPlan(stage.dimR, vectorLength);
  int64_t mainR = fold.mainR;
  int64_t tailR = fold.tailR;
  int64_t base = mainR / vectorLength;
  int64_t mainTimes = fold.mainTimes;
  int64_t tailFolds = fold.tailFolds;
  // AscendC's inplaceRepeats: how many vectors of the head the tail touches.
  int64_t tailVecs = fold.tailUnits;
  int64_t numLeaves = fold.numLeaves;
  LDBG("buildARReductionStage: outer="
       << stage.outer << " dimR=" << stage.dimR << " mainR=" << mainR
       << " tailR=" << tailR << " base=" << base << " folds=" << fold.folds
       << " mainTimes=" << mainTimes << " tailFolds=" << tailFolds
       << " tailVecs=" << tailVecs << " numLeaves=" << numLeaves);
  for (int64_t stageIndex = 0, currentUnits = base; stageIndex < mainTimes;
       ++stageIndex) {
    int64_t nextUnits = currentUnits / 16;
    LDBG("buildARReductionStage: stage " << stageIndex << ": " << currentUnits
                                         << " -> " << nextUnits << " vectors");
    currentUnits = nextUnits;
  }

  TreeReductionBuilder builder(rewriter, loc, inputElemTy, computeElemTy,
                               stage.outer, vectorLength);
  assert(builder.computeElemTy == computeElemTy &&
         "AR compute type must match the stage type");

  bool needsFoldStage = tailR > 0 || mainTimes > 0;
  // In-place folding is only legal when the input is dead after the reduction
  // and already carries the compute type; otherwise use a private tensor.
  bool useSeparateWorkingTensor =
      needsFoldStage && (inputElemTy != computeElemTy || !canReuseInput);
  Value workingTensor = stageInput;
  if (useSeparateWorkingTensor)
    workingTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{stage.outer, mainR}, computeElemTy);

  bool mergeIntoAccumulator = static_cast<bool>(finalAccumulator);
  bool mergeScalar = false;
  Value result;
  if (mergeIntoAccumulator) {
    auto accumulatorType = cast<RankedTensorType>(finalAccumulator.getType());
    assert(accumulatorType.getNumElements() == stage.outer &&
           "AR result must match the final accumulator");
    mergeScalar = accumulatorType.getRank() == 0;
    result = accumulatorType.getRank() > 1
                 ? collapseToFlatTensor(rewriter, loc, finalAccumulator)
                 : finalAccumulator;
  } else {
    result = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{stage.outer}, computeElemTy);
  }

  AffineMap minMap = AffineMap::get(
      1, 1,
      {rewriter.getAffineConstantExpr(vectorLength),
       rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)},
      rewriter.getContext());

  Value outerBound = rewriter.create<arith::ConstantIndexOp>(loc, stage.outer);

  // Reads one vector of the reduction axis at [aIdx, rIdx) and promotes it to
  // the compute type. A non-null mask zero-fills inactive lanes.
  auto readVec = [&](Value tensor, Value aIdx, Value rIdx,
                     Value mask1D) -> Value {
    auto elemTy = cast<RankedTensorType>(tensor.getType()).getElementType();
    bool isCompute = elemTy == computeElemTy;
    Value raw = builder.readSliceMaskedAR(
        tensor, aIdx, rIdx, mask1D,
        isCompute ? builder.vec1DComputeTy : builder.vec1DInTy,
        isCompute ? builder.computeCstZero : builder.inputCstZero);
    return isCompute ? raw : builder.castToComputeTy(raw);
  };

  // Writes a full compute-type vector back to [aIdx, rIdx); every fold target
  // lies inside [0, mainR), so the transfer is always in bounds.
  auto writeVec = [&](Value vec1D, Value tensor, Value aIdx,
                      Value rIdx) -> Value {
    auto elemTy = cast<RankedTensorType>(tensor.getType()).getElementType();
    auto sliceTy = RankedTensorType::get({vectorLength}, elemTy);
    SmallVector<OpFoldResult> offsets = {aIdx, rIdx};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(vectorLength)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, tensor, offsets, sizes, strides);
    Value written = rewriter
                        .create<vector::TransferWriteOp>(loc, vec1D, slice,
                                                         ValueRange{builder.c0},
                                                         ArrayRef<bool>{true})
                        .getResult();
    return rewriter
        .create<tensor::InsertSliceOp>(loc, written, tensor, offsets, sizes,
                                       strides)
        .getResult();
  };

  Value baseVal = rewriter.create<arith::ConstantIndexOp>(loc, base);

  // Stage 0: fold the non-power-of-two tail into the head, one vector at a
  // time, mirroring the needInplaceAdd prologue of ReduceAR.
  if (tailR > 0) {
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mainRVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    Value tailVecsVal = rewriter.create<arith::ConstantIndexOp>(loc, tailVecs);
    Value zeroVec = rewriter.create<vector::BroadcastOp>(
        loc, builder.vec1DComputeTy, builder.computeCstZero);

    auto loopA = rewriter.create<scf::ForOp>(
        loc, builder.c0, outerBound, builder.c1, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();

    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, tailVecsVal,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();

    Value rIdx = rewriter.create<arith::MulIOp>(loc, ivR, builder.c64);
    Value minLen = rewriter.create<affine::AffineMinOp>(
        loc, minMap, ValueRange{rIdx, tailRVal});
    Value mask1D = builder.createLoopMask1D(minLen);
    Value tailIdx = rewriter.create<arith::AddIOp>(loc, rIdx, mainRVal);

    // When folding in place the write chain runs through loopRAcc; read from
    // it as well so the dependence survives bufferization.
    Value readTensor = useSeparateWorkingTensor ? stageInput : loopRAcc;
    Value mainVec = readVec(readTensor, ivA, rIdx, nullptr);
    Value tailVec = readVec(readTensor, ivA, tailIdx, mask1D);
    Value cleanTail =
        rewriter.create<arith::SelectOp>(loc, mask1D, tailVec, zeroVec);
    Value folded = builder.createAdd(mainVec, cleanTail);
    Value insertSliceOp = writeVec(folded, loopRAcc, ivA, rIdx);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    workingTensor = loopA.getResult(0);
  }

  // The main stage reads every vector of [0, mainR), but stage 0 only
  // materialized the first `tailVecs`. Forward the rest.
  if (useSeparateWorkingTensor && mainTimes > 0 && tailVecs < base) {
    Value tailVecsVal = rewriter.create<arith::ConstantIndexOp>(loc, tailVecs);
    auto loopA = rewriter.create<scf::ForOp>(
        loc, builder.c0, outerBound, builder.c1, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();

    auto loopR = rewriter.create<scf::ForOp>(loc, tailVecsVal, baseVal,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();

    Value rIdx = rewriter.create<arith::MulIOp>(loc, ivR, builder.c64);
    Value forwarded = readVec(stageInput, ivA, rIdx, nullptr);
    Value insertSliceOp = writeVec(forwarded, loopRAcc, ivA, rIdx);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    workingTensor = loopA.getResult(0);
  }

  // Main stages: each one folds 16 vectors into one, i.e. `loopRNum >>= 4`.
  if (mainTimes > 0) {
    Value mainTimesVal =
        rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
    auto loopMain =
        rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal, builder.c1,
                                    ValueRange{workingTensor, baseVal});
    rewriter.setInsertionPointToStart(loopMain.getBody());
    Value loopMainAcc = loopMain.getRegionIterArg(0),
          currentVecs = loopMain.getRegionIterArg(1);
    Value nextVecs =
        rewriter.create<arith::DivUIOp>(loc, currentVecs, builder.c16);
    Value offsetR = rewriter.create<arith::MulIOp>(loc, nextVecs, builder.c64);

    auto loopA = rewriter.create<scf::ForOp>(
        loc, builder.c0, outerBound, builder.c1, ValueRange{loopMainAcc});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();

    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, nextVecs,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    Value rBase = rewriter.create<arith::MulIOp>(loc, ivR, builder.c64);

    SmallVector<Value> vRegs;
    vRegs.reserve(16);
    for (int i = 0; i < 16; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value stride = rewriter.create<arith::MulIOp>(loc, iVal, offsetR);
      Value rIdx = rewriter.create<arith::AddIOp>(loc, rBase, stride);
      vRegs.push_back(readVec(loopRAcc, ivA, rIdx, nullptr));
    }
    Value reducedVec = builder.buildTreeReduction(vRegs);
    Value insertSliceOp = writeVec(reducedVec, loopRAcc, ivA, rBase);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc,
                                  ValueRange{loopA.getResult(0), nextVecs});
    rewriter.setInsertionPointAfter(loopMain);
    workingTensor = loopMain.getResult(0);
  }

  // Final fold: the 2^tailFolds surviving vectors go through one register
  // tree, then the single vector collapses onto the accumulator element.
  auto loopATail = rewriter.create<scf::ForOp>(loc, builder.c0, outerBound,
                                               builder.c1, ValueRange{result});
  rewriter.setInsertionPointToStart(loopATail.getBody());
  Value loopAAcc = loopATail.getRegionIterArg(0),
        ivA = loopATail.getInductionVar();

  SmallVector<Value> leaves;
  leaves.reserve(numLeaves);
  for (int64_t i = 0; i < numLeaves; ++i) {
    Value rIdx = rewriter.create<arith::ConstantIndexOp>(loc, i * vectorLength);
    Value readTensor = workingTensor;
    if (useSeparateWorkingTensor && mainTimes == 0 && i >= tailVecs)
      readTensor = stageInput;
    leaves.push_back(readVec(readTensor, ivA, rIdx, nullptr));
  }
  Value finalVec = builder.buildTreeReduction(leaves);

  Type elemTy = builder.computeElemTy;
  auto vec2DTy = VectorType::get({1, vectorLength}, elemTy);
  Value accVec2D = rewriter.create<vector::ShapeCastOp>(loc, vec2DTy, finalVec);
  auto vec1xComputeTy = VectorType::get({1}, elemTy);
  auto zeroScalarVector = rewriter.create<vector::BroadcastOp>(
      loc, vec1xComputeTy, builder.computeCstZero);

  Value nextResult;
  if (mergeIntoAccumulator && !mergeScalar) {
    // Preserve the existing AR order: the accumulator is the reduction init
    // and a single tag suppresses a second merge.
    auto accumulatorType = cast<RankedTensorType>(loopAAcc.getType());
    Type accumulatorElemTy = accumulatorType.getElementType();
    auto accVecTy = VectorType::get({1}, accumulatorElemTy);
    Value accumulatorZero = rewriter.create<arith::ConstantOp>(
        loc, accumulatorElemTy, rewriter.getZeroAttr(accumulatorElemTy));
    SmallVector<OpFoldResult> offsets = {ivA};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
    auto sliceTy = RankedTensorType::get({1}, accumulatorElemTy);
    Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, loopAAcc, offsets, sizes, strides);
    Value accRead = rewriter.create<vector::TransferReadOp>(
        loc, accVecTy, extractedSlice, ValueRange{builder.c0}, accumulatorZero);
    Value accReadCompute = accRead;
    if (accumulatorElemTy != elemTy) {
      accReadCompute =
          isa<FloatType>(accumulatorElemTy)
              ? rewriter.create<arith::ExtFOp>(loc, vec1xComputeTy, accRead)
                    .getResult()
              : rewriter.create<arith::ExtSIOp>(loc, vec1xComputeTy, accRead)
                    .getResult();
    }

    SmallVector<int64_t> reduceDims = {1};
    auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, vector::CombiningKind::ADD, accVec2D, accReadCompute,
        rewriter.getI64ArrayAttr(reduceDims));
    multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    Value toWriteAcc = multiReduceOp.getResult();
    if (accumulatorElemTy != elemTy) {
      toWriteAcc =
          isa<FloatType>(accumulatorElemTy)
              ? rewriter.create<arith::TruncFOp>(loc, accVecTy, toWriteAcc)
                    .getResult()
              : rewriter.create<arith::TruncIOp>(loc, accVecTy, toWriteAcc)
                    .getResult();
    }
    Value writtenSlice =
        rewriter
            .create<vector::TransferWriteOp>(loc, toWriteAcc, extractedSlice,
                                             ValueRange{builder.c0})
            .getResult();
    nextResult = rewriter.create<tensor::InsertSliceOp>(
        loc, writtenSlice, loopAAcc, offsets, sizes, strides);
  } else {
    auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, vector::CombiningKind::ADD, accVec2D, zeroScalarVector,
        rewriter.getI64ArrayAttr({1}));
    multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    if (mergeScalar) {
      Value reduced =
          rewriter.create<vector::ExtractOp>(loc, multiReduceOp.getResult(), 0);
      nextResult = mergeMultiAxisScalarIntoAccumulator(rewriter, loc, reduced,
                                                       loopAAcc, computeElemTy);
    } else {
      nextResult = rewriter
                       .create<vector::TransferWriteOp>(
                           loc, multiReduceOp.getResult(), loopAAcc,
                           ValueRange{ivA}, ArrayRef<bool>{true})
                       .getResult();
    }
  }

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextResult});
  rewriter.setInsertionPointAfter(loopATail);
  return loopATail.getResult(0);
}

Value TreeReduceV2Pass::buildMultiAxisStage(
    IRRewriter &rewriter, Location loc, Value stageInput, Type inputElemTy,
    Type computeElemTy, int64_t vectorLength, const ReductionStagePlan &stage,
    ReductionStageLayout layout, bool canReuseInput,
    Value finalAccumulator) const {
  assert(stage.outer > 0 && stage.dimR > 0 && stage.inner > 0 &&
         "multi-axis stage must have a positive static shape");

  if (stage.inner == 1) {
    assert(layout == ReductionStageLayout::AR2D &&
           "unit inner size must use the AR2D view");
    // Long suffix stages use the same Ascend-style AR folding as single-axis
    // AR; the short case still follows the existing group-reduce lowering.
    if (stage.dimR > vectorLength)
      return buildARReductionStage(rewriter, loc, stageInput, inputElemTy,
                                   computeElemTy, vectorLength, stage,
                                   canReuseInput, finalAccumulator);
    return buildSuffixMultiAxisStage(rewriter, loc, stageInput, inputElemTy,
                                     computeElemTy, vectorLength, stage,
                                     finalAccumulator);
  }

  return buildBatchedRAReductionStage(
      rewriter, loc, stageInput, inputElemTy, computeElemTy, vectorLength,
      stage, layout, /*allowInputReuse=*/canReuseInput, finalAccumulator);
}

void TreeReduceV2Pass::buildMultiAxisReduction(
    IRRewriter &rewriter, Location loc, scf::ForOp targetForOp,
    Value inputTensor, RankedTensorType accTensorType,
    const LogicalReductionShape &logicalShape, Type computeElemTy,
    int64_t vectorLength) const {
  FailureOr<SmallVector<ReductionStagePlan>> stagePlans =
      buildReductionStagePlan(logicalShape);
  assert(succeeded(stagePlans) && !stagePlans->empty() &&
         "validated multi-axis shape must have a stage plan");

  auto wrapperLoop = rewriter.create<scf::ForOp>(
      loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
      rewriter.create<arith::ConstantIndexOp>(loc, 1),
      rewriter.create<arith::ConstantIndexOp>(loc, 1),
      ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());

  auto inputType = cast<RankedTensorType>(inputTensor.getType());
  Type currentElemTy = inputType.getElementType();
  Value current = inputTensor;
  for (const auto &[stageIndex, stage] : llvm::enumerate(*stagePlans)) {
    LLVM_DEBUG({
      DBGS() << "Multi-axis stage " << stageIndex << ": axes=[";
      llvm::interleaveComma(stage.reductionDims, llvm::dbgs());
      llvm::dbgs() << "] input=[";
      llvm::interleaveComma(stage.inputShape, llvm::dbgs());
      llvm::dbgs() << "] output=[";
      llvm::interleaveComma(stage.outputShape, llvm::dbgs());
      llvm::dbgs() << "] outer=" << stage.outer << " dimR=" << stage.dimR
                   << " inner=" << stage.inner << "\n";
    });
    bool isFinalStage = stageIndex + 1 == stagePlans->size();
    // Stage 0 may reuse the original input only when it is dead after this
    // reduction; later stages own their fresh predecessor output. Check the
    // original value before creating the stage view, otherwise the new
    // collapse/expand operation itself looks like an out-of-scope use.
    bool canReuseInput = stageIndex == 0
                             ? (currentElemTy == computeElemTy &&
                                !hasUseOutsideScope(inputTensor, targetForOp))
                             : currentElemTy == computeElemTy;
    ReductionStageView view =
        getReductionStageView(rewriter, loc, current, stage);
    Value finalAccumulator =
        isFinalStage ? wrapperLoop.getRegionIterArg(0) : Value{};
    Value stageResult = buildMultiAxisStage(
        rewriter, loc, view.tensor, currentElemTy, computeElemTy, vectorLength,
        stage, view.layout, canReuseInput, finalAccumulator);
    current = isFinalStage ? stageResult
                           : expandFlatTensorToShape(rewriter, loc, stageResult,
                                                     stage.outputShape);
    currentElemTy = computeElemTy;
  }

  Value result = current;
  if (result.getType() != accTensorType) {
    auto flatResultType = cast<RankedTensorType>(result.getType());
    assert(accTensorType.getRank() > 1 && flatResultType.getRank() == 1 &&
           flatResultType.getElementType() == accTensorType.getElementType() &&
           flatResultType.getNumElements() == accTensorType.getNumElements() &&
           "final multi-axis result must be a flattened accumulator");
    ReassociationIndices indices;
    indices.reserve(accTensorType.getRank());
    for (int64_t dim = 0; dim < accTensorType.getRank(); ++dim)
      indices.push_back(dim);
    SmallVector<ReassociationIndices> reassociation = {std::move(indices)};
    result = rewriter.create<tensor::ExpandShapeOp>(loc, accTensorType, result,
                                                    reassociation);
  }
  rewriter.create<scf::YieldOp>(loc, result);
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

void TreeReduceV2Pass::buildDirectRegisterRA(
    IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder,
    scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA,
    RankedTensorType accTensorType) const {
  assert(accTensorType.getRank() == 1 &&
         "direct register RA expects a rank-1 accumulator");
  assert(dimR > 0 && dimR <= 64 &&
         "direct register RA supports one bounded tree");

  auto wrapperLoop =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.c1, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());
  Value initAcc = wrapperLoop.getRegionIterArg(0);

  AffineMap minMap = AffineMap::get(
      1, 1,
      {rewriter.getAffineConstantExpr(builder.vectorLength),
       rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)},
      rewriter.getContext());
  auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal,
                                           builder.c64, ValueRange{initAcc});
  rewriter.setInsertionPointToStart(loopA.getBody());
  Value loopAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();
  Value minA = rewriter.create<affine::AffineMinOp>(
      loc, minMap, ValueRange{ivA, builder.dimAVal});
  Value mask = builder.createLoopMask1D(minA);

  uint64_t alignedR = llvm::PowerOf2Ceil(static_cast<uint64_t>(dimR));
  auto buildRegisterTree = [&](auto &&self, uint64_t start, uint64_t stride,
                               uint64_t count) -> Value {
    if (count == 1) {
      if (start >= static_cast<uint64_t>(dimR))
        return rewriter.create<vector::SplatOp>(loc, builder.vec1DComputeTy,
                                                builder.computeCstZero);
      Value rIdx = rewriter.create<arith::ConstantIndexOp>(loc, start);
      Value row =
          builder.readSliceMaskedRA(inputTensor, rIdx, ivA, mask,
                                    builder.vec1DInTy, builder.inputCstZero);
      return builder.castToComputeTy(row);
    }

    // SplitReductionOp pairs the lower and upper halves at every level.  Walk
    // that tree depth-first so the generated association remains identical,
    // while only O(log N) vector partials are live instead of all N leaves.
    Value lhs = self(self, start, stride * 2, count / 2);
    Value rhs = self(self, start + stride, stride * 2, count / 2);
    return builder.createAdd(rhs, lhs);
  };
  auto accSliceTy =
      RankedTensorType::get({ShapedType::kDynamic}, builder.accElemTy);
  SmallVector<OpFoldResult> offsets = {ivA};
  SmallVector<OpFoldResult> sizes = {minA};
  SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
  Value accSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, accSliceTy, loopAcc, offsets, sizes, strides);
  Value oldAcc = builder.read1DAccMasked(
      accSlice, builder.c0, mask, builder.vec1DAccTy, builder.accCstZero);
  Value oldAccCompute = builder.castToComputeTy(oldAcc);
  Value result;
  if (alignedR == 1) {
    Value leaf = buildRegisterTree(buildRegisterTree, /*start=*/0,
                                   /*stride=*/1, /*count=*/1);
    result = builder.createAdd(leaf, oldAccCompute);
  } else {
    // SplitReductionOp feeds the original destination only to the final
    // two-way combine. Preserve that association: right + (left + init).
    // Intermediate neutral additions are deliberately omitted.
    Value left = buildRegisterTree(buildRegisterTree, /*start=*/0,
                                   /*stride=*/2, alignedR / 2);
    Value right = buildRegisterTree(buildRegisterTree, /*start=*/1,
                                    /*stride=*/2, alignedR / 2);
    result = builder.createAdd(right, builder.createAdd(left, oldAccCompute));
  }
  if (builder.accElemTy != builder.computeElemTy) {
    result =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::TruncFOp>(loc, builder.vec1DAccTy, result)
                  .getResult()
            : rewriter.create<arith::TruncIOp>(loc, builder.vec1DAccTy, result)
                  .getResult();
  }

  auto writeMask = rewriter.create<vector::MaskOp>(
      loc, TypeRange{accSliceTy}, mask, static_cast<Operation *>(nullptr),
      [&builder, loc, result, accSlice](OpBuilder &b, Operation *) {
        Value written = b.create<vector::TransferWriteOp>(
                             loc, result, accSlice, ValueRange{builder.c0},
                             ArrayRef<bool>{true})
                            .getResult();
        b.create<vector::YieldOp>(loc, written);
      });
  Value inserted = rewriter.create<tensor::InsertSliceOp>(
      loc, writeMask.getResult(0), loopAcc, offsets, sizes, strides);
  rewriter.create<scf::YieldOp>(loc, inserted);

  rewriter.setInsertionPointAfter(loopA);
  rewriter.create<scf::YieldOp>(loc, loopA.getResult(0));
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

static bool hasUseOutsideScope(Value value, Operation *scope) {
  return llvm::any_of(value.getUses(), [scope](OpOperand &use) {
    Operation *owner = use.getOwner();
    return owner != scope && !scope->isAncestor(owner);
  });
}

void TreeReduceV2Pass::rewriteSingleAxisRA(
    IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder,
    scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA,
    RankedTensorType accTensorType) const {
  LDBG("rewriteSingleAxisRA: dimR=" << dimR << " dimA=" << dimA);

  auto inputType = cast<RankedTensorType>(inputTensor.getType());
  assert(inputType.getRank() == 2 && "single-axis RA expects a rank-2 input");
  assert(inputType.getDimSize(0) == dimR && inputType.getDimSize(1) == dimA &&
         "single-axis RA expects an [R, A] tensor");
  assert(dimA > 1 && "the batched RA stage requires a non-unit inner extent");
  assert(accTensorType.getRank() == 1 &&
         accTensorType.getNumElements() == dimA &&
         "single-axis RA expects a rank-1 accumulator");

  ReductionStagePlan stage;
  stage.inputShape = {dimR, dimA};
  stage.outputShape = {dimA};
  stage.reductionDims = {0};
  stage.reductionBegin = 0;
  stage.reductionRank = 1;
  stage.outer = 1;
  stage.dimR = dimR;
  stage.inner = dimA;

  bool canReuseInput = builder.inElemTy == builder.computeElemTy &&
                       !hasUseOutsideScope(inputTensor, targetForOp);

  auto wrapperLoop =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.c1, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());
  Value merged = buildBatchedRAReductionStage(
      rewriter, loc, inputTensor, inputType.getElementType(),
      builder.computeElemTy, builder.vectorLength, stage,
      ReductionStageLayout::RA2D, /*allowInputReuse=*/canReuseInput,
      wrapperLoop.getRegionIterArg(0));
  rewriter.create<scf::YieldOp>(loc, merged);
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

void TreeReduceV2Pass::buildRA(IRRewriter &rewriter, Location loc,
                               TreeReductionBuilder &builder,
                               scf::ForOp targetForOp, Value inputTensor,
                               int64_t dimR, int64_t dimA,
                               RankedTensorType accTensorType) const {
  LDBG("buildRA: dimR=" << dimR << " dimA=" << dimA);
  auto wrapperLoop =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.c1, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());
  Value initAcc = wrapperLoop.getRegionIterArg(0);

  AffineMap minMap = AffineMap::get(
      1, 1,
      {rewriter.getAffineConstantExpr(builder.vectorLength),
       rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)},
      rewriter.getContext());

  // RA registers cover one row each.
  ReductionFoldPlan fold = computeReductionFoldPlan(dimR, /*unit=*/1);
  int64_t mainR = fold.mainR;
  int64_t tailR = fold.tailR;
  int64_t mainTimes = fold.mainTimes;
  int64_t tailFolds = fold.tailFolds;
  LDBG("mainR=" << mainR << " tailR=" << tailR << " folds=" << fold.folds
                << " mainTimes=" << mainTimes << " tailFolds=" << tailFolds);

  Value sourceTensor = builder.promoteToComputeType(inputTensor, dimR, minMap);
  Value workingTensor = sourceTensor;
  bool preserveInput = sourceTensor == inputTensor &&
                       hasUseOutsideScope(inputTensor, targetForOp);
  bool needsPartialTensor = tailR > 0 || mainTimes > 0;
  bool useSeparateWorkingTensor = preserveInput && needsPartialTensor;
  if (useSeparateWorkingTensor) {
    auto inputType = cast<RankedTensorType>(inputTensor.getType());
    workingTensor = rewriter.create<tensor::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());
  }

  auto read2D = [&rewriter, loc, &builder](Value tensor, Value rIdx, Value cIdx,
                                           Value minA, Value mask) -> Value {
    auto sliceTy =
        RankedTensorType::get({1, ShapedType::kDynamic}, builder.computeElemTy);
    SmallVector<OpFoldResult> offsets = {rIdx, cIdx};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, tensor, offsets, sizes, strides);

    auto vec2DTy =
        VectorType::get({1, builder.vectorLength}, builder.computeElemTy);
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vec2DTy}, mask, nullptr,
        [loc, vec2DTy, slice, &builder](OpBuilder &b, Operation *) {
          Value readOp =
              b.create<vector::TransferReadOp>(
                   loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                   builder.computeCstZero, ArrayRef<bool>{true, true})
                  .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  };

  auto write2D = [&rewriter, loc, &builder](Value vec2D, Value tensor,
                                            Value rIdx, Value cIdx, Value minA,
                                            Value mask) -> Value {
    auto sliceTy =
        RankedTensorType::get({1, ShapedType::kDynamic}, builder.computeElemTy);
    SmallVector<OpFoldResult> offsets = {rIdx, cIdx};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, tensor, offsets, sizes, strides);

    auto maskWriteOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{sliceTy}, mask, nullptr,
        [&builder, loc, vec2D, slice](OpBuilder &b, Operation *) {
          Value writeOp =
              b.create<vector::TransferWriteOp>(
                   loc, vec2D, slice, ValueRange{builder.c0, builder.c0},
                   ArrayRef<bool>{true, true})
                  .getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    return rewriter
        .create<tensor::InsertSliceOp>(loc, maskWriteOp.getResult(0), tensor,
                                       offsets, sizes, strides)
        .getResult();
  };

  if (tailR > 0) {
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mainRVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    auto loopA =
        rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal,
                                    builder.c64, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(
        loc, minMap, ValueRange{ivA, builder.dimAVal});

    auto mask2DTy =
        VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
    Value currentMask = rewriter.create<vector::CreateMaskOp>(
        loc, mask2DTy, ValueRange{builder.c1, minA});

    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, tailRVal,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    Value tailIdx = rewriter.create<arith::AddIOp>(loc, ivR, mainRVal);

    Value readTensor = useSeparateWorkingTensor ? sourceTensor : loopRAcc;
    Value vecMain = read2D(readTensor, ivR, ivA, minA, currentMask);
    Value vecTail = read2D(readTensor, tailIdx, ivA, minA, currentMask);
    Value reducedVec = builder.createAdd(vecMain, vecTail);
    Value insertSliceOp =
        write2D(reducedVec, loopRAcc, ivR, ivA, minA, currentMask);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    workingTensor = loopA.getResult(0);
  }

  // A later tree-reduction stage expects every active row in [0, mainR) to
  // reside in workingTensor. Tail folding above materializes [0, tailR);
  // forward the remaining rows with legal vector transfers instead of a
  // whole-tensor linalg.copy. Small reductions without a main stage keep the
  // rows in sourceTensor and select their source directly in the final fold.
  if (useSeparateWorkingTensor && mainTimes > 0 && tailR < mainR) {
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mainRVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    auto loopA =
        rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal,
                                    builder.c64, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(
        loc, minMap, ValueRange{ivA, builder.dimAVal});

    auto mask2DTy =
        VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
    Value currentMask = rewriter.create<vector::CreateMaskOp>(
        loc, mask2DTy, ValueRange{builder.c1, minA});

    auto loopR = rewriter.create<scf::ForOp>(loc, tailRVal, mainRVal,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    Value forwarded = read2D(sourceTensor, ivR, ivA, minA, currentMask);
    Value insertSliceOp =
        write2D(forwarded, loopRAcc, ivR, ivA, minA, currentMask);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    workingTensor = loopA.getResult(0);
  }

  if (mainTimes > 0) {
    Value loopRNumVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    Value mainTimesVal =
        rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
    auto loopMain =
        rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal, builder.c1,
                                    ValueRange{workingTensor, loopRNumVal});
    rewriter.setInsertionPointToStart(loopMain.getBody());

    Value loopMainAcc = loopMain.getRegionIterArg(0),
          currentDimR = loopMain.getRegionIterArg(1);
    Value nextRNumVal =
        rewriter.create<arith::DivUIOp>(loc, currentDimR, builder.c16);
    auto loopA = rewriter.create<scf::ForOp>(
        loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{loopMainAcc});
    rewriter.setInsertionPointToStart(loopA.getBody());

    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(
        loc, minMap, ValueRange{ivA, builder.dimAVal});

    auto mask2DTy =
        VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
    Value currentMask = rewriter.create<vector::CreateMaskOp>(
        loc, mask2DTy, ValueRange{builder.c1, minA});

    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, nextRNumVal,
                                             builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();

    SmallVector<Value> vRegs;
    vRegs.reserve(16);
    for (int i = 0; i < 16; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value offsetR = rewriter.create<arith::MulIOp>(loc, iVal, nextRNumVal);
      Value rIdx = rewriter.create<arith::AddIOp>(loc, ivR, offsetR);
      vRegs.push_back(read2D(loopRAcc, rIdx, ivA, minA, currentMask));
    }

    Value finalVecAdd = builder.buildTreeReduction(vRegs);
    Value insertSliceOp =
        write2D(finalVecAdd, loopRAcc, ivR, ivA, minA, currentMask);

    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc,
                                  ValueRange{loopA.getResult(0), nextRNumVal});
    rewriter.setInsertionPointAfter(loopMain);
    workingTensor = loopMain.getResult(0);
  }

  int64_t numLeaves = fold.numLeaves;
  LDBG("numLeaves=" << numLeaves);
  auto loopATail = rewriter.create<scf::ForOp>(
      loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{initAcc});
  rewriter.setInsertionPointToStart(loopATail.getBody());
  Value finalLoopAAcc = loopATail.getRegionIterArg(0),
        ivA = loopATail.getInductionVar();
  Value minA = rewriter.create<affine::AffineMinOp>(
      loc, minMap, ValueRange{ivA, builder.dimAVal});

  auto mask2DTy =
      VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
  Value currentMask = rewriter.create<vector::CreateMaskOp>(
      loc, mask2DTy, ValueRange{builder.c1, minA});

  SmallVector<Value> tailRegs;
  tailRegs.reserve(std::min<int64_t>(numLeaves, 64));
  for (int64_t i = 0; i < numLeaves && i < 64; ++i) {
    Value offsetR = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value readTensor = workingTensor;
    if (useSeparateWorkingTensor && mainTimes == 0 && i >= tailR)
      readTensor = sourceTensor;
    tailRegs.push_back(read2D(readTensor, offsetR, ivA, minA, currentMask));
  }
  Value finalOutputVec = builder.buildTreeReduction(tailRegs);

  auto vec1DTy = VectorType::get({builder.vectorLength}, builder.computeElemTy);
  Value finalOutputVec1D =
      rewriter.create<vector::ShapeCastOp>(loc, vec1DTy, finalOutputVec);

  if (accTensorType.getRank() == 0) {
    VectorType vec0DType = VectorType::get({}, builder.accElemTy);
    Value finalScalar = rewriter.create<vector::ExtractElementOp>(
        loc, finalOutputVec1D, builder.c0);

    Value oldScalar =
        rewriter.create<tensor::ExtractOp>(loc, finalLoopAAcc, ValueRange{});
    Value oldScalarCompute = oldScalar;
    if (builder.accElemTy != builder.computeElemTy) {
      oldScalarCompute = isa<FloatType>(builder.accElemTy)
                             ? rewriter
                                   .create<arith::ExtFOp>(
                                       loc, builder.computeElemTy, oldScalar)
                                   .getResult()
                             : rewriter
                                   .create<arith::ExtSIOp>(
                                       loc, builder.computeElemTy, oldScalar)
                                   .getResult();
    }

    Value totalSumCompute = builder.createAdd(finalScalar, oldScalarCompute);

    Value finalSum = totalSumCompute;
    if (builder.accElemTy != builder.computeElemTy) {
      finalSum = isa<FloatType>(builder.accElemTy)
                     ? rewriter
                           .create<arith::TruncFOp>(loc, builder.accElemTy,
                                                    totalSumCompute)
                           .getResult()
                     : rewriter
                           .create<arith::TruncIOp>(loc, builder.accElemTy,
                                                    totalSumCompute)
                           .getResult();
    }
    Value broadcastOp =
        rewriter.create<vector::BroadcastOp>(loc, vec0DType, finalSum);
    Value writeOp = rewriter
                        .create<vector::TransferWriteOp>(
                            loc, broadcastOp, finalLoopAAcc, ValueRange{})
                        .getResult();
    rewriter.create<scf::YieldOp>(loc, ValueRange{writeOp});
  } else {
    auto accSliceTy =
        RankedTensorType::get({ShapedType::kDynamic}, builder.accElemTy);
    SmallVector<OpFoldResult> accOffsets = {ivA};
    SmallVector<OpFoldResult> accSizes = {minA};
    SmallVector<OpFoldResult> accStrides = {rewriter.getIndexAttr(1)};
    Value accSlice =
        rewriter
            .create<tensor::ExtractSliceOp>(loc, accSliceTy, finalLoopAAcc,
                                            accOffsets, accSizes, accStrides)
            .getResult();

    Value currentMask1D = builder.createLoopMask1D(minA);

    auto maskReadOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{builder.vec1DAccTy}, currentMask1D,
        static_cast<Operation *>(nullptr),
        [&builder, loc, accSlice](OpBuilder &b, Operation *) {
          Value readOp =
              b.create<vector::TransferReadOp>(
                   loc, builder.vec1DAccTy, accSlice, ValueRange{builder.c0},
                   builder.accCstZero, ArrayRef<bool>{true})
                  .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    Value oldAccVec = maskReadOp.getResult(0);

    Value oldAccCompute = builder.castToComputeTy(oldAccVec);
    Value totalSumCompute = builder.createAdd(finalOutputVec1D, oldAccCompute);

    Value toWriteAcc = totalSumCompute;
    if (builder.accElemTy != builder.computeElemTy) {
      toWriteAcc = isa<FloatType>(builder.accElemTy)
                       ? rewriter
                             .create<arith::TruncFOp>(loc, builder.vec1DAccTy,
                                                      totalSumCompute)
                             .getResult()
                       : rewriter
                             .create<arith::TruncIOp>(loc, builder.vec1DAccTy,
                                                      totalSumCompute)
                             .getResult();
    }

    auto writeMaskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{accSliceTy}, currentMask1D,
        static_cast<Operation *>(nullptr),
        [&builder, loc, toWriteAcc, accSlice](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(
                               loc, toWriteAcc, accSlice,
                               ValueRange{builder.c0}, ArrayRef<bool>{true})
                              .getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    Value insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, writeMaskOp.getResult(0), finalLoopAAcc, accOffsets, accSizes,
        accStrides);
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
  }

  rewriter.setInsertionPointAfter(loopATail);
  rewriter.create<scf::YieldOp>(loc, ValueRange{loopATail.getResult(0)});
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

void TreeReduceV2Pass::buildAscendCopyOut(IRRewriter &rewriter, Location loc,
                                          TreeReductionBuilder &builder,
                                          scf::ForOp targetForOp,
                                          Value inputTensor, int64_t dimR,
                                          int64_t dimA) const {
  LDBG("buildAscendCopyOut: dimR=" << dimR << " dimA=" << dimA);

  auto loopA =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(loopA.getBody());

  Value loopAAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();

  Value extractedVal = rewriter.create<tensor::ExtractOp>(
      loc, inputTensor, ValueRange{ivA, builder.c0});

  Value toWriteAcc = extractedVal;
  if (builder.inElemTy != builder.accElemTy) {
    toWriteAcc =
        isa<FloatType>(builder.accElemTy)
            ? rewriter
                  .create<arith::ExtFOp>(loc, builder.accElemTy, extractedVal)
                  .getResult()
            : rewriter
                  .create<arith::ExtSIOp>(loc, builder.accElemTy, extractedVal)
                  .getResult();
  }

  Value nextAcc = rewriter.create<tensor::InsertOp>(loc, toWriteAcc, loopAAcc,
                                                    ValueRange{ivA});

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
  rewriter.setInsertionPointAfter(loopA);
  rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::buildAscendShortReduceAR(IRRewriter &rewriter,
                                                Location loc,
                                                TreeReductionBuilder &builder,
                                                scf::ForOp targetForOp,
                                                Value inputTensor, int64_t dimR,
                                                int64_t dimA) const {
  LDBG("buildAscendShortReduceAR: dimR=" << dimR << " dimA=" << dimA);
  int64_t vlElems = builder.vectorLength;

  auto loopA =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(loopA.getBody());
  Value loopAAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();

  Value dimRVal = rewriter.create<arith::ConstantIndexOp>(loc, dimR);

  auto mask2DTy = VectorType::get({1, vlElems}, rewriter.getI1Type());
  Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value mask2D = rewriter.create<vector::CreateMaskOp>(
      loc, mask2DTy, ValueRange{c1Val, dimRVal});

  auto sliceTy =
      RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
  SmallVector<OpFoldResult> inOffsets = {ivA, builder.c0};
  SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), dimRVal};
  SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
  Value slice = rewriter.create<tensor::ExtractSliceOp>(
      loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);

  auto vec2DTy = VectorType::get({1, vlElems}, builder.inElemTy);
  auto maskOp = rewriter.create<vector::MaskOp>(
      loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation *>(nullptr),
      [&builder, loc, vec2DTy, slice](OpBuilder &b, Operation *) {
        Value readOp =
            b.create<vector::TransferReadOp>(
                 loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                 builder.inputCstZero, llvm::ArrayRef<bool>{true, false})
                .getResult();
        b.create<vector::YieldOp>(loc, readOp);
      });

  Value computeRead = builder.castToComputeTy(maskOp.getResult(0));

  auto computeVec2DTy = VectorType::get({1, vlElems}, builder.computeElemTy);
  Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, computeVec2DTy,
                                                       builder.computeCstZero);
  Value cleanRead =
      rewriter.create<arith::SelectOp>(loc, mask2D, computeRead, zeroVec);

  SmallVector<OpFoldResult> accOffsets = {ivA};
  SmallVector<OpFoldResult> accSizes = {rewriter.getIndexAttr(1)};
  SmallVector<OpFoldResult> accStrides = {rewriter.getIndexAttr(1)};
  auto accSliceTy = RankedTensorType::get({1}, builder.accElemTy);

  Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, accSliceTy, loopAAcc, accOffsets, accSizes, accStrides);

  auto accVecTy = VectorType::get({1}, builder.accElemTy);
  Value accRead = rewriter.create<vector::TransferReadOp>(
      loc, accVecTy, extractedSlice, ValueRange{builder.c0}, builder.accCstZero,
      llvm::ArrayRef<bool>{true});

  Value accReadCompute = accRead;
  auto accVecComputeTy = VectorType::get({1}, builder.computeElemTy);
  if (builder.accElemTy != builder.computeElemTy) {
    accReadCompute =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::ExtFOp>(loc, accVecComputeTy, accRead)
                  .getResult()
            : rewriter.create<arith::ExtSIOp>(loc, accVecComputeTy, accRead)
                  .getResult();
  }

  SmallVector<int64_t> reduceDims = {1};
  auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
      loc, vector::CombiningKind::ADD, cleanRead, accReadCompute,
      rewriter.getI64ArrayAttr(reduceDims));
  multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
  Value toWriteAcc = multiReduceOp.getResult();

  if (builder.accElemTy != builder.computeElemTy) {
    toWriteAcc =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::TruncFOp>(loc, accVecTy, toWriteAcc)
                  .getResult()
            : rewriter.create<arith::TruncIOp>(loc, accVecTy, toWriteAcc)
                  .getResult();
  }

  Value writtenSlice =
      rewriter
          .create<vector::TransferWriteOp>(loc, toWriteAcc, extractedSlice,
                                           ValueRange{builder.c0},
                                           llvm::ArrayRef<bool>{true})
          .getResult();

  Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
      loc, writtenSlice, loopAAcc, accOffsets, accSizes, accStrides);

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
  rewriter.setInsertionPointAfter(loopA);
  rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::buildAscendGroupReduce(
    IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder,
    scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA,
    bool needFoldR) const {
  LDBG("buildAscendGroupReduce: dimR=" << dimR << " dimA=" << dimA
                                       << " needFoldR=" << needFoldR);
  int64_t blockDataLen = getBlockDataLen(builder.inElemTy);
  if (blockDataLen <= 0) {
    LDBG("buildAscendGroupReduce: skipping - invalid blockDataLen="
         << blockDataLen);
    return;
  }

  auto loopA =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(loopA.getBody());
  Value loopAAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();

  auto safeRead2D = [&rewriter, loc, blockDataLen, &builder, ivA,
                     inputTensor](Value rIdxVal, Value remainRVal) -> Value {
    auto mask2DTy = VectorType::get({1, blockDataLen}, rewriter.getI1Type());
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value mask2D = rewriter.create<vector::CreateMaskOp>(
        loc, mask2DTy, ValueRange{c1Val, remainRVal});

    auto sliceTy =
        RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
    SmallVector<OpFoldResult> inOffsets = {ivA, rIdxVal};
    SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), remainRVal};
    SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);

    auto vec2DTy = VectorType::get({1, blockDataLen}, builder.inElemTy);
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation *>(nullptr),
        [&builder, loc, vec2DTy, slice](OpBuilder &b, Operation *) {
          Value readOp =
              b.create<vector::TransferReadOp>(
                   loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                   builder.inputCstZero, llvm::ArrayRef<bool>{true, false})
                  .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });

    Value computeRead = builder.castToComputeTy(maskOp.getResult(0));

    auto computeVec2DTy =
        VectorType::get({1, blockDataLen}, builder.computeElemTy);
    Value zeroVec = rewriter.create<vector::BroadcastOp>(
        loc, computeVec2DTy, builder.computeCstZero);
    Value cleanRead =
        rewriter.create<arith::SelectOp>(loc, mask2D, computeRead, zeroVec);

    return cleanRead;
  };

  SmallVector<OpFoldResult> accOffsets = {ivA};
  SmallVector<OpFoldResult> accSizes = {rewriter.getIndexAttr(1)};
  SmallVector<OpFoldResult> accStrides = {rewriter.getIndexAttr(1)};
  auto accSliceTy = RankedTensorType::get({1}, builder.accElemTy);

  Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, accSliceTy, loopAAcc, accOffsets, accSizes, accStrides);

  auto accVecTy = VectorType::get({1}, builder.accElemTy);
  Value accRead = rewriter.create<vector::TransferReadOp>(
      loc, accVecTy, extractedSlice, ValueRange{builder.c0}, builder.accCstZero,
      llvm::ArrayRef<bool>{true});

  Value accReadCompute = accRead;
  auto accVecComputeTy = VectorType::get({1}, builder.computeElemTy);
  if (builder.accElemTy != builder.computeElemTy) {
    accReadCompute =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::ExtFOp>(loc, accVecComputeTy, accRead)
                  .getResult()
            : rewriter.create<arith::ExtSIOp>(loc, accVecComputeTy, accRead)
                  .getResult();
  }

  Value finalBlockVec2D;

  if (needFoldR) {
    int64_t innerFoldNum = (dimR + blockDataLen - 1) / blockDataLen;

    Value dimRVal0 = rewriter.create<arith::ConstantIndexOp>(
        loc, std::min(dimR, blockDataLen));
    Value foldedVec = safeRead2D(builder.c0, dimRVal0);

    for (int64_t loopR = 1; loopR < innerFoldNum; ++loopR) {
      Value offsetR =
          rewriter.create<arith::ConstantIndexOp>(loc, loopR * blockDataLen);
      int64_t remainR = dimR - loopR * blockDataLen;
      Value remainRVal = rewriter.create<arith::ConstantIndexOp>(
          loc, std::min(remainR, blockDataLen));

      Value vreg1Compute = safeRead2D(offsetR, remainRVal);
      foldedVec = rewriter.create<arith::AddFOp>(loc, foldedVec, vreg1Compute);
    }
    finalBlockVec2D = foldedVec;
  } else {
    Value dimRVal = rewriter.create<arith::ConstantIndexOp>(loc, dimR);
    finalBlockVec2D = safeRead2D(builder.c0, dimRVal);
  }

  SmallVector<int64_t> reduceDims = {1};
  auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
      loc, vector::CombiningKind::ADD, finalBlockVec2D, accReadCompute,
      rewriter.getI64ArrayAttr(reduceDims));
  multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
  Value toWriteAcc = multiReduceOp.getResult();

  if (builder.accElemTy != builder.computeElemTy) {
    toWriteAcc =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::TruncFOp>(loc, accVecTy, toWriteAcc)
                  .getResult()
            : rewriter.create<arith::TruncIOp>(loc, accVecTy, toWriteAcc)
                  .getResult();
  }

  Value writtenSlice =
      rewriter
          .create<vector::TransferWriteOp>(loc, toWriteAcc, extractedSlice,
                                           ValueRange{builder.c0},
                                           llvm::ArrayRef<bool>{true})
          .getResult();

  Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
      loc, writtenSlice, loopAAcc, accOffsets, accSizes, accStrides);

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
  rewriter.setInsertionPointAfter(loopA);
  rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::rewriteSingleAxisAR(
    IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder,
    scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA,
    RankedTensorType accTensorType) const {
  LDBG("rewriteSingleAxisAR: dimR=" << dimR << " dimA=" << dimA);

  auto inputType = cast<RankedTensorType>(inputTensor.getType());
  assert(inputType.getRank() == 2 && inputType.getDimSize(0) == dimA &&
         inputType.getDimSize(1) == dimR &&
         "single-axis AR expects a [dimA, dimR] tensor");
  assert(accTensorType.getNumElements() == dimA &&
         "single-axis AR result must match the accumulator");

  ReductionStagePlan stage;
  stage.inputShape = {dimA, dimR};
  stage.outputShape = {dimA};
  stage.reductionDims = {1};
  stage.reductionBegin = 1;
  stage.reductionRank = 1;
  stage.outer = dimA;
  stage.dimR = dimR;
  stage.inner = 1;

  bool canReuseInput = builder.inElemTy == builder.computeElemTy &&
                       !hasUseOutsideScope(inputTensor, targetForOp);
  Value merged = buildARReductionStage(
      rewriter, loc, inputTensor, inputType.getElementType(),
      builder.computeElemTy, builder.vectorLength, stage, canReuseInput,
      targetForOp.getInitArgs()[0]);
  rewriter.replaceOp(targetForOp, merged);
}

void TreeReduceV2Pass::buildAR(IRRewriter &rewriter, Location loc,
                               TreeReductionBuilder &builder,
                               scf::ForOp targetForOp, Value inputTensor,
                               int64_t dimR, int64_t dimA,
                               RankedTensorType accTensorType) const {

  LDBG("buildAR: dimR=" << dimR << " dimA=" << dimA);
  int64_t blockDataLen = getBlockDataLen(builder.inElemTy);
  int64_t vlElems = builder.vectorLength;

  ReduceType groupReduceType = ReduceType::IS_REDUCE_SUM;

  if (dimR == 1) {
    buildAscendCopyOut(rewriter, loc, builder, targetForOp, inputTensor, dimR,
                       dimA);
    return;
  } else if (dimR <= vlElems) {
    if (groupReduceType != ReduceType::OTHERS) {
      if (dimR == blockDataLen) {
        buildAscendGroupReduce(rewriter, loc, builder, targetForOp, inputTensor,
                               dimR, dimA, false);
      } else if (dimR < vlElems / 2) {
        buildAscendGroupReduce(rewriter, loc, builder, targetForOp, inputTensor,
                               dimR, dimA, true);
      } else {
        buildAscendShortReduceAR(rewriter, loc, builder, targetForOp,
                                 inputTensor, dimR, dimA);
      }
    } else {
      buildAscendShortReduceAR(rewriter, loc, builder, targetForOp, inputTensor,
                               dimR, dimA);
    }
    return;
  }

  if (ascendStyleAR) {
    rewriteSingleAxisAR(rewriter, loc, builder, targetForOp, inputTensor, dimR,
                        dimA, accTensorType);
    return;
  }

  auto loopA =
      rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1,
                                  ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(loopA.getBody());
  Value loopAAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();

  int64_t mainR = (dimR / vlElems) * vlElems;
  int64_t tailR = dimR % vlElems;
  int64_t folds = (vlElems > 0) ? mainR / vlElems : 0;
  static constexpr int64_t numVectorRegs = 16;
  int64_t mainTimes = folds / numVectorRegs;
  int64_t tailFolds = folds % numVectorRegs;
  LDBG("mainR=" << mainR << " tailR=" << tailR << " folds=" << folds
                << " mainTimes=" << mainTimes << " tailFolds=" << tailFolds);

  Value accVec = rewriter.create<vector::BroadcastOp>(
      loc, builder.vec1DComputeTy, builder.computeCstZero);

  if (mainTimes > 0) {
    Value mainTimesVal =
        rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
    auto loopMain = rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal,
                                                builder.c1, ValueRange{accVec});
    rewriter.setInsertionPointToStart(loopMain.getBody());
    Value iterAccVec = loopMain.getRegionIterArg(0);
    Value ivMain = loopMain.getInductionVar();

    SmallVector<Value> vRegs;
    vRegs.reserve(numVectorRegs);
    for (int i = 0; i < numVectorRegs; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * vlElems);
      Value blockOffset = rewriter.create<arith::MulIOp>(
          loc, ivMain,
          rewriter.create<arith::ConstantIndexOp>(loc,
                                                  numVectorRegs * vlElems));
      Value rIdx = rewriter.create<arith::AddIOp>(loc, blockOffset, iVal);
      Value readRaw =
          builder.readSliceMaskedAR(inputTensor, ivA, rIdx, nullptr,
                                    builder.vec1DInTy, builder.inputCstZero);
      vRegs.push_back(builder.castToComputeTy(readRaw));
    }
    Value reducedVec = builder.buildTreeReduction(vRegs);
    Value nextAccVec = builder.createAdd(iterAccVec, reducedVec);
    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAccVec});
    rewriter.setInsertionPointAfter(loopMain);
    accVec = loopMain.getResult(0);
  }

  if (tailFolds > 0) {
    SmallVector<Value> vRegs;
    vRegs.reserve(tailFolds);
    Value baseOffset = rewriter.create<arith::ConstantIndexOp>(
        loc, mainTimes * numVectorRegs * vlElems);
    for (int i = 0; i < tailFolds; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * vlElems);
      Value rIdx = rewriter.create<arith::AddIOp>(loc, baseOffset, iVal);
      Value readRaw =
          builder.readSliceMaskedAR(inputTensor, ivA, rIdx, nullptr,
                                    builder.vec1DInTy, builder.inputCstZero);
      vRegs.push_back(builder.castToComputeTy(readRaw));
    }
    Value reducedVec = builder.buildTreeReduction(vRegs);
    accVec = builder.createAdd(accVec, reducedVec);
  }

  if (tailR > 0) {
    Value rIdx = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);

    auto mask2DTy = VectorType::get({1, vlElems}, rewriter.getI1Type());
    Value mask2D = rewriter.create<vector::CreateMaskOp>(
        loc, mask2DTy, ValueRange{builder.c1, tailRVal});

    auto sliceTy =
        RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
    SmallVector<OpFoldResult> inOffsets = {ivA, rIdx};
    SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), tailRVal};
    SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);
    auto vec2DTy = VectorType::get({1, vlElems}, builder.inElemTy);
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation *>(nullptr),
        [loc, vec2DTy, slice, &builder](OpBuilder &b, Operation *) {
          Value readOp =
              b.create<vector::TransferReadOp>(
                   loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                   builder.inputCstZero, llvm::ArrayRef<bool>{true, false})
                  .getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });

    auto vec1DTy = VectorType::get({vlElems}, builder.inElemTy);
    Value raw1D =
        rewriter.create<vector::ShapeCastOp>(loc, vec1DTy, maskOp.getResult(0));

    auto mask1DTy = VectorType::get({vlElems}, rewriter.getI1Type());
    Value mask1D = rewriter.create<vector::CreateMaskOp>(loc, mask1DTy,
                                                         ValueRange{tailRVal});
    Value computeRead = builder.castToComputeTy(raw1D);
    Value zeroVec = rewriter.create<vector::BroadcastOp>(
        loc, builder.vec1DComputeTy, builder.computeCstZero);
    Value cleanRead =
        rewriter.create<arith::SelectOp>(loc, mask1D, computeRead, zeroVec);

    accVec = builder.createAdd(accVec, cleanRead);
  }
  Type elemTy = builder.computeElemTy;
  auto vec2DTy = VectorType::get({1, vlElems}, elemTy);
  Value accVec2D = rewriter.create<vector::ShapeCastOp>(loc, vec2DTy, accVec);

  auto vec1xAccTy = VectorType::get({1}, builder.accElemTy);
  auto vec1xComputeTy = VectorType::get({1}, elemTy);

  SmallVector<OpFoldResult> offsets = {ivA};
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
  SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
  auto sliceTy = RankedTensorType::get({1}, builder.accElemTy);

  Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, sliceTy, loopAAcc, offsets, sizes, strides);

  Value accRead = rewriter.create<vector::TransferReadOp>(
      loc, vec1xAccTy, extractedSlice, ValueRange{builder.c0},
      builder.accCstZero);

  Value accReadCompute = accRead;
  if (builder.accElemTy != elemTy) {
    accReadCompute =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::ExtFOp>(loc, vec1xComputeTy, accRead)
                  .getResult()
            : rewriter.create<arith::ExtSIOp>(loc, vec1xComputeTy, accRead)
                  .getResult();
  }
  SmallVector<int64_t> reduceDims = {1};
  ArrayAttr reduceDimsAttr = rewriter.getI64ArrayAttr(reduceDims);
  auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
      loc, vector::CombiningKind::ADD, accVec2D, accReadCompute,
      reduceDimsAttr);
  multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
  Value toWriteAcc = multiReduceOp.getResult();
  if (builder.accElemTy != elemTy) {
    toWriteAcc =
        isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::TruncFOp>(loc, vec1xAccTy, toWriteAcc)
                  .getResult()
            : rewriter.create<arith::TruncIOp>(loc, vec1xAccTy, toWriteAcc)
                  .getResult();
  }
  Value writtenSlice =
      rewriter
          .create<vector::TransferWriteOp>(loc, toWriteAcc, extractedSlice,
                                           ValueRange{builder.c0})
          .getResult();

  Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
      loc, writtenSlice, loopAAcc, offsets, sizes, strides);

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
  rewriter.setInsertionPointAfter(loopA);
  rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

static bool isCVCases(ModuleOp moduleOp) {
  auto result = moduleOp.walk([](Operation *op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcOp->hasAttr(hivm::TPartOfMixAttr::name))
        return WalkResult::interrupt();
    }
    if (isa<scope::ScopeOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

static FailureOr<scf::ForOp>
findMultiAxisLoopNestRoot(vector::MultiDimReductionOp reduceOp,
                          RankedTensorType accumulatorType) {
  scf::ForOp current = reduceOp->getParentOfType<scf::ForOp>();
  if (!current || current.getInitArgs().size() != 1 ||
      current->getNumResults() != 1 ||
      current.getInitArgs()[0].getType() != accumulatorType)
    return failure();

  while (auto parent = current->getParentOfType<scf::ForOp>()) {
    if (parent.getInitArgs().size() != 1 || parent->getNumResults() != 1 ||
        parent.getInitArgs()[0].getType() != accumulatorType)
      return failure();

    Block *body = parent.getBody();
    auto yield = dyn_cast<scf::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != current.getResult(0) ||
        current.getInitArgs()[0] != parent.getRegionIterArg(0) ||
        std::distance(body->begin(), body->end()) != 2 ||
        &body->front() != current.getOperation())
      return failure();
    current = parent;
  }
  return current;
}

void TreeReduceV2Pass::runOnOperation() {
  auto moduleOp = getOperation();
  LDBG("=== Processing func: " << moduleOp.getSymName()
                               << " enableRA=" << enableRA
                               << " enableAR=" << enableAR << " ===");

  if (directRegisterRA && !onlyMarked) {
    moduleOp.emitError(
        "direct-register-ra requires the only-marked safety contract");
    signalPassFailure();
    return;
  }

  // VF fusion is fully bypassed on part_of_mix kernels,
  // so TreeReduceV2 should also skip them.
  if (isCVCases(moduleOp) && !onlyMarked && !onlyLegacyScope) {
    LDBG("=== Skipping CV case ===");
    if (onlyLegacyScope)
      moduleOp->removeAttr(hfusion::kLegacyTreeReductionScopeAttr);
    return;
  }

  if (onlyLegacyScope &&
      !moduleOp->hasAttr(hfusion::kLegacyTreeReductionScopeAttr)) {
    LDBG("=== Skipping non-legacy tree-reduction scope ===");
    return;
  }

  llvm::SmallVector<vector::MultiDimReductionOp> reduceOps;
  moduleOp.walk([&](vector::MultiDimReductionOp reduceOp) {
    reduceOps.push_back(reduceOp);
  });
  LDBG("Found " << reduceOps.size() << " MultiDimReductionOps");

  for (auto targetReduceOp : reduceOps) {
    LDBG("--- Processing MultiDimReductionOp: " << *targetReduceOp);

    // TreeReduceV2 only handles ADD reductions.
    if (targetReduceOp.getKind() != vector::CombiningKind::ADD) {
      LDBG("Skipping - not an ADD reduction (kind="
           << vector::stringifyCombiningKind(targetReduceOp.getKind()) << ")");
      continue;
    }

    llvm::DenseSet<Value> seen;
    SmallVector<Value> inputCandidates;
    collectOriginalTensorCandidates(targetReduceOp.getSource(), seen,
                                    inputCandidates);

    Value inputTensor;
    LogicalReductionShape logicalShape;

    scf::ForOp targetForOp = targetReduceOp->getParentOfType<scf::ForOp>();
    if (!targetForOp) {
      LDBG("Skipping - no parent scf::ForOp");
      continue;
    }
    if (targetForOp.getInitArgs().empty()) {
      LDBG("Skipping - ForOp has no init args");
      continue;
    }
    if (targetForOp->getNumResults() != 1) {
      LDBG("Skipping - ForOp has multiple results, not supported");
      continue;
    }

    auto accTensorType =
        dyn_cast<RankedTensorType>(targetForOp.getInitArgs()[0].getType());
    if (!accTensorType) {
      LDBG("Skipping - accumulator is not a ranked tensor");
      continue;
    }

    bool hasAmbiguousInput = false;
    for (Value candidate : inputCandidates) {
      auto inputType = dyn_cast<RankedTensorType>(candidate.getType());
      if (!inputType)
        continue;
      auto inferredShape =
          inferLogicalReductionShape(targetReduceOp, inputType, accTensorType);
      if (failed(inferredShape))
        continue;
      if (inputType.getRank() != 2) {
        llvm::DenseSet<Value> pathSeen;
        if (!hasIdentityAxisMapping(targetReduceOp.getSource(), candidate,
                                    pathSeen))
          continue;
      }

      if (inputTensor && inputTensor != candidate) {
        hasAmbiguousInput = true;
        break;
      }
      inputTensor = candidate;
      logicalShape = std::move(*inferredShape);
    }
    if (!inputTensor || hasAmbiguousInput) {
      LDBG("Skipping - could not infer a unique reduction input");
      continue;
    }
    LLVM_DEBUG({
      DBGS() << "Inferred reduction shape: input=[";
      llvm::interleaveComma(logicalShape.inputShape, llvm::dbgs());
      llvm::dbgs() << "] reduce axes=[";
      llvm::interleaveComma(logicalShape.reductionDims, llvm::dbgs());
      llvm::dbgs() << "] reduce shape=[";
      llvm::interleaveComma(logicalShape.reductionShape, llvm::dbgs());
      llvm::dbgs() << "] parallel axes=[";
      llvm::interleaveComma(logicalShape.parallelDims, llvm::dbgs());
      llvm::dbgs() << "] output=[";
      llvm::interleaveComma(logicalShape.parallelShape, llvm::dbgs());
      llvm::dbgs() << "] RA permutation=[";
      llvm::interleaveComma(logicalShape.raPermutation, llvm::dbgs());
      llvm::dbgs() << "] equivalent RA shape=[" << logicalShape.dimR << ", "
                   << logicalShape.dimA << "]\n";
    });

    // Shape recovery is useful independently of rewrite eligibility. Keep it
    // ahead of the register-tree marker check so debug-only validation can
    // inspect reductions that AutoVectorizeV2 chose not to rewrite.
    bool isMultiAxis = logicalShape.reductionDims.size() > 1;
    if (onlyMarked && isMultiAxis) {
      auto loopNestRoot =
          findMultiAxisLoopNestRoot(targetReduceOp, accTensorType);
      if (failed(loopNestRoot)) {
        LDBG("Skipping - multi-axis reduction is not a perfect loop nest");
        continue;
      }
      targetForOp = *loopNestRoot;
      LDBG("Selected structurally safe multi-axis loop nest");
    } else if (onlyMarked) {
      scf::ForOp markedLoop;
      for (Operation *ancestor = targetReduceOp.getOperation(); ancestor;
           ancestor = ancestor->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(ancestor);
            loop && loop->hasAttr(hfusion::kRegisterTreeReductionLoopAttr)) {
          markedLoop = loop;
          break;
        }
        if (isa<func::FuncOp>(ancestor))
          break;
      }
      if (!markedLoop) {
        LDBG("Skipping - reduction loop is not marked for register tree");
        continue;
      }
      // Replace exactly the loop selected by AutoVectorizeV2. Climbing above
      // the marker could erase unrelated wrapper-loop work if later pipeline
      // changes introduce another enclosing scf.for.
      targetForOp = markedLoop;
    } else {
      while (auto parentFor = targetForOp->getParentOfType<scf::ForOp>())
        targetForOp = parentFor;
    }
    if (targetForOp.getInitArgs().empty()) {
      LDBG("Skipping tree rewrite - ForOp has no init args");
      continue;
    }
    if (targetForOp->getNumResults() != 1) {
      LDBG("Skipping tree rewrite - ForOp has multiple results");
      continue;
    }
    auto rewriteAccTensorType =
        dyn_cast<RankedTensorType>(targetForOp.getInitArgs()[0].getType());
    if (!rewriteAccTensorType || rewriteAccTensorType != accTensorType) {
      LDBG("Skipping tree rewrite - accumulator type changed across loops");
      continue;
    }

    IRRewriter rewriter(targetForOp.getContext());
    Location loc = targetForOp.getLoc();
    rewriter.setInsertionPoint(targetForOp);

    auto inputTensorType = cast<RankedTensorType>(inputTensor.getType());
    Type inElemTy = inputTensorType.getElementType();
    if (!inElemTy.isF32() && !inElemTy.isF16() && !inElemTy.isBF16()) {
      LDBG("Skipping - unsupported element type: " << inElemTy);
      continue;
    }
    Type computeElemTy = inElemTy;
    if (isa<FloatType>(inElemTy) && inElemTy.getIntOrFloatBitWidth() < 32)
      computeElemTy = rewriter.getF32Type();
    int64_t vectorLength = getVectorLength(computeElemTy);
    int64_t blockDataLen = getBlockDataLen(inElemTy);
    if (vectorLength <= 0 || blockDataLen <= 0) {
      LDBG("Skipping - invalid vectorLength="
           << vectorLength << " blockDataLen=" << blockDataLen);
      continue;
    }

    if (isMultiAxis) {
      auto stagePlans = buildReductionStagePlan(logicalShape);
      if (failed(stagePlans)) {
        LDBG("Skipping multi-axis tree rewrite - invalid stage plan");
        continue;
      }
      LDBG("Building multi-axis tree reduction: stages="
           << stagePlans->size() << " computeElemTy=" << computeElemTy
           << " vectorLength=" << vectorLength);
      buildMultiAxisReduction(rewriter, loc, targetForOp, inputTensor,
                              accTensorType, logicalShape, computeElemTy,
                              vectorLength);
      LDBG("--- Done processing MultiDimReductionOp");
      continue;
    }

    // The existing tree builders use two tensor indices and a rank-1 (or
    // scalar) accumulator. Keep higher-rank reductions at analysis only.
    int64_t dimR = -1, dimA = -1;
    bool isAR = false;
    if (!isValid2DReduction(targetReduceOp, inputTensor, dimR, dimA, isAR)) {
      LDBG("Skipping tree rewrite - equivalent 2D view not materialized");
      continue;
    }

    // Early exit if the corresponding direction is disabled.
    if (isAR && !enableAR) {
      LDBG("Skipping - AR disabled (enableAR=false)");
      continue;
    }
    if (!isAR && !enableRA) {
      LDBG("Skipping - RA disabled (enableRA=false)");
      continue;
    }
    LDBG("elemTy=" << inElemTy << " computeElemTy=" << computeElemTy
                   << " vectorLength=" << vectorLength
                   << " blockDataLen=" << blockDataLen);

    TreeReductionBuilder builder(rewriter, loc, inElemTy,
                                 accTensorType.getElementType(), dimA,
                                 vectorLength);
    LDBG("Building " << (isAR ? "AR" : "RA") << " reduction: dimR=" << dimR
                     << " dimA=" << dimA << " accType=" << accTensorType);

    if (directRegisterRA && !isAR && dimR <= 64 && accTensorType.getRank() == 1)
      buildDirectRegisterRA(rewriter, loc, builder, targetForOp, inputTensor,
                            dimR, dimA, accTensorType);
    else if (isAR)
      buildAR(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA,
              accTensorType);
    else if (ascendStyleRA && dimA > 1)
      rewriteSingleAxisRA(rewriter, loc, builder, targetForOp, inputTensor,
                          dimR, dimA, accTensorType);
    else
      buildRA(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA,
              accTensorType);

    LDBG("--- Done processing MultiDimReductionOp");
  }
  if (onlyLegacyScope)
    moduleOp->removeAttr(hfusion::kLegacyTreeReductionScopeAttr);
  LDBG("=== Done processing func: " << moduleOp.getSymName() << " ===");
}

} // namespace

std::unique_ptr<Pass>
hfusion::createTreeReduceV2Pass(const TreeReduceV2Options &options) {
  return std::make_unique<TreeReduceV2Pass>(options);
}
