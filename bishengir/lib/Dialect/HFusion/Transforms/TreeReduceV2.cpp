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
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Debug.h"

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
bool isValidTreeReducePattern(linalg::GenericOp op, SmallVector<int> &ReductionIndex, Block &body, bool isNotMulti) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.empty()) return false;
  
  for (size_t i = 0; i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::reduction) ReductionIndex.push_back(i);
  }
  if (ReductionIndex.empty() || ((ReductionIndex.size() == 1) ^ isNotMulti)) return false;
  if (!isa<IntOp, FloatOp>(body.front()) || body.getOperations().size() > 2) return false;
  
  auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!inputType || !all_of(inputType.getShape(), [](int64_t sz) { return !ShapedType::isDynamic(sz); })) return false;
  if (inputType.getElementType().getIntOrFloatBitWidth() == 1) return false;
  return true;
}

TypedAttr createFillAttr(Operation &frontOp, RankedTensorType inputType, PatternRewriter &rewriter, int addType) {
  auto elemType = inputType.getElementType();
  if (addType == 0) {
    if (isa<arith::AddIOp>(frontOp)) return rewriter.getIntegerAttr(elemType, 0);
    if (isa<arith::MaxSIOp>(frontOp)) return rewriter.getIntegerAttr(elemType, llvm::APInt::getSignedMinValue(elemType.getIntOrFloatBitWidth()));
    if (isa<arith::MinSIOp>(frontOp)) return rewriter.getIntegerAttr(elemType, llvm::APInt::getSignedMaxValue(elemType.getIntOrFloatBitWidth()));
  } else {
    auto &sem = dyn_cast<mlir::FloatType>(elemType).getFloatSemantics();
    if (isa<arith::AddFOp>(frontOp)) return rewriter.getFloatAttr(elemType, 0.0);
    if (isa<arith::MaxNumFOp>(frontOp)) return rewriter.getFloatAttr(elemType, llvm::APFloat::getInf(sem, true));
    if (isa<arith::MinNumFOp>(frontOp)) return rewriter.getFloatAttr(elemType, llvm::APFloat::getInf(sem, false));
  }
  llvm_unreachable("not complete yet");
}

int calculateCurrentReductionDim(int reductionDim, SmallVector<int> &beenReductionDim) {
  int currentDim = reductionDim;
  for (int prev : beenReductionDim) {
    if (prev < reductionDim) currentDim--;
  }
  return currentDim;
}

template <typename IntOp, typename FloatOp>
void splitMultiReduction(PatternRewriter &rewriter, linalg::GenericOp op, RankedTensorType &currentType,
                         Value &currentResult, SmallVector<int> &ReductionIndex, SmallVector<int> &beenReductionDim,
                         Value output, Block &body, int addType) {
  for (size_t idx = 0; idx < ReductionIndex.size(); ++idx) {
    int reductionDim = ReductionIndex[idx];
    int currentReductionDim = calculateCurrentReductionDim(reductionDim, beenReductionDim);

    SmallVector<int64_t> intermediateShape(currentType.getShape().begin(), currentType.getShape().end());
    intermediateShape.erase(intermediateShape.begin() + currentReductionDim);
    auto intermediateType = RankedTensorType::get(intermediateShape, currentType.getElementType());
    Value initTensor;

    if (idx == ReductionIndex.size() - 1) {
      initTensor = output;
      if (currentType.getShape().back() == 1) {
        SmallVector<ReassociationExprs, 4> reassociationMap(currentType.getRank() - 1);
        for (int i = 0; i < currentType.getRank() - 1; i++) reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
        reassociationMap.back().push_back(rewriter.getAffineDimExpr(currentType.getRank() - 1));
        currentResult = rewriter.create<tensor::CollapseShapeOp>(op.getLoc(), currentResult, reassociationMap);
        currentType = intermediateType;
        beenReductionDim.push_back(reductionDim);
        break;
      }
    } else {
      initTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), intermediateShape, currentType.getElementType());
      TypedAttr zeroAttr = createFillAttr(body.front(), currentType, rewriter, addType);
      Value zeroVal = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
      initTensor = rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange{zeroVal}, ValueRange{initTensor}).getResult(0);
    }

    SmallVector<AffineMap> indexingMaps = {
      AffineMap::getMultiDimIdentityMap(currentType.getRank(), rewriter.getContext())
    };

    SmallVector<AffineExpr> outputExprs;
    for (int64_t i = 0; i < currentType.getRank(); ++i) {
      if (i != currentReductionDim) outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    indexingMaps.push_back(AffineMap::get(currentType.getRank(), 0, outputExprs, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterTypes(currentType.getRank(), utils::IteratorType::parallel);
    iterTypes[currentReductionDim] = utils::IteratorType::reduction;

    auto reduceOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), intermediateType, ValueRange{currentResult}, ValueRange{initTensor}, indexingMaps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = addType == 0 ? b.create<IntOp>(loc, args[0], args[1]).getResult()
                                      : b.create<FloatOp>(loc, args[0], args[1]).getResult();
          b.create<linalg::YieldOp>(loc, result);
        });
        
    currentResult = reduceOp.getResult(0);
    currentType = intermediateType;
    beenReductionDim.push_back(reductionDim);
  }
}

template <typename IntOp, typename FloatOp>
void convertStridedReduceToMultiReduce(func::FuncOp func, PatternRewriter &rewriter) {
  auto originalInsertPoint = rewriter.saveInsertionPoint();
  SmallVector<linalg::GenericOp> opList;
  func.walk([&](linalg::GenericOp genericOp) { opList.push_back(genericOp); });
  
  for (auto op : opList) {
    rewriter.setInsertionPoint(op);
    SmallVector<int> ReductionIndex;
    Block &body = op.getRegion().front();
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body, false)) continue;
    
    int addType = isa<FloatOp>(body.front());
    Value currentResult = op.getInputs()[0];
    auto currentType = cast<RankedTensorType>(currentResult.getType());
    llvm::sort(ReductionIndex);
    
    SmallVector<int> beenReductionDim;
    rewriter.setInsertionPointAfter(op);
    splitMultiReduction<IntOp, FloatOp>(rewriter, op, currentType, currentResult, ReductionIndex,
                                        beenReductionDim, op.getOutputs()[0], body, addType);
    op->getResult(0).replaceAllUsesWith(currentResult);
    op->erase();
  }
  rewriter.restoreInsertionPoint(originalInsertPoint);
}

Value padReductionInput(PatternRewriter &rewriter, Location loc, Value input,
                        int64_t reductionDim, bool addType, Operation *reductionOp) {
  RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> newShape(inputType.getShape().begin(), inputType.getShape().end());
  newShape[reductionDim] += 1;
  
  Value paddedInput = rewriter.create<tensor::EmptyOp>(loc, newShape, inputType.getElementType());
  TypedAttr zeroAttr = createFillAttr(*reductionOp, inputType, rewriter, addType);
  Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  paddedInput = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{paddedInput}).getResult(0);

  SmallVector<OpFoldResult> offsets(inputType.getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(), rewriter.getIndexAttr(1));
  for (int64_t dimSize : inputType.getShape()) sizes.push_back(rewriter.getIndexAttr(dimSize));
  
  Value inputSubview = rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes, strides);
  return rewriter.create<tensor::InsertSliceOp>(loc, inputSubview, paddedInput, offsets, sizes, strides);
}

template <typename IntOp, typename FloatOp>
Value handleLargeChunkCase(PatternRewriter &rewriter, Location loc, Value input,
                           SmallVector<OpFoldResult> offsets, SmallVector<OpFoldResult> sizes,
                           SmallVector<OpFoldResult> strides, Value acc, Value corresponding_chunk,
                           SmallVector<AffineMap> indexingMaps_reduce, SmallVector<utils::IteratorType> iteratorTypes_reduce,
                           SmallVector<utils::IteratorType> iteratorTypes, RankedTensorType inputType,
                           int addType, bool needMerge, Operation *reductionOp) {
  auto tmpType = dyn_cast<RankedTensorType>(corresponding_chunk.getType());
  auto accType = dyn_cast<RankedTensorType>(acc.getType());
  Value initTensor;
  int count = 0;
  bool needInsert = false;
  
  if (accType.getRank() == 0) {
    Value empty = rewriter.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{}, tmpType.getElementType());
    TypedAttr zeroAttr = createFillAttr(*reductionOp, tmpType, rewriter, addType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{empty}).getResult(0);
  } else {
    SmallVector<int64_t> newShape(tmpType.getShape().begin(), tmpType.getShape().end());
    for (auto iteratorType : iteratorTypes_reduce) {
      if (iteratorType == utils::IteratorType::reduction) {
        newShape.erase(newShape.begin() + count);
        break;
      }
      count++;
    }
    needInsert = (newShape != accType.getShape());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, newShape, tmpType.getElementType());
    TypedAttr zeroAttr = createFillAttr(*reductionOp, tmpType, rewriter, addType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{empty}).getResult(0);
  }

  auto current_chunk = rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets, sizes, strides);
  SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(inputType.getRank()));

  auto added = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{corresponding_chunk.getType()}, ValueRange{current_chunk.getResult(), corresponding_chunk},
      ValueRange{current_chunk.getResult()}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value sum = addType == 0 ? nestedBuilder.create<IntOp>(nestedLoc, blockArgs[0], blockArgs[1]).getResult()
                                 : nestedBuilder.create<FloatOp>(nestedLoc, blockArgs[0], blockArgs[1]).getResult();
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });

  if (needMerge) {
    Value targetAcc = needInsert ? initTensor : acc;
    added = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{targetAcc.getType()}, ValueRange{added.getResult(0)}, ValueRange{targetAcc},
        indexingMaps_reduce, iteratorTypes_reduce,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
          Value sum = addType == 0 ? nestedBuilder.create<IntOp>(nestedLoc, blockArgs[0], blockArgs[1]).getResult()
                                   : nestedBuilder.create<FloatOp>(nestedLoc, blockArgs[0], blockArgs[1]).getResult();
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
        });

    if (needInsert) {
      offsets.erase(offsets.begin() + count);
      sizes.erase(sizes.begin() + count);
      strides.erase(strides.begin() + count);
      return rewriter.create<tensor::InsertSliceOp>(loc, added.getResult(0), acc, offsets, sizes, strides).getResult();
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
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body, true)) continue;

    int addType = isa<FloatOp>(body.front());
    Value input = op.getInputs()[0];
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t vectorSize = inputType.getElementType().getIntOrFloatBitWidth() == 64 ? 64 :
                         64 * 32 / static_cast<int64_t>(inputType.getElementType().getIntOrFloatBitWidth());
    int64_t dim0Value = inputType.getShape()[ReductionIndex[0]];
    
    if (dim0Value <= vectorSize) continue;

    if (inputType.getRank() - 1 == ReductionIndex[0]) {
      auto insertPoint = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfterValue(input);
      int remaining_value = 2 * 32 / (inputType.getElementType().getIntOrFloatBitWidth() / 8);
      if (inputType.getShape().back() % remaining_value) {
        SmallVector<int64_t> newShape(inputType.getShape().begin(), inputType.getShape().end());
        newShape.back() = (newShape.back() + remaining_value - 1) / remaining_value * remaining_value;
        auto resultType = RankedTensorType::get(newShape, inputType.getElementType());
        
        Value empty = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultType.getShape(), resultType.getElementType());
        TypedAttr zeroAttr = createFillAttr(body.front(), inputType, rewriter, addType);
        Value zeroVal = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
        Value initTensor = rewriter.create<linalg::FillOp>(op.getLoc(), ValueRange{zeroVal}, ValueRange{empty}).getResult(0);
        
        SmallVector<OpFoldResult> offsets(inputType.getRank(), rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(inputType.getRank(), rewriter.getIndexAttr(1));
        for (int64_t dimSize : inputType.getShape()) sizes.push_back(rewriter.getIndexAttr(dimSize));
        
        Value paddedInput = rewriter.create<tensor::InsertSliceOp>(op.getLoc(), input, initTensor, offsets, sizes, strides);
        rewriter.replaceUsesWithIf(input, paddedInput, [&](OpOperand &opOperand) { return opOperand.getOwner() != paddedInput.getDefiningOp(); });
        
        input = paddedInput;
        inputType = resultType;
        dim0Value = inputType.getShape()[ReductionIndex[0]];
      }
      rewriter.restoreInsertionPoint(insertPoint);
    }

    Location loc = op.getLoc();
    if (dim0Value % 2 != 0) {
      input = padReductionInput(rewriter, loc, input, ReductionIndex[0], addType, &body.front());
      inputType = cast<RankedTensorType>(input.getType());
      dim0Value = inputType.getShape()[ReductionIndex[0]];
    }

    int64_t dimXValue = inputType.getShape().back();
    int64_t half_point_value = inputType.getRank() - 1 == ReductionIndex[0] ? (dimXValue + 1) / 2 : dimXValue; 
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
        offsets2.push_back(rewriter.getIndexAttr(inputType.getShape()[dim] / 2));
        iteratorTypes_reduce.push_back(utils::IteratorType::reduction);
      } else {
        sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[dim]));
        sizes2.push_back(rewriter.getIndexAttr(inputType.getShape()[dim]));
        offsets2.push_back(rewriter.getIndexAttr(0));
        iteratorTypes_reduce.push_back(utils::IteratorType::parallel);
      }
    }

    for (int64_t i = 0; i <= 1; i++) {
      int64_t current_start_value = (i == 0 ? 0 : half_point_value < vectorSize ? half_point_value : half_point_value / vectorSize * vectorSize);
      int64_t remaining_value = half_point_value - current_start_value;
      if (!remaining_value) break;

      int64_t chunk_size_value = std::min(remaining_value, std::max(vectorSize, half_point_value / vectorSize * vectorSize));
      int64_t corresponding_start_value = current_start_value + (inputType.getRank() - 1 == ReductionIndex[0] ? half_point_value : 0);
      int64_t corresponding_size_value = std::max((int64_t)0, std::min(chunk_size_value, dimXValue - corresponding_start_value));
      
      offsets.back() = rewriter.getIndexAttr(current_start_value);
      offsets2.back() = rewriter.getIndexAttr(corresponding_start_value);
      sizes.back() = rewriter.getIndexAttr(corresponding_size_value);
      
      SmallVector<AffineMap> indexingMaps_reduce = {
          rewriter.getMultiDimIdentityMap(inputType.getRank()),
          AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext()).dropResult(ReductionIndex[0])
      };
      
      Value corresponding_chunk = rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets2, sizes, strides);
      bool needMerge = (cast<RankedTensorType>(corresponding_chunk.getType()).getShape()[ReductionIndex[0]] != 0);
      
      acc = handleLargeChunkCase<IntOp, FloatOp>(
          rewriter, loc, input, offsets, sizes, strides, acc, corresponding_chunk, indexingMaps_reduce, iteratorTypes_reduce,
          iteratorTypes, inputType, addType, needMerge, &body.front());
    }
    op->getResult(0).replaceAllUsesWith(acc);
    op->erase();
  }
  rewriter.restoreInsertionPoint(originalInsertPoint);
}

struct TreeReducePattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp func, PatternRewriter &rewriter) const override {
    auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
    if (!hacc::utils::isDevice(func) ||
        (fusionKind.has_value() && (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
                                    fusionKind.value() == mlir::hfusion::FusionKind::SingleCube))) {
      return failure();
    }
    convertStridedReduceToMultiReduce<arith::AddIOp, arith::AddFOp>(func, rewriter);
    convertStridedReduceToMultiReduce<arith::MaxSIOp, arith::MaxNumFOp>(func, rewriter);
    convertStridedReduceToMultiReduce<arith::MinSIOp, arith::MinNumFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::AddIOp, arith::AddFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MaxSIOp, arith::MaxNumFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MinSIOp, arith::MinNumFOp>(func, rewriter);
    return success();
  }
};

Value traceBackToOriginalTensor(Value val) {
  Value currentVal = val;
  while (currentVal) {
    Operation *defOp = currentVal.getDefiningOp();
    if (!defOp) break;

    if (auto maskOp = dyn_cast<vector::MaskOp>(defOp)) {
      vector::TransferReadOp innerRead = nullptr;
      maskOp.walk([&](vector::TransferReadOp op) { innerRead = op; });
      if (innerRead) { currentVal = innerRead.getSource(); continue; }
    }
    if (auto readOp = dyn_cast<vector::TransferReadOp>(defOp)) { currentVal = readOp.getSource(); continue; }
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) { currentVal = extractOp.getSource(); continue; }
    if (auto extfOp = dyn_cast<arith::ExtFOp>(defOp)) { currentVal = extfOp.getIn(); continue; }
    if (auto truncfOp = dyn_cast<arith::TruncFOp>(defOp)) { currentVal = truncfOp.getIn(); continue; }
    if (auto extiOp = dyn_cast<arith::ExtSIOp>(defOp)) { currentVal = extiOp.getIn(); continue; }
    break;
  }
  return currentVal;
}

class TreeReductionBuilder {
public:
  IRRewriter &rewriter;
  Location loc;
  Type inElemTy, accElemTy, computeElemTy;
  VectorType vec1DInTy, vec2DInTy, vec1DAccTy, vec1DComputeTy;
  Value c0, c1, c16, c64, dimAVal, inputCstZero, accCstZero, computeCstZero;
  int64_t dimA;

  TreeReductionBuilder(IRRewriter &rewriter, Location loc, Type inTy, Type accTy, int64_t dimA)
      : rewriter(rewriter), loc(loc), inElemTy(inTy), accElemTy(accTy), dimA(dimA) {
    c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    c64 = rewriter.create<arith::ConstantIndexOp>(loc, 64);
    dimAVal = rewriter.create<arith::ConstantIndexOp>(loc, dimA);
    
    computeElemTy = inElemTy;
    if (isa<FloatType>(inElemTy) && inElemTy.getIntOrFloatBitWidth() < 32) computeElemTy = rewriter.getF32Type();

    vec1DInTy = VectorType::get({64}, inElemTy);
    vec2DInTy = VectorType::get({1, 64}, inElemTy);
    vec1DAccTy = VectorType::get({64}, accElemTy);
    vec1DComputeTy = VectorType::get({64}, computeElemTy);
    
    inputCstZero = rewriter.create<arith::ConstantOp>(loc, inElemTy, rewriter.getZeroAttr(inElemTy));
    accCstZero = rewriter.create<arith::ConstantOp>(loc, accElemTy, rewriter.getZeroAttr(accElemTy));
    computeCstZero = rewriter.create<arith::ConstantOp>(loc, computeElemTy, rewriter.getZeroAttr(computeElemTy));
  }

  Value castToComputeTy(Value vec) {
    if (inElemTy == computeElemTy) return vec;
    if (isa<FloatType>(inElemTy)) return rewriter.create<arith::ExtFOp>(loc, vec1DComputeTy, vec);
    return rewriter.create<arith::ExtSIOp>(loc, vec1DComputeTy, vec);
  }

  Value createAdd(Value lhs, Value rhs) {
    Type ty = lhs.getType();
    if (auto vecTy = dyn_cast<VectorType>(ty)) ty = vecTy.getElementType();
    
    if (isa<FloatType>(ty)) return rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
  }

  Value buildTreeReduction(SmallVector<Value> &currentLevel) {
    while (currentLevel.size() > 1) {
      SmallVector<Value> nextLevel;
      int halfSize = currentLevel.size() / 2;
      nextLevel.reserve(halfSize + (currentLevel.size() % 2));
      for (int i = 0; i < halfSize; ++i) nextLevel.push_back(createAdd(currentLevel[i], currentLevel[i + halfSize]));
      if (currentLevel.size() % 2 != 0) nextLevel.push_back(currentLevel.back());
      currentLevel = std::move(nextLevel);
    }
    return currentLevel.front();
  }

  Value createLoopMask1D(Value minBoundary) {
    return rewriter.create<vector::CreateMaskOp>(loc, VectorType::get({64}, rewriter.getI1Type()), ValueRange{minBoundary});
  }

  Value readSliceMaskedRA(Value tensor, Value rIdx, Value aIdx, Value mask1D, VectorType vecTy, Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, (Operation *)nullptr, [&](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
              loc, vecTy, tensor, ValueRange{rIdx, aIdx}, cstZero, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }
  
  Value read1DAccMasked(Value tensor, Value idx, Value mask1D, VectorType vecTy, Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, (Operation *)nullptr, [&](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
              loc, vecTy, tensor, ValueRange{idx}, cstZero, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }

  Value readSliceMaskedAR(Value tensor, Value aIdx, Value rIdx, Value mask1D, VectorType vecTy, Value cstZero) {
    if (mask1D) {
      auto maskOp = rewriter.create<vector::MaskOp>(
          loc, TypeRange{vecTy}, mask1D, (Operation *)nullptr, [&](OpBuilder &b, Operation *) {
            Value readOp = b.create<vector::TransferReadOp>(
                loc, vecTy, tensor, ValueRange{aIdx, rIdx}, cstZero, ArrayRef<bool>{false}).getResult();
            b.create<vector::YieldOp>(loc, readOp);
          });
      return maskOp.getResult(0);
    } else {
      return rewriter.create<vector::TransferReadOp>(
          loc, vecTy, tensor, ValueRange{aIdx, rIdx}, cstZero, ArrayRef<bool>{false}).getResult();
    }
  }

  Value write1DToSlice(Value vec1D, Value tensor, Value rIdx, Value aIdx, Value minA, Value mask1D) {
    auto elemTy = cast<RankedTensorType>(tensor.getType()).getElementType();
    auto sliceTy = RankedTensorType::get({ShapedType::kDynamic}, elemTy);
    SmallVector<OpFoldResult> offsets = {rIdx, aIdx};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, tensor, offsets, sizes, strides);
    
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{sliceTy}, mask1D, (Operation *)nullptr, [&](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(loc, vec1D, slice, ValueRange{c0}, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    return rewriter.create<tensor::InsertSliceOp>(loc, maskOp.getResult(0), tensor, offsets, sizes, strides).getResult();
  }

  Value promoteToComputeType(Value inputTensor, int64_t dimR, AffineMap minMap) {
    if (inElemTy == computeElemTy) return inputTensor;
    auto emptyTy = RankedTensorType::get({dimR, dimA}, computeElemTy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, emptyTy.getShape(), computeElemTy);
    Value dimRVal = rewriter.create<arith::ConstantIndexOp>(loc, dimR);
    
    auto loopR = rewriter.create<scf::ForOp>(loc, c0, dimRVal, c1, ValueRange{emptyTensor});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value rAcc = loopR.getRegionIterArg(0);
    Value ivR = loopR.getInductionVar();
    
    auto loopA = rewriter.create<scf::ForOp>(loc, c0, dimAVal, c64, ValueRange{rAcc});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value aAcc = loopA.getRegionIterArg(0);
    Value ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, dimAVal});
    Value mask1D = createLoopMask1D(minA);
    
    Value rawRead = readSliceMaskedRA(inputTensor, ivR, ivA, mask1D, vec1DInTy, inputCstZero);
    Value castedRead = castToComputeTy(rawRead);
    Value insertOp = write1DToSlice(castedRead, aAcc, ivR, ivA, minA, mask1D);
    
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertOp});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopA.getResult(0)});
    rewriter.setInsertionPointAfter(loopR);
    return loopR.getResult(0);
  }
};

struct TreeReduceV2Pass : public impl::TreeReduceV2Base<TreeReduceV2Pass> {
  using TreeReduceV2Base<TreeReduceV2Pass>::TreeReduceV2Base;
  void runOnOperation() override;

private:
  bool isValid2DReduction(vector::MultiDimReductionOp reduceOp, Value inputTensor, int64_t &dimR, int64_t &dimA, bool &isAR) {
    if (reduceOp.getKind() != vector::CombiningKind::ADD) return false;
    auto reduceDims = reduceOp.getReductionDims();
    if (reduceDims.empty() || reduceDims.size() != 1) return false;
    
    int64_t reducedDim = cast<mlir::IntegerAttr>(reduceDims[0]).getInt();
    auto rankedTy = dyn_cast_or_null<RankedTensorType>(inputTensor.getType());
    if (!rankedTy || rankedTy.getRank() != 2) return false;
    
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

  void buildRA(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType);
  void buildAR(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType);
};

void TreeReduceV2Pass::buildRA(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType) {
  auto wrapperLoop = rewriter.create<scf::ForOp>(loc, builder.c0, builder.c1, builder.c1, ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());
  Value initAcc = wrapperLoop.getRegionIterArg(0); 

  AffineMap minMap = AffineMap::get(1, 1, {rewriter.getAffineConstantExpr(64), rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)}, rewriter.getContext());
  Value workingTensor = builder.promoteToComputeType(inputTensor, dimR, minMap);

  int64_t mainR = 1;
  while (mainR * 2 <= dimR) mainR *= 2;
  int64_t tailR = dimR - mainR;
  int64_t folds = llvm::Log2_64(mainR);
  int64_t mainTimes = folds / 4;
  int64_t tailFolds = folds % 4;

  if (tailR > 0) {
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mainRVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, builder.dimAVal});
    Value currentMask = builder.createLoopMask1D(minA);
    
    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, tailRVal, builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    Value tailIdx = rewriter.create<arith::AddIOp>(loc, ivR, mainRVal); 
    
    Value vecMain = builder.readSliceMaskedRA(loopRAcc, ivR, ivA, currentMask, builder.vec1DComputeTy, builder.computeCstZero);
    Value vecTail = builder.readSliceMaskedRA(loopRAcc, tailIdx, ivA, currentMask, builder.vec1DComputeTy, builder.computeCstZero);
    Value reducedVec = builder.createAdd(vecMain, vecTail);
    Value insertSliceOp = builder.write1DToSlice(reducedVec, loopRAcc, ivR, ivA, minA, currentMask);
    
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    workingTensor = loopA.getResult(0); 
  }

  if (mainTimes > 0) {
    Value loopRNumVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    Value mainTimesVal = rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
    auto loopMain = rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal, builder.c1, ValueRange{workingTensor, loopRNumVal});
    rewriter.setInsertionPointToStart(loopMain.getBody());
    
    Value loopMainAcc = loopMain.getRegionIterArg(0), currentDimR = loopMain.getRegionIterArg(1);
    Value nextRNumVal = rewriter.create<arith::DivUIOp>(loc, currentDimR, builder.c16);
    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{loopMainAcc});
    rewriter.setInsertionPointToStart(loopA.getBody());
    
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, builder.dimAVal});
    Value currentMask = builder.createLoopMask1D(minA);
    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, nextRNumVal, builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    
    SmallVector<Value> vRegs;
    vRegs.reserve(16);
    for (int i = 0; i < 16; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value offsetR = rewriter.create<arith::MulIOp>(loc, iVal, nextRNumVal);
      Value rIdx = rewriter.create<arith::AddIOp>(loc, ivR, offsetR);
      vRegs.push_back(builder.readSliceMaskedRA(loopRAcc, rIdx, ivA, currentMask, builder.vec1DComputeTy, builder.computeCstZero));
    }
    
    Value finalVecAdd = builder.buildTreeReduction(vRegs);
    Value insertSliceOp = builder.write1DToSlice(finalVecAdd, loopRAcc, ivR, ivA, minA, currentMask);
    
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopA.getResult(0), nextRNumVal});
    rewriter.setInsertionPointAfter(loopMain);
    workingTensor = loopMain.getResult(0);
  }

  int numLeaves = 1 << tailFolds; 
  auto loopATail = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{initAcc});
  rewriter.setInsertionPointToStart(loopATail.getBody());
  Value finalLoopAAcc = loopATail.getRegionIterArg(0), ivA = loopATail.getInductionVar();
  Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, builder.dimAVal});
  Value currentMask = builder.createLoopMask1D(minA);
  
  SmallVector<Value> tailRegs;
  tailRegs.reserve(numLeaves);
  for (int i = 0; i < numLeaves; ++i) {
    Value offsetR = rewriter.create<arith::ConstantIndexOp>(loc, i);
    tailRegs.push_back(builder.readSliceMaskedRA(workingTensor, offsetR, ivA, currentMask, builder.vec1DComputeTy, builder.computeCstZero));
  }
  Value finalOutputVec = builder.buildTreeReduction(tailRegs); 

  if (accTensorType.getRank() == 0) {
    VectorType vec0DType = VectorType::get({}, builder.accElemTy);
    Value finalScalar = rewriter.create<vector::ExtractElementOp>(loc, finalOutputVec, builder.c0);
    
    Value oldScalar = rewriter.create<tensor::ExtractOp>(loc, finalLoopAAcc, ValueRange{});
    Value oldScalarCompute = oldScalar;
    if (builder.accElemTy != builder.computeElemTy) {
      oldScalarCompute = isa<FloatType>(builder.accElemTy) ? rewriter.create<arith::ExtFOp>(loc, builder.computeElemTy, oldScalar).getResult()
                                                           : rewriter.create<arith::ExtSIOp>(loc, builder.computeElemTy, oldScalar).getResult();
    }
    
    Value totalSumCompute = builder.createAdd(finalScalar, oldScalarCompute);
    
    Value finalSum = totalSumCompute;
    if (builder.accElemTy != builder.computeElemTy) {
      finalSum = isa<FloatType>(builder.accElemTy) ? rewriter.create<arith::TruncFOp>(loc, builder.accElemTy, totalSumCompute).getResult()
                                                   : rewriter.create<arith::TruncIOp>(loc, builder.accElemTy, totalSumCompute).getResult();
    }
    Value broadcastOp = rewriter.create<vector::BroadcastOp>(loc, vec0DType, finalSum);
    Value writeOp = rewriter.create<vector::TransferWriteOp>(loc, broadcastOp, finalLoopAAcc, ValueRange{}).getResult();
    rewriter.create<scf::YieldOp>(loc, ValueRange{writeOp});
  } else {
    auto accSlice = rewriter.create<tensor::ExtractSliceOp>(loc, finalLoopAAcc, SmallVector<OpFoldResult>{ivA}, SmallVector<OpFoldResult>{minA}, SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
    
    Value oldAccVec = builder.read1DAccMasked(finalLoopAAcc, ivA, currentMask, builder.vec1DAccTy, builder.accCstZero);
    Value oldAccCompute = builder.castToComputeTy(oldAccVec);
    Value totalSumCompute = builder.createAdd(finalOutputVec, oldAccCompute);
    
    Value toWriteAcc = totalSumCompute;
    if (builder.accElemTy != builder.computeElemTy) {
      toWriteAcc = isa<FloatType>(builder.accElemTy) ? rewriter.create<arith::TruncFOp>(loc, builder.vec1DAccTy, totalSumCompute).getResult()
                                                     : rewriter.create<arith::TruncIOp>(loc, builder.vec1DAccTy, totalSumCompute).getResult();
    }
    
    auto writeMaskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{accSlice.getType()}, currentMask, (Operation *)nullptr, [&](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(loc, toWriteAcc, accSlice, ValueRange{builder.c0}, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    Value insertSliceOp = rewriter.create<tensor::InsertSliceOp>(loc, writeMaskOp.getResult(0), finalLoopAAcc, SmallVector<OpFoldResult>{ivA}, SmallVector<OpFoldResult>{minA}, SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
  }

  rewriter.setInsertionPointAfter(loopATail);
  rewriter.create<scf::YieldOp>(loc, ValueRange{loopATail.getResult(0)});
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

void TreeReduceV2Pass::buildAR(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType) {
  auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1, ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(loopA.getBody());
  Value loopAAcc = loopA.getRegionIterArg(0);
  Value ivA = loopA.getInductionVar();

  int64_t mainR = (dimR / 64) * 64;
  int64_t tailR = dimR % 64;
  int64_t folds = mainR / 64;
  int64_t mainTimes = folds / 16;
  int64_t tailFolds = folds % 16;

  Value accVec = rewriter.create<vector::BroadcastOp>(loc, builder.vec1DComputeTy, builder.computeCstZero);

  if (mainTimes > 0) {
    Value mainTimesVal = rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
    auto loopMain = rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal, builder.c1, ValueRange{accVec});
    rewriter.setInsertionPointToStart(loopMain.getBody());
    Value iterAccVec = loopMain.getRegionIterArg(0);
    Value ivMain = loopMain.getInductionVar();
    
    SmallVector<Value> vRegs;
    vRegs.reserve(16);
    for (int i = 0; i < 16; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * 64);
      Value blockOffset = rewriter.create<arith::MulIOp>(loc, ivMain, rewriter.create<arith::ConstantIndexOp>(loc, 1024));
      Value rIdx = rewriter.create<arith::AddIOp>(loc, blockOffset, iVal);
      Value readRaw = builder.readSliceMaskedAR(inputTensor, ivA, rIdx, nullptr, builder.vec1DInTy, builder.inputCstZero);
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
    Value baseOffset = rewriter.create<arith::ConstantIndexOp>(loc, mainTimes * 1024);
    for (int i = 0; i < tailFolds; ++i) {
      Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * 64);
      Value rIdx = rewriter.create<arith::AddIOp>(loc, baseOffset, iVal);
      Value readRaw = builder.readSliceMaskedAR(inputTensor, ivA, rIdx, nullptr, builder.vec1DInTy, builder.inputCstZero);
      vRegs.push_back(builder.castToComputeTy(readRaw));
    }
    Value reducedVec = builder.buildTreeReduction(vRegs);
    accVec = builder.createAdd(accVec, reducedVec);
  }

  if (tailR > 0) {
    Value rIdx = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mask = builder.createLoopMask1D(tailRVal);
    Value readRaw = builder.readSliceMaskedAR(inputTensor, ivA, rIdx, mask, builder.vec1DInTy, builder.inputCstZero);
    accVec = builder.createAdd(accVec, builder.castToComputeTy(readRaw));
  }

  Value scalar = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, accVec);

  if (builder.accElemTy != builder.computeElemTy) {
    scalar = isa<FloatType>(builder.accElemTy) ? rewriter.create<arith::TruncFOp>(loc, builder.accElemTy, scalar).getResult()
                                               : rewriter.create<arith::TruncIOp>(loc, builder.accElemTy, scalar).getResult();
  }

  Value currentVal;
  if (accTensorType.getRank() == 0) {
    currentVal = rewriter.create<tensor::ExtractOp>(loc, loopAAcc, ValueRange{});
  } else {
    currentVal = rewriter.create<tensor::ExtractOp>(loc, loopAAcc, ValueRange{ivA});
  }
  
  Value finalScalar = builder.createAdd(currentVal, scalar);
  
  Value nextAcc;
  if (accTensorType.getRank() == 0) {
    nextAcc = rewriter.create<tensor::InsertOp>(loc, finalScalar, loopAAcc, ValueRange{});
  } else {
    nextAcc = rewriter.create<tensor::InsertOp>(loc, finalScalar, loopAAcc, ValueRange{ivA});
  }

  rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
  rewriter.setInsertionPointAfter(loopA);
  rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::runOnOperation() {
  auto funcOp = getOperation();
  vector::MultiDimReductionOp targetReduceOp = nullptr;
  funcOp.walk([&](vector::MultiDimReductionOp reduceOp) {
    targetReduceOp = reduceOp;
    return WalkResult::interrupt();
  });
  if (!targetReduceOp) return;

  Value inputTensor = traceBackToOriginalTensor(targetReduceOp.getSource());
  int64_t dimR = -1, dimA = -1;
  bool isAR = false;
  if (!isValid2DReduction(targetReduceOp, inputTensor, dimR, dimA, isAR)) return;
  
  scf::ForOp targetForOp = targetReduceOp->getParentOfType<scf::ForOp>();
  if (!targetForOp) return;
  while (auto parentFor = targetForOp->getParentOfType<scf::ForOp>()) targetForOp = parentFor;
  if (targetForOp.getInitArgs().empty()) return;

  IRRewriter rewriter(targetForOp.getContext());
  Location loc = targetForOp.getLoc();
  rewriter.setInsertionPoint(targetForOp);

  auto inputTensorType = cast<RankedTensorType>(inputTensor.getType());
  auto accTensorType = cast<RankedTensorType>(targetForOp.getInitArgs()[0].getType());
  TreeReductionBuilder builder(rewriter, loc, inputTensorType.getElementType(), accTensorType.getElementType(), dimA);

  if (isAR) {
    buildAR(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA, accTensorType);
  } else {
    buildRA(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA, accTensorType);
  }
}

} // namespace

std::unique_ptr<Pass> hfusion::createTreeReduceV2Pass() {
  return std::make_unique<TreeReduceV2Pass>();
}