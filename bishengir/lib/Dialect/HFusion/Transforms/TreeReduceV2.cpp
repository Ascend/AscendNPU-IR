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
#include "llvm/ADT/STLExtras.h"
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
    if (auto readOp = dyn_cast<vector::TransferReadOp>(defOp)) {
      currentVal = readOp.getSource(); continue;
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(defOp)) {
      currentVal = writeOp.getSource(); continue;
    }
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      currentVal = extractOp.getSource(); continue;
    }
    if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(defOp)) {
      currentVal = insertOp.getDest(); continue;
    }
    if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(defOp)) {
      currentVal = shapeCastOp.getSource(); continue;
    }
    if (auto extfOp = dyn_cast<arith::ExtFOp>(defOp)) {
      currentVal = extfOp.getIn(); continue;
    }
    if (auto truncfOp = dyn_cast<arith::TruncFOp>(defOp)) {
      currentVal = truncfOp.getIn(); continue;
    }
    if (auto extiOp = dyn_cast<arith::ExtSIOp>(defOp)) {
      currentVal = extiOp.getIn(); continue;
    }
    if (auto trunciOp = dyn_cast<arith::TruncIOp>(defOp)) {
      currentVal = trunciOp.getIn(); continue;
    }
    if (auto fptosiOp = dyn_cast<arith::FPToSIOp>(defOp)) {
      currentVal = fptosiOp.getIn(); continue;
    }
    if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(defOp)) {
      currentVal = sitofpOp.getIn(); continue;
    }

    if (auto addfOp = dyn_cast<arith::AddFOp>(defOp)) {
      currentVal = addfOp.getLhs(); continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      if (forOp->hasAttr("reductionLoop")) {
        (void)forOp.walk([&](tensor::ExtractSliceOp sliceOp) {
          Value source = sliceOp.getSource();
          if (!llvm::is_contained(forOp.getRegionIterArgs(), source)) {
            currentVal = source;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        break;
      }
      break;
    }
    LDBG("traceBackToOriginalTensor: stopping at unrecognized op: " << defOp->getName());
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
  int64_t vectorLength;

  TreeReductionBuilder(IRRewriter &rewriter, Location loc, Type inTy, Type accTy, int64_t dimA, int64_t vLen)
      : rewriter(rewriter), loc(loc), inElemTy(inTy), accElemTy(accTy), dimA(dimA), vectorLength(vLen) {
    c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    c64 = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    dimAVal = rewriter.create<arith::ConstantIndexOp>(loc, dimA);
    
    computeElemTy = inElemTy;
    if (isa<FloatType>(inElemTy) && inElemTy.getIntOrFloatBitWidth() < 32) computeElemTy = rewriter.getF32Type();

    vec1DInTy = VectorType::get({vectorLength}, inElemTy);
    vec2DInTy = VectorType::get({1, vectorLength}, inElemTy);
    vec1DAccTy = VectorType::get({vectorLength}, accElemTy);
    vec1DComputeTy = VectorType::get({vectorLength}, computeElemTy);
    
    inputCstZero = rewriter.create<arith::ConstantOp>(loc, inElemTy, rewriter.getZeroAttr(inElemTy));
    accCstZero = rewriter.create<arith::ConstantOp>(loc, accElemTy, rewriter.getZeroAttr(accElemTy));
    computeCstZero = rewriter.create<arith::ConstantOp>(loc, computeElemTy, rewriter.getZeroAttr(computeElemTy));
  }

  Value castToComputeTy(Value vec) {
    auto vecTy = dyn_cast<VectorType>(vec.getType());
    if (!vecTy) return vec; 
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
    return rewriter.create<vector::CreateMaskOp>(loc,
        VectorType::get({vectorLength}, rewriter.getI1Type()),
        ValueRange{minBoundary});
  }

  Value readSliceMaskedRA(Value tensor, Value rIdx, Value aIdx, Value mask1D, VectorType vecTy, Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, static_cast<Operation*>(nullptr), [this, vecTy, tensor, rIdx, aIdx, cstZero](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
              loc, vecTy, tensor, ValueRange{rIdx, aIdx}, cstZero, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }
  
  Value read1DAccMasked(Value tensor, Value idx, Value mask1D, VectorType vecTy, Value cstZero) {
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vecTy}, mask1D, static_cast<Operation*>(nullptr), [this, vecTy, tensor, idx, cstZero](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
              loc, vecTy, tensor, ValueRange{idx}, cstZero, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    return maskOp.getResult(0);
  }

  Value readSliceMaskedAR(Value tensor, Value aIdx, Value rIdx, Value mask1D, VectorType vecTy, Value cstZero) {
    if (mask1D) {
      auto maskOp = rewriter.create<vector::MaskOp>(
          loc, TypeRange{vecTy}, mask1D, static_cast<Operation*>(nullptr), [this, vecTy, tensor, aIdx, rIdx, cstZero](OpBuilder &b, Operation *) {
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
        loc, TypeRange{sliceTy}, mask1D, static_cast<Operation*>(nullptr), [this, vec1D, slice](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(loc, vec1D, slice, ValueRange{c0}, ArrayRef<bool>{false}).getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    return rewriter.create<tensor::InsertSliceOp>(loc, maskOp.getResult(0), tensor, offsets, sizes, strides).getResult();
  }

  Value promoteToComputeType(Value inputTensor, int64_t dimR, AffineMap minMap) {
    if (inElemTy == computeElemTy) return inputTensor;
    LDBG("Promoting to compute type: " << inElemTy << " -> " << computeElemTy);
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
  int64_t getVectorLength(Type elemTy) const {
    int64_t bytesPerElement = elemTy.getIntOrFloatBitWidth() / 8;
    if (bytesPerElement == 0) bytesPerElement = 1;
    return vectorBytes / bytesPerElement;
  }
  int64_t getBlockDataLen(Type elemTy) const {
    int64_t bytesPerElement = elemTy.getIntOrFloatBitWidth() / 8;
    if (bytesPerElement == 0) bytesPerElement = 1;
    return blockBytes / bytesPerElement;
  }
  static constexpr int64_t vectorBytes = 256;
  static constexpr int64_t blockBytes = 32;

  enum class ReduceType { 
      IS_REDUCE_SUM, 
      IS_REDUCE_MAX, 
      IS_REDUCE_MIN, 
      OTHERS 
  };

  void buildAscendCopyOut(IRRewriter &rewriter, Location loc, 
                            TreeReductionBuilder &builder, scf::ForOp targetForOp, 
                            Value inputTensor, int64_t dimR, int64_t dimA) const;

  void buildAscendGroupReduce(IRRewriter &rewriter, Location loc, 
                              TreeReductionBuilder &builder, scf::ForOp targetForOp, 
                              Value inputTensor, int64_t dimR, int64_t dimA, 
                              bool needFoldR) const;

  void buildAscendShortReduceAR(IRRewriter &rewriter, Location loc, 
                                TreeReductionBuilder &builder, scf::ForOp targetForOp, 
                                Value inputTensor, int64_t dimR, int64_t dimA) const;

  void buildRA(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType) const;
  void buildAR(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType) const;
};

void TreeReduceV2Pass::buildRA(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, RankedTensorType accTensorType) const {
  LDBG("buildRA: dimR=" << dimR << " dimA=" << dimA);
  auto wrapperLoop = rewriter.create<scf::ForOp>(loc, builder.c0, builder.c1, builder.c1, ValueRange{targetForOp.getInitArgs()[0]});
  rewriter.setInsertionPointToStart(wrapperLoop.getBody());
  Value initAcc = wrapperLoop.getRegionIterArg(0); 

  AffineMap minMap = AffineMap::get(1, 1, {rewriter.getAffineConstantExpr(builder.vectorLength), rewriter.getAffineSymbolExpr(0) - rewriter.getAffineDimExpr(0)}, rewriter.getContext());
  Value workingTensor = builder.promoteToComputeType(inputTensor, dimR, minMap);

  int64_t mainR = 1;
  while (mainR > 0 && mainR * 2 <= dimR) mainR *= 2;
  if (mainR == 0) mainR = 1;
  int64_t tailR = dimR - mainR;
  int64_t folds = llvm::Log2_64(mainR);
  int64_t mainTimes = folds / 4;
  int64_t tailFolds = folds % 4;
  LDBG("mainR=" << mainR << " tailR=" << tailR << " folds=" << folds
                << " mainTimes=" << mainTimes << " tailFolds=" << tailFolds);

  auto read2D = [&rewriter, loc, &builder](Value tensor, Value rIdx, Value cIdx, Value minA, Value mask) -> Value {
      auto sliceTy = RankedTensorType::get({1, ShapedType::kDynamic}, builder.computeElemTy);
      SmallVector<OpFoldResult> offsets = {rIdx, cIdx};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
      Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, tensor, offsets, sizes, strides);
      
      auto vec2DTy = VectorType::get({1, builder.vectorLength}, builder.computeElemTy);
      auto maskOp = rewriter.create<vector::MaskOp>(loc, TypeRange{vec2DTy}, mask, nullptr, [loc, vec2DTy, slice, &builder](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(
              loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0}, builder.computeCstZero, ArrayRef<bool>{true, true}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
      });
      return maskOp.getResult(0);
  };

  auto write2D = [&rewriter, loc, &builder](Value vec2D, Value tensor, Value rIdx, Value cIdx, Value minA, Value mask) -> Value {
      auto sliceTy = RankedTensorType::get({1, ShapedType::kDynamic}, builder.computeElemTy);
      SmallVector<OpFoldResult> offsets = {rIdx, cIdx};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1), minA};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
      Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, tensor, offsets, sizes, strides);
      
      auto maskWriteOp = rewriter.create<vector::MaskOp>(loc, TypeRange{sliceTy}, mask, nullptr, [&builder, loc, vec2D, slice](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(
              loc, vec2D, slice, ValueRange{builder.c0, builder.c0}, ArrayRef<bool>{true, true}).getResult();
          b.create<vector::YieldOp>(loc, writeOp);
      });
      return rewriter.create<tensor::InsertSliceOp>(loc, maskWriteOp.getResult(0), tensor, offsets, sizes, strides).getResult();
  };

  if (tailR > 0) {
    Value tailRVal = rewriter.create<arith::ConstantIndexOp>(loc, tailR);
    Value mainRVal = rewriter.create<arith::ConstantIndexOp>(loc, mainR);
    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{workingTensor});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0), ivA = loopA.getInductionVar();
    Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, builder.dimAVal});
    
    auto mask2DTy = VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
    Value currentMask = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{builder.c1, minA});
    
    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, tailRVal, builder.c1, ValueRange{loopAAcc});
    rewriter.setInsertionPointToStart(loopR.getBody());
    Value loopRAcc = loopR.getRegionIterArg(0), ivR = loopR.getInductionVar();
    Value tailIdx = rewriter.create<arith::AddIOp>(loc, ivR, mainRVal); 
    
    Value vecMain = read2D(loopRAcc, ivR, ivA, minA, currentMask);
    Value vecTail = read2D(loopRAcc, tailIdx, ivA, minA, currentMask);
    Value reducedVec = builder.createAdd(vecMain, vecTail);
    Value insertSliceOp = write2D(reducedVec, loopRAcc, ivR, ivA, minA, currentMask);
    
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
    
    auto mask2DTy = VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
    Value currentMask = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{builder.c1, minA});

    auto loopR = rewriter.create<scf::ForOp>(loc, builder.c0, nextRNumVal, builder.c1, ValueRange{loopAAcc});
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
    Value insertSliceOp = write2D(finalVecAdd, loopRAcc, ivR, ivA, minA, currentMask);
    
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
    rewriter.setInsertionPointAfter(loopR);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopR.getResult(0)});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.create<scf::YieldOp>(loc, ValueRange{loopA.getResult(0), nextRNumVal});
    rewriter.setInsertionPointAfter(loopMain);
    workingTensor = loopMain.getResult(0);
  }

  int64_t numLeaves = (tailFolds < 63) ? (1LL << tailFolds) : INT64_MAX;
  LDBG("numLeaves=" << numLeaves);
  auto loopATail = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c64, ValueRange{initAcc});
  rewriter.setInsertionPointToStart(loopATail.getBody());
  Value finalLoopAAcc = loopATail.getRegionIterArg(0), ivA = loopATail.getInductionVar();
  Value minA = rewriter.create<affine::AffineMinOp>(loc, minMap, ValueRange{ivA, builder.dimAVal});
  
  auto mask2DTy = VectorType::get({1, builder.vectorLength}, rewriter.getI1Type());
  Value currentMask = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{builder.c1, minA});
  
  SmallVector<Value> tailRegs;
  tailRegs.reserve(std::min<int64_t>(numLeaves, 64));
  for (int64_t i = 0; i < numLeaves && i < 64; ++i) {
    Value offsetR = rewriter.create<arith::ConstantIndexOp>(loc, i);
    tailRegs.push_back(read2D(workingTensor, offsetR, ivA, minA, currentMask));
  }
  Value finalOutputVec = builder.buildTreeReduction(tailRegs); 

  auto vec1DTy = VectorType::get({builder.vectorLength}, builder.computeElemTy);
  Value finalOutputVec1D = rewriter.create<vector::ShapeCastOp>(loc, vec1DTy, finalOutputVec);

  if (accTensorType.getRank() == 0) {
    VectorType vec0DType = VectorType::get({}, builder.accElemTy);
    Value finalScalar = rewriter.create<vector::ExtractElementOp>(loc, finalOutputVec1D, builder.c0);
    
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
    auto accSliceTy = RankedTensorType::get({ShapedType::kDynamic}, builder.accElemTy);
    SmallVector<OpFoldResult> accOffsets = {ivA};
    SmallVector<OpFoldResult> accSizes = {minA};
    SmallVector<OpFoldResult> accStrides = {rewriter.getIndexAttr(1)};
    Value accSlice = rewriter.create<tensor::ExtractSliceOp>(loc, accSliceTy, finalLoopAAcc, accOffsets, accSizes, accStrides).getResult();
    
    Value currentMask1D = builder.createLoopMask1D(minA);

    auto maskReadOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{builder.vec1DAccTy}, currentMask1D, static_cast<Operation*>(nullptr), [&builder, loc, accSlice](OpBuilder &b, Operation *) {
          Value readOp = b.create<vector::TransferReadOp>(loc, builder.vec1DAccTy, accSlice, ValueRange{builder.c0}, builder.accCstZero, ArrayRef<bool>{true}).getResult();
          b.create<vector::YieldOp>(loc, readOp);
        });
    Value oldAccVec = maskReadOp.getResult(0);

    Value oldAccCompute = builder.castToComputeTy(oldAccVec);
    Value totalSumCompute = builder.createAdd(finalOutputVec1D, oldAccCompute);
    
    Value toWriteAcc = totalSumCompute;
    if (builder.accElemTy != builder.computeElemTy) {
      toWriteAcc = isa<FloatType>(builder.accElemTy) ? rewriter.create<arith::TruncFOp>(loc, builder.vec1DAccTy, totalSumCompute).getResult()
                                                     : rewriter.create<arith::TruncIOp>(loc, builder.vec1DAccTy, totalSumCompute).getResult();
    }
    
    auto writeMaskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{accSliceTy}, currentMask1D, static_cast<Operation*>(nullptr), [&builder, loc, toWriteAcc, accSlice](OpBuilder &b, Operation *) {
          Value writeOp = b.create<vector::TransferWriteOp>(loc, toWriteAcc, accSlice, ValueRange{builder.c0}, ArrayRef<bool>{true}).getResult();
          b.create<vector::YieldOp>(loc, writeOp);
        });
    Value insertSliceOp = rewriter.create<tensor::InsertSliceOp>(loc, writeMaskOp.getResult(0), finalLoopAAcc, accOffsets, accSizes, accStrides);
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp});
  }

  rewriter.setInsertionPointAfter(loopATail);
  rewriter.create<scf::YieldOp>(loc, ValueRange{loopATail.getResult(0)});
  rewriter.setInsertionPointAfter(wrapperLoop);
  rewriter.replaceOp(targetForOp, wrapperLoop.getResult(0));
}

void TreeReduceV2Pass::buildAscendCopyOut(IRRewriter &rewriter, Location loc, 
                                          TreeReductionBuilder &builder, scf::ForOp targetForOp, 
                                          Value inputTensor, int64_t dimR, int64_t dimA) const {
    LDBG("buildAscendCopyOut: dimR=" << dimR << " dimA=" << dimA);

    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1, 
                                            ValueRange{targetForOp.getInitArgs()[0]});
    rewriter.setInsertionPointToStart(loopA.getBody());
    
    Value loopAAcc = loopA.getRegionIterArg(0);
    Value ivA = loopA.getInductionVar();

    Value extractedVal = rewriter.create<tensor::ExtractOp>(
        loc, inputTensor, ValueRange{ivA, builder.c0});

    Value toWriteAcc = extractedVal;
    if (builder.inElemTy != builder.accElemTy) {
        toWriteAcc = isa<FloatType>(builder.accElemTy) ? 
            rewriter.create<arith::ExtFOp>(loc, builder.accElemTy, extractedVal).getResult() : 
            rewriter.create<arith::ExtSIOp>(loc, builder.accElemTy, extractedVal).getResult();
    }

    Value nextAcc = rewriter.create<tensor::InsertOp>(
        loc, toWriteAcc, loopAAcc, ValueRange{ivA});

    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::buildAscendShortReduceAR(IRRewriter &rewriter, Location loc, 
                                                TreeReductionBuilder &builder, scf::ForOp targetForOp, 
                                                Value inputTensor, int64_t dimR, int64_t dimA) const {
    LDBG("buildAscendShortReduceAR: dimR=" << dimR << " dimA=" << dimA);
    int64_t vlElems = builder.vectorLength;
    
    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1, 
                                              ValueRange{targetForOp.getInitArgs()[0]});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0);
    Value ivA = loopA.getInductionVar();

    Value dimRVal = rewriter.create<arith::ConstantIndexOp>(loc, dimR);
    
    auto mask2DTy = VectorType::get({1, vlElems}, rewriter.getI1Type());
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value mask2D = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{c1Val, dimRVal});

    auto sliceTy = RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
    SmallVector<OpFoldResult> inOffsets = {ivA, builder.c0};
    SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), dimRVal};
    SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);

    auto vec2DTy = VectorType::get({1, vlElems}, builder.inElemTy);
    auto maskOp = rewriter.create<vector::MaskOp>(
        loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation*>(nullptr), [&builder, loc, vec2DTy, slice](OpBuilder &b, Operation *) {
            Value readOp = b.create<vector::TransferReadOp>(
                loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                builder.inputCstZero, llvm::ArrayRef<bool>{true, false}
            ).getResult();
            b.create<vector::YieldOp>(loc, readOp);
        });
        
    Value computeRead = builder.castToComputeTy(maskOp.getResult(0));

    auto computeVec2DTy = VectorType::get({1, vlElems}, builder.computeElemTy);
    Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, computeVec2DTy, builder.computeCstZero);
    Value cleanRead = rewriter.create<arith::SelectOp>(loc, mask2D, computeRead, zeroVec);

    SmallVector<OpFoldResult> accOffsets = {ivA};
    SmallVector<OpFoldResult> accSizes = {rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> accStrides = {rewriter.getIndexAttr(1)};
    auto accSliceTy = RankedTensorType::get({1}, builder.accElemTy);
    
    Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, accSliceTy, loopAAcc, accOffsets, accSizes, accStrides);

    auto accVecTy = VectorType::get({1}, builder.accElemTy);
    Value accRead = rewriter.create<vector::TransferReadOp>(
        loc, accVecTy, extractedSlice, ValueRange{builder.c0}, builder.accCstZero, llvm::ArrayRef<bool>{true});

    Value accReadCompute = accRead;
    auto accVecComputeTy = VectorType::get({1}, builder.computeElemTy);
    if (builder.accElemTy != builder.computeElemTy) {
        accReadCompute = isa<FloatType>(builder.accElemTy) ? 
            rewriter.create<arith::ExtFOp>(loc, accVecComputeTy, accRead).getResult() : 
            rewriter.create<arith::ExtSIOp>(loc, accVecComputeTy, accRead).getResult();
    }

    SmallVector<int64_t> reduceDims = {1}; 
    auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, vector::CombiningKind::ADD, cleanRead, accReadCompute, rewriter.getI64ArrayAttr(reduceDims));
    multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    Value toWriteAcc = multiReduceOp.getResult();

    if (builder.accElemTy != builder.computeElemTy) {
        toWriteAcc = isa<FloatType>(builder.accElemTy) ? 
            rewriter.create<arith::TruncFOp>(loc, accVecTy, toWriteAcc).getResult() : 
            rewriter.create<arith::TruncIOp>(loc, accVecTy, toWriteAcc).getResult();
    }

    Value writtenSlice = rewriter.create<vector::TransferWriteOp>(
        loc, toWriteAcc, extractedSlice, ValueRange{builder.c0}, llvm::ArrayRef<bool>{true}).getResult();

    Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
        loc, writtenSlice, loopAAcc, accOffsets, accSizes, accStrides);

    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::buildAscendGroupReduce(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, 
                                              scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, 
                                              bool needFoldR) const {
    LDBG("buildAscendGroupReduce: dimR=" << dimR << " dimA=" << dimA << " needFoldR=" << needFoldR);
    int64_t blockDataLen = getBlockDataLen(builder.inElemTy);
    
    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1, 
                                              ValueRange{targetForOp.getInitArgs()[0]});
    rewriter.setInsertionPointToStart(loopA.getBody());
    Value loopAAcc = loopA.getRegionIterArg(0);
    Value ivA = loopA.getInductionVar();

    auto safeRead2D = [&rewriter, loc, blockDataLen, &builder, ivA, inputTensor](Value rIdxVal, Value remainRVal) -> Value {
        auto mask2DTy = VectorType::get({1, blockDataLen}, rewriter.getI1Type());
        Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value mask2D = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{c1Val, remainRVal});

        auto sliceTy = RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
        SmallVector<OpFoldResult> inOffsets = {ivA, rIdxVal};
        SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), remainRVal};
        SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
        Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);

        auto vec2DTy = VectorType::get({1, blockDataLen}, builder.inElemTy);
        auto maskOp = rewriter.create<vector::MaskOp>(
            loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation*>(nullptr), [&builder, loc, vec2DTy, slice](OpBuilder &b, Operation *) {
                Value readOp = b.create<vector::TransferReadOp>(
                    loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                    builder.inputCstZero, llvm::ArrayRef<bool>{true, false}
                ).getResult();
                b.create<vector::YieldOp>(loc, readOp);
            });
            
        Value computeRead = builder.castToComputeTy(maskOp.getResult(0));

        auto computeVec2DTy = VectorType::get({1, blockDataLen}, builder.computeElemTy);
        Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, computeVec2DTy, builder.computeCstZero);
        Value cleanRead = rewriter.create<arith::SelectOp>(loc, mask2D, computeRead, zeroVec);

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
        loc, accVecTy, extractedSlice, ValueRange{builder.c0}, builder.accCstZero, llvm::ArrayRef<bool>{true});

    Value accReadCompute = accRead;
    auto accVecComputeTy = VectorType::get({1}, builder.computeElemTy);
    if (builder.accElemTy != builder.computeElemTy) {
        accReadCompute = isa<FloatType>(builder.accElemTy) ? 
            rewriter.create<arith::ExtFOp>(loc, accVecComputeTy, accRead).getResult() : 
            rewriter.create<arith::ExtSIOp>(loc, accVecComputeTy, accRead).getResult();
    }

    Value finalBlockVec2D;

    if (needFoldR) {
        int64_t innerFoldNum = (dimR + blockDataLen - 1) / blockDataLen;
        
        Value dimRVal0 = rewriter.create<arith::ConstantIndexOp>(loc, std::min(dimR, blockDataLen));
        Value foldedVec = safeRead2D(builder.c0, dimRVal0);

        for (int64_t loopR = 1; loopR < innerFoldNum; ++loopR) {
            Value offsetR = rewriter.create<arith::ConstantIndexOp>(loc, loopR * blockDataLen);
            int64_t remainR = dimR - loopR * blockDataLen;
            Value remainRVal = rewriter.create<arith::ConstantIndexOp>(loc, std::min(remainR, blockDataLen));
            
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
        loc, vector::CombiningKind::ADD, finalBlockVec2D, accReadCompute, rewriter.getI64ArrayAttr(reduceDims));
    multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    Value toWriteAcc = multiReduceOp.getResult();

    if (builder.accElemTy != builder.computeElemTy) {
        toWriteAcc = isa<FloatType>(builder.accElemTy) ? 
            rewriter.create<arith::TruncFOp>(loc, accVecTy, toWriteAcc).getResult() : 
            rewriter.create<arith::TruncIOp>(loc, accVecTy, toWriteAcc).getResult();
    }

    Value writtenSlice = rewriter.create<vector::TransferWriteOp>(
        loc, toWriteAcc, extractedSlice, ValueRange{builder.c0}, llvm::ArrayRef<bool>{true}).getResult();

    Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
        loc, writtenSlice, loopAAcc, accOffsets, accSizes, accStrides);

    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::buildAR(IRRewriter &rewriter, Location loc, TreeReductionBuilder &builder, 
                                scf::ForOp targetForOp, Value inputTensor, int64_t dimR, int64_t dimA, 
                                RankedTensorType accTensorType) const {

    LDBG("buildAR: dimR=" << dimR << " dimA=" << dimA);
    int64_t blockDataLen = getBlockDataLen(builder.inElemTy);
    int64_t vlElems = builder.vectorLength;

    ReduceType groupReduceType = ReduceType::IS_REDUCE_SUM; 

    if (dimR == 1) {
        buildAscendCopyOut(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA);
        return;
    } else if (dimR <= vlElems) {
        if (groupReduceType != ReduceType::OTHERS) {
            if (dimR == blockDataLen) {
                buildAscendGroupReduce(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA, false);
            } else if (dimR < vlElems / 2) {
                buildAscendGroupReduce(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA, true);
            } else {
                buildAscendShortReduceAR(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA);
            }
        } else {
            buildAscendShortReduceAR(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA);
        }
        return;
    }

    auto loopA = rewriter.create<scf::ForOp>(loc, builder.c0, builder.dimAVal, builder.c1, ValueRange{targetForOp.getInitArgs()[0]});
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

    Value accVec = rewriter.create<vector::BroadcastOp>(loc, builder.vec1DComputeTy, builder.computeCstZero);

    if (mainTimes > 0) {
        Value mainTimesVal = rewriter.create<arith::ConstantIndexOp>(loc, mainTimes);
        auto loopMain = rewriter.create<scf::ForOp>(loc, builder.c0, mainTimesVal, builder.c1, ValueRange{accVec});
        rewriter.setInsertionPointToStart(loopMain.getBody());
        Value iterAccVec = loopMain.getRegionIterArg(0);
        Value ivMain = loopMain.getInductionVar();
        
        SmallVector<Value> vRegs;
        vRegs.reserve(numVectorRegs);
        for (int i = 0; i < numVectorRegs; ++i) {
            Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * vlElems);
            Value blockOffset = rewriter.create<arith::MulIOp>(loc, ivMain, rewriter.create<arith::ConstantIndexOp>(loc, numVectorRegs * vlElems));
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
        Value baseOffset = rewriter.create<arith::ConstantIndexOp>(loc, mainTimes * numVectorRegs * vlElems);
        for (int i = 0; i < tailFolds; ++i) {
            Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i * vlElems);
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

        auto mask2DTy = VectorType::get({1, vlElems}, rewriter.getI1Type());
        Value mask2D = rewriter.create<vector::CreateMaskOp>(loc, mask2DTy, ValueRange{builder.c1, tailRVal});

        auto sliceTy = RankedTensorType::get({1, ShapedType::kDynamic}, builder.inElemTy);
        SmallVector<OpFoldResult> inOffsets = {ivA, rIdx};
        SmallVector<OpFoldResult> inSizes = {rewriter.getIndexAttr(1), tailRVal};
        SmallVector<OpFoldResult> inStrides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
        Value slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceTy, inputTensor, inOffsets, inSizes, inStrides);
        auto vec2DTy = VectorType::get({1, vlElems}, builder.inElemTy);
        auto maskOp = rewriter.create<vector::MaskOp>(
            loc, TypeRange{vec2DTy}, mask2D, static_cast<Operation*>(nullptr), [loc, vec2DTy, slice, &builder](OpBuilder &b, Operation *) {
                Value readOp = b.create<vector::TransferReadOp>(
                    loc, vec2DTy, slice, ValueRange{builder.c0, builder.c0},
                    builder.inputCstZero, llvm::ArrayRef<bool>{true, false}
                ).getResult();
                b.create<vector::YieldOp>(loc, readOp);
            });
            
        auto vec1DTy = VectorType::get({vlElems}, builder.inElemTy);
        Value raw1D = rewriter.create<vector::ShapeCastOp>(loc, vec1DTy, maskOp.getResult(0));
        
        auto mask1DTy = VectorType::get({vlElems}, rewriter.getI1Type());
        Value mask1D = rewriter.create<vector::CreateMaskOp>(loc, mask1DTy, ValueRange{tailRVal});
        Value computeRead = builder.castToComputeTy(raw1D);
        Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, builder.vec1DComputeTy, builder.computeCstZero);
        Value cleanRead = rewriter.create<arith::SelectOp>(loc, mask1D, computeRead, zeroVec);

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
        loc, vec1xAccTy, extractedSlice, ValueRange{builder.c0}, builder.accCstZero);

    Value accReadCompute = accRead;
    if (builder.accElemTy != elemTy) {
        accReadCompute = isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::ExtFOp>(loc, vec1xComputeTy, accRead).getResult()
            : rewriter.create<arith::ExtSIOp>(loc, vec1xComputeTy, accRead).getResult();
    }
    SmallVector<int64_t> reduceDims = {1};
    ArrayAttr reduceDimsAttr = rewriter.getI64ArrayAttr(reduceDims);
    auto multiReduceOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, vector::CombiningKind::ADD, accVec2D, accReadCompute, reduceDimsAttr);
    multiReduceOp->setAttr("withoutInitMergeOp", rewriter.getUnitAttr());
    Value toWriteAcc = multiReduceOp.getResult();
    if (builder.accElemTy != elemTy) {
        toWriteAcc = isa<FloatType>(builder.accElemTy)
            ? rewriter.create<arith::TruncFOp>(loc, vec1xAccTy, toWriteAcc).getResult()
            : rewriter.create<arith::TruncIOp>(loc, vec1xAccTy, toWriteAcc).getResult();
    }
    Value writtenSlice = rewriter.create<vector::TransferWriteOp>(
        loc, toWriteAcc, extractedSlice, ValueRange{builder.c0}).getResult();

    Value nextAcc = rewriter.create<tensor::InsertSliceOp>(
        loc, writtenSlice, loopAAcc, offsets, sizes, strides);

    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});
    rewriter.setInsertionPointAfter(loopA);
    rewriter.replaceOp(targetForOp, loopA.getResult(0));
}

void TreeReduceV2Pass::runOnOperation() {
  auto funcOp = getOperation();
  LDBG("=== Processing func: " << funcOp.getSymName()
                             << " enableRA=" << enableRA
                             << " enableAR=" << enableAR << " ===");

  // VF fusion is fully bypassed on part_of_mix kernels,
  // so TreeReduceV2 should also skip them.
  if (funcOp->hasAttr(hivm::TPartOfMixAttr::name)) {
    LDBG("=== Skipping func: " << funcOp.getSymName() << " (part_of_mix) ===");
    return;
  }

  llvm::SmallVector<vector::MultiDimReductionOp> reduceOps;
  funcOp.walk([&](vector::MultiDimReductionOp reduceOp) {
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

    Value inputTensor = traceBackToOriginalTensor(targetReduceOp.getSource());
    int64_t dimR = -1, dimA = -1;
    bool isAR = false;
    if (!isValid2DReduction(targetReduceOp, inputTensor, dimR, dimA, isAR)) {
      LDBG("Skipping - not a valid 2D reduction (input rank != 2)");
      continue;
    }
    LDBG("Valid 2D reduction: dimR=" << dimR << " dimA=" << dimA
                                     << " isAR=" << isAR);

    auto inputTensorType = cast<RankedTensorType>(inputTensor.getType());
    Type inElemTy = inputTensorType.getElementType();
    if (!inElemTy.isF32() && !inElemTy.isF16() && !inElemTy.isBF16()) {
      LDBG("Skipping - unsupported element type: " << inElemTy);
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

    scf::ForOp targetForOp = targetReduceOp->getParentOfType<scf::ForOp>();
    if (!targetForOp) {
      LDBG("Skipping - no parent scf::ForOp");
      continue;
    }
    while (auto parentFor = targetForOp->getParentOfType<scf::ForOp>())
      targetForOp = parentFor;
    if (targetForOp.getInitArgs().empty()) {
      LDBG("Skipping - ForOp has no init args");
      continue;
    }
    if (targetForOp->getNumResults() != 1) {
      LDBG("Skipping - ForOp has multiple results, not supported");
      continue;
    }

    IRRewriter rewriter(targetForOp.getContext());
    Location loc = targetForOp.getLoc();
    rewriter.setInsertionPoint(targetForOp);

    auto accTensorType =
        cast<RankedTensorType>(targetForOp.getInitArgs()[0].getType());
    Type computeElemTy = inElemTy;
    if (isa<FloatType>(inElemTy) && inElemTy.getIntOrFloatBitWidth() < 32)
      computeElemTy = rewriter.getF32Type();
    int64_t vectorLength = getVectorLength(computeElemTy);
    int64_t blockDataLen = getBlockDataLen(inElemTy);

    if (vectorLength <= 0 || blockDataLen <= 0) {
      LDBG("Skipping - invalid vectorLength=" << vectorLength
                                              << " blockDataLen=" << blockDataLen);
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

    if (isAR)
      buildAR(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA,
              accTensorType);
    else
      buildRA(rewriter, loc, builder, targetForOp, inputTensor, dimR, dimA,
              accTensorType);

    LDBG("--- Done processing MultiDimReductionOp");
  }
  LDBG("=== Done processing func: " << funcOp.getSymName() << " ===");
}

} // namespace

std::unique_ptr<Pass> hfusion::createTreeReduceV2Pass(const TreeReduceV2Options &options) {
  return std::make_unique<TreeReduceV2Pass>(options);
}