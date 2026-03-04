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

using namespace mlir;

namespace {
template <typename IntOp, typename FloatOp>
bool isValidTreeReducePattern(linalg::GenericOp op,
                              SmallVector<int> &ReductionIndex, Block &body,
                              bool isNotMulti) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
    return false;
  }
  // 2. Check if iterator_types includes reduction
  auto iteratorTypes = op.getIteratorTypesArray();
  if (iteratorTypes.size() < 1) {
    return false;
  }
  int i = 0;
  for (auto type : iteratorTypes) {
    if (type == utils::IteratorType::reduction) {
      ReductionIndex.push_back(i);
    }
    i++;
  }
  if (ReductionIndex.empty()) {
    return false;
  }
  if ((ReductionIndex.size() == 1) ^ isNotMulti) {
    return false;
  }
  // 3. Check if it is an addition operation
  if (!isa<IntOp, FloatOp>(body.front())) {
    return false;
  }
  if (body.getOperations().size() > 2) {
    return false;
  }
  auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  if (!inputType) {
    return false;
  } else {
    return all_of(inputType.getShape(),
                  [](int64_t sz) { return !ShapedType::isDynamic(sz); });
  }
  if (inputType.getElementType().getIntOrFloatBitWidth() == 1) {
    return false;
  }
  return true;
}

TypedAttr createFillAttr(Operation &frontOp, RankedTensorType inputType,
                         PatternRewriter &rewriter, int addType) {
  TypedAttr zeroAttr;
  if (addType == 0) { // Integer
    if (isa<arith::AddIOp>(frontOp)) {
      zeroAttr = rewriter.getIntegerAttr(inputType.getElementType(), 0);
    } else if (isa<arith::MaxSIOp>(frontOp)) {
      zeroAttr = rewriter.getIntegerAttr(
          inputType.getElementType(),
          llvm::APInt::getSignedMinValue(
              inputType.getElementType().getIntOrFloatBitWidth()));
    } else if (isa<arith::MinSIOp>(frontOp)) {
      zeroAttr = rewriter.getIntegerAttr(
          inputType.getElementType(),
          llvm::APInt::getSignedMaxValue(
              inputType.getElementType().getIntOrFloatBitWidth()));
    } else {
      llvm_unreachable("not complete yet");
    }
  } else { // float
    auto &sem = dyn_cast<mlir::FloatType>(inputType.getElementType())
                    .getFloatSemantics();
    if (isa<arith::AddFOp>(frontOp)) {
      zeroAttr = rewriter.getFloatAttr(inputType.getElementType(), 0.0);
    } else if (isa<arith::MaxNumFOp>(frontOp)) {
      zeroAttr = rewriter.getFloatAttr(inputType.getElementType(),
                                       llvm::APFloat::getInf(sem, true));
    } else if (isa<arith::MinNumFOp>(frontOp)) {
      zeroAttr = rewriter.getFloatAttr(inputType.getElementType(),
                                       llvm::APFloat::getInf(sem, false));
    } else {
      llvm_unreachable("not complete yet");
    }
  }
  return zeroAttr;
}

int calculateCurrentRecutionDim(int reductionDim,
                                SmallVector<int> &beenReductionDim) {
  int currentReductionDim = reductionDim;
  for (int prevReduced : beenReductionDim) {
    if (prevReduced < reductionDim) {
      currentReductionDim--;
    }
  }
  return currentReductionDim;
}

template <typename IntOp, typename FloatOp>
void splitMultiReduction(PatternRewriter &rewriter, linalg::GenericOp op,
                         RankedTensorType &currentType, Value &currentResult,
                         SmallVector<int> &ReductionIndex,
                         SmallVector<int> &beenReductionDim, Value output,
                         Block &body, int addType) {
  for (size_t idx = 0; idx < ReductionIndex.size(); ++idx) {
    int reductionDim = ReductionIndex[idx];

    // Calculate the actual position of the reduction dimension in the current
    // tensor. Because the previous reduce operation has already changed the
    // dimensions of the tensor.
    int currentReductionDim =
        calculateCurrentRecutionDim(reductionDim, beenReductionDim);

    // Calculate the shape of the intermediate result: remove the current
    // reduction dimension.
    SmallVector<int64_t> intermediateShape(currentType.getShape().begin(),
                                           currentType.getShape().end());
    intermediateShape.erase(intermediateShape.begin() + currentReductionDim);

    // Creating tensor type of intermediate results
    auto intermediateType =
        RankedTensorType::get(intermediateShape, currentType.getElementType());

    // Create init tensor for intermediate results
    Value initTensor;
    if (idx == ReductionIndex.size() - 1) {
      // The final step uses the raw output.
      initTensor = output;
      if (currentType.getShape().back() == 1) {
        SmallVector<ReassociationExprs, 4> reassociationMap(
            currentType.getShape().size() - 1);
        for (size_t i = 0; i < currentType.getShape().size() - 1; i++) {
          reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
        }
        reassociationMap.back().push_back(
            rewriter.getAffineDimExpr(currentType.getShape().size() - 1));
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
      Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
      // Fill the entire tensor using linalg.generic
      auto identityMap =
          rewriter.getMultiDimIdentityMap(intermediateShape.size());
      initTensor =
          rewriter
              .create<linalg::GenericOp>(
                  op.getLoc(), intermediateType,
                  /*inputs=*/ValueRange{},
                  /*outputs=*/ValueRange{initTensor},
                  /*indexingMaps=*/ArrayRef<AffineMap>{identityMap},
                  /*iteratorTypes=*/
                  SmallVector<utils::IteratorType>(
                      intermediateShape.size(), utils::IteratorType::parallel),
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    b.create<linalg::YieldOp>(loc, zero);
                  })
              .getResult(0);
    }

    // Building indexing maps for the current step
    SmallVector<AffineMap> indexingMaps;

    // Input map: identity mapping, using the rank of currentType
    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(
        currentType.getRank(), rewriter.getContext());
    indexingMaps.push_back(inputMap);

    // Output map: Remove the reduction dimension
    SmallVector<AffineExpr> outputExprs;
    for (int64_t i = 0; i < currentType.getRank(); ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    outputExprs.erase(outputExprs.begin() + currentReductionDim);

    AffineMap outputMap = AffineMap::get(currentType.getRank(), 0, outputExprs,
                                         rewriter.getContext());
    indexingMaps.push_back(outputMap);
    // Construct iterator types: the current dimension is reduction, others
    // are parallel.
    SmallVector<utils::IteratorType> iterTypes(currentType.getRank(),
                                               utils::IteratorType::parallel);
    iterTypes[currentReductionDim] = utils::IteratorType::reduction;

    // Create the reduce operation for the current step
    auto reduceOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), intermediateType, ValueRange{currentResult},
        ValueRange{initTensor}, indexingMaps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result;
          if (addType == 0) {
            result = b.create<IntOp>(loc, args[0], args[1]);
          } else {
            result = b.create<FloatOp>(loc, args[0], args[1]);
          }
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
  func.walk([&](Operation *insideOp) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(insideOp)) {
      opList.push_back(genericOp);
    }
  });
  for (auto op : opList) {
    rewriter.setInsertionPoint(op);
    SmallVector<int> ReductionIndex;
    Block &body = op.getRegion().front();
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body,
                                                  false)) {
      continue;
    }
    int addType = isa<FloatOp>(body.front());
    // 4. Obtain input and output tensors
    Value input = op.getInputs()[0];
    Value output = op.getOutputs()[0];

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    // 5. Sort the reduction dimensions to ensure they are processed in order.
    llvm::sort(ReductionIndex);

    // 6. Gradual reduction: process one reduction dimension at a time.
    Value currentResult = input;
    auto currentType = inputType;

    SmallVector<int> beenReductionDim;
    rewriter.setInsertionPointAfter(op);
    splitMultiReduction<IntOp, FloatOp>(
        rewriter, op, currentType, currentResult, ReductionIndex,
        beenReductionDim, output, body, addType);
    op->getResult(0).replaceAllUsesWith(currentResult);
    op->erase();
  }
  rewriter.restoreInsertionPoint(originalInsertPoint);
}

Value padReductionInput(PatternRewriter &rewriter, Location loc, Value input,
                        int64_t reductionDim, bool addType,
                        Operation *reductionOp) {
  RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
  int64_t dim0Value = inputType.getShape()[reductionDim];

  // Create a new tensor shape by adding 1 to the reduction dimension.
  SmallVector<int64_t> newShape(inputType.getShape().begin(),
                                inputType.getShape().end());
  newShape[reductionDim] = dim0Value + 1;

  // Create an empty tensor with specified values.
  auto newTensorType =
      RankedTensorType::get(newShape, inputType.getElementType());
  Value paddedInput = rewriter.create<tensor::EmptyOp>(
      loc, newShape, inputType.getElementType());

  // Create zero values for filling.
  TypedAttr zeroAttr =
      createFillAttr(*reductionOp, inputType, rewriter, addType);
  Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

  // Use linalg.generic to fill the entire tensor with zeros.
  auto identityMap = rewriter.getMultiDimIdentityMap(newShape.size());
  paddedInput = rewriter
                    .create<linalg::GenericOp>(
                        loc, newTensorType,
                        /*inputs=*/ValueRange{},
                        /*outputs=*/ValueRange{paddedInput},
                        /*indexingMaps=*/ArrayRef<AffineMap>{identityMap},
                        /*iteratorTypes=*/
                        SmallVector<utils::IteratorType>(
                            newShape.size(), utils::IteratorType::parallel),
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          b.create<linalg::YieldOp>(loc, zero);
                        })
                    .getResult(0);

  // reference to tensor.extract_slice
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    rewriter.getIndexAttr(1));

  for (int64_t dimSize : inputType.getShape()) {
    sizes.push_back(rewriter.getIndexAttr(dimSize));
  }

  // Create subview from the original input
  Value inputSubview = rewriter.create<tensor::ExtractSliceOp>(
      loc, input, offsets, sizes, strides);
  // Prepare the parameters for insert_slice - inserting at the beginning of
  // paddedInput
  Value inserted = rewriter.create<tensor::InsertSliceOp>(
      loc, inputSubview, paddedInput, offsets, sizes, strides);

  return inserted;
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
  Value initTensor = nullptr;
  int count = 0;
  auto accType = dyn_cast<RankedTensorType>(acc.getType());
  bool needInsert = false;
  if (accType.getShape().size() == 0) {
    Value empty = rewriter.create<tensor::EmptyOp>(loc, SmallVector<int64_t>{},
                                                   tmpType.getElementType());
    TypedAttr zeroAttr =
        createFillAttr(*reductionOp, tmpType, rewriter, addType);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    initTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
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
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto identityMap = rewriter.getMultiDimIdentityMap(newShape.size());
    initTensor =
        rewriter
            .create<linalg::GenericOp>(
                loc,
                RankedTensorType::get(newShape, inputType.getElementType()),
                /*inputs=*/ValueRange{},
                /*outputs=*/ValueRange{empty},
                /*indexingMaps=*/ArrayRef<AffineMap>{identityMap},
                /*iteratorTypes=*/
                SmallVector<utils::IteratorType>(newShape.size(),
                                                 utils::IteratorType::parallel),
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, zero);
                })
            .getResult(0);
  }
  // Extract current block
  auto current_chunk = rewriter.create<tensor::ExtractSliceOp>(
      loc, input, offsets, sizes, strides);
  // VADD
  SmallVector<AffineMap> indexingMaps = {
      rewriter.getMultiDimIdentityMap(inputType.getShape().size()),
      rewriter.getMultiDimIdentityMap(inputType.getShape().size()),
      rewriter.getMultiDimIdentityMap(inputType.getShape().size())};

  auto added = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{corresponding_chunk.getType()},
      ValueRange{current_chunk.getResult(), corresponding_chunk},
      ValueRange{current_chunk.getResult()}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value sum;
        if (addType == 0) {
          sum = nestedBuilder.create<IntOp>(nestedLoc, blockArgs[0],
                                            blockArgs[1]);
        } else {
          sum = nestedBuilder.create<FloatOp>(nestedLoc, blockArgs[0],
                                              blockArgs[1]);
        }
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });
  if (needMerge) {
    if (needInsert) {
      added = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{initTensor.getType()}, ValueRange{added.getResult(0)},
          ValueRange{initTensor}, indexingMaps_reduce, iteratorTypes_reduce,
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange blockArgs) {
            Value sum;
            if (addType == 0) {
              sum = nestedBuilder.create<IntOp>(nestedLoc, blockArgs[0],
                                                blockArgs[1]);
            } else {
              sum = nestedBuilder.create<FloatOp>(nestedLoc, blockArgs[0],
                                                  blockArgs[1]);
            }
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
          });
      offsets.erase(offsets.begin() + count);
      sizes.erase(sizes.begin() + count);
      strides.erase(strides.begin() + count);
      auto insert = rewriter.create<tensor::InsertSliceOp>(
          loc, added.getResult(0), acc, offsets, sizes, strides);
      return insert->getResults()[0];
    } else {
      added = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{acc.getType()}, ValueRange{added.getResult(0)},
          ValueRange{acc}, indexingMaps_reduce, iteratorTypes_reduce,
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange blockArgs) {
            Value sum;
            if (addType == 0) {
              sum = nestedBuilder.create<IntOp>(nestedLoc, blockArgs[0],
                                                blockArgs[1]);
            } else {
              sum = nestedBuilder.create<FloatOp>(nestedLoc, blockArgs[0],
                                                  blockArgs[1]);
            }
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
          });
    }
  }
  return added.getResult(0);
}

template <typename IntOp, typename FloatOp>
void convertLinalgToTreeReduce(func::FuncOp func, PatternRewriter &rewriter) {
  auto originalInsertPoint = rewriter.saveInsertionPoint();

  SmallVector<linalg::GenericOp> opList;
  func.walk([&](Operation *insideOp) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(insideOp)) {
      opList.push_back(genericOp);
    }
  });
  for (auto op : opList) {
    rewriter.setInsertionPoint(op);
    SmallVector<int> ReductionIndex;
    Block &body = op.getRegion().front();
    if (!isValidTreeReducePattern<IntOp, FloatOp>(op, ReductionIndex, body,
                                                  true)) {
      continue;
    }
    int addType = isa<FloatOp>(body.front());
    // 4. Obtain input and output tensors
    Value input = op.getInputs()[0];
    Value output = op.getOutputs()[0];

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t vectorSize = inputType.getElementType().getIntOrFloatBitWidth() == 64 ? 64 :
        64 * 32 / static_cast<int64_t>(inputType.getElementType().getIntOrFloatBitWidth());
    int64_t dim0Value = inputType.getShape()[ReductionIndex[0]];
    if (dim0Value <= vectorSize) {
      continue;
    }
    if (inputType.getRank() - 1 == ReductionIndex[0]) {
      auto insertPoint = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfterValue(input);
      int remaining_value =
          static_cast<int>(2 * 32 / (inputType.getElementType().getIntOrFloatBitWidth() / 8));
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
        Value zero = rewriter.create<arith::ConstantOp>(op.getLoc(), zeroAttr);
        auto identityMap = rewriter.getMultiDimIdentityMap(newShape.size());
        Value initTensor =
            rewriter
                .create<linalg::GenericOp>(
                    op.getLoc(), resultType,
                    /*inputs=*/ValueRange{},
                    /*outputs=*/ValueRange{empty},
                    /*indexingMaps=*/ArrayRef<AffineMap>{identityMap},
                    /*iteratorTypes=*/
                    SmallVector<utils::IteratorType>(
                        newShape.size(), utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                      b.create<linalg::YieldOp>(loc, zero);
                    })
                .getResult(0);
        SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(inputType.getRank(),
                                          rewriter.getIndexAttr(1));
        for (int64_t dimSize : inputType.getShape()) {
          sizes.push_back(rewriter.getIndexAttr(dimSize));
        }
        auto paddedInput = rewriter.create<tensor::InsertSliceOp>(
            op.getLoc(), input, initTensor, offsets, sizes, strides);
        rewriter.replaceUsesWithIf(input, paddedInput,
                                   [&](OpOperand &opOperand) {
                                     Operation *userOp = opOperand.getOwner();
                                     return userOp != paddedInput;
                                   });
        input = paddedInput.getResult();
        inputType = resultType;
        dim0Value = inputType.getShape()[ReductionIndex[0]];
      }
      rewriter.restoreInsertionPoint(insertPoint);
    }
    Location loc = op.getLoc();

    // Check whether the reduction dimension is odd; if it is, perform padding.
    bool needPadding = (dim0Value % 2 != 0);
    if (needPadding) {
      // Determine whether it is an integer type
      // Obtains the reduction operator.
      Operation *reductionOp = &body.front();

      // Calling the encapsulated function
      input = padReductionInput(rewriter, loc, input, ReductionIndex[0],
                                addType, reductionOp);

      // Update inputType and dim0Value
      inputType = mlir::cast<RankedTensorType>(input.getType());
      dim0Value = inputType.getShape()[ReductionIndex[0]];
    }
    // Calculate static cycle parameters
    int64_t dimXValue = inputType.getShape().back();
    int64_t half_point_value = inputType.getRank() - 1 == ReductionIndex[0]
                                   ? (dimXValue + 1) / 2
                                   : dimXValue; // Ceil(dim0/2)

    Value acc = output;
    // 7. Static Loop Unrolling
    SmallVector<int64_t> addShape;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> offsets2;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> sizes2;
    SmallVector<OpFoldResult> strides;
    SmallVector<utils::IteratorType> iteratorTypes;
    iteratorTypes.reserve(inputType.getShape().size());
    SmallVector<utils::IteratorType> iteratorTypes_reduce;
    iteratorTypes_reduce.reserve(inputType.getShape().size());
    for (int64_t dim = 0; dim < inputType.getRank(); dim++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
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
      iteratorTypes.push_back(utils::IteratorType::parallel);
    }

    for (int64_t i = 0; i <= 1; i++) {
      // Calculate the starting position and size of the current block
      int64_t current_start_value = (i == 0 ? 0 : half_point_value < vectorSize ?
                                     half_point_value : half_point_value / vectorSize * vectorSize);
      int64_t remaining_value = half_point_value - current_start_value;
      if (!remaining_value) {
        break;
      }
      int64_t chunk_size_value = std::min(remaining_value,
                                    std::max(vectorSize, half_point_value / vectorSize * vectorSize));
      // Calculate the starting position of the corresponding second half.
      int64_t corresponding_start_value = current_start_value;
      if (inputType.getRank() - 1 == ReductionIndex[0]) {
        corresponding_start_value += half_point_value;
      }
      // Calculate the size of the corresponding block in the latter part.
      int64_t corresponding_remaining_value =
          dimXValue - corresponding_start_value;
      int64_t corresponding_size_value =
          std::min(chunk_size_value, corresponding_remaining_value);
      corresponding_size_value =
          std::max(corresponding_size_value, (int64_t)(0));
      offsets.back() = rewriter.getIndexAttr(current_start_value);
      offsets2.back() = rewriter.getIndexAttr(corresponding_start_value);
      sizes.back() = rewriter.getIndexAttr(corresponding_size_value);
      SmallVector<AffineMap> indexingMaps_reduce = {
          rewriter.getMultiDimIdentityMap(inputType.getShape().size()),
          AffineMap::getMultiDimIdentityMap(inputType.getShape().size(),
                                            rewriter.getContext())
              .dropResult(ReductionIndex[0])};
      // Extract the corresponding latter half block
      Value corresponding_chunk = rewriter.create<tensor::ExtractSliceOp>(
          loc, input, offsets2, sizes, strides);
      bool needMerge = (mlir::cast<RankedTensorType>(corresponding_chunk.getType())
                                                    .getShape()[ReductionIndex[0]]
                                                    !=0);
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
      // Skip this for now
      return failure();
    }
    // Currently, only for the Triton reduce sum op, reduce max op and reduce min op.
    convertStridedReduceToMultiReduce<arith::AddIOp, arith::AddFOp>(func,
                                                                    rewriter);
    convertStridedReduceToMultiReduce<arith::MaxSIOp, arith::MaxNumFOp>(func,
                                                                    rewriter);
    convertStridedReduceToMultiReduce<arith::MinSIOp, arith::MinNumFOp>(func,
                                                                    rewriter);
    convertLinalgToTreeReduce<arith::AddIOp, arith::AddFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MaxSIOp, arith::MaxNumFOp>(func, rewriter);
    convertLinalgToTreeReduce<arith::MinSIOp, arith::MinNumFOp>(func, rewriter);
    return success();
  }
};
} // namespace
void hfusion::populateTreeReducePattern(RewritePatternSet &patterns) {
  patterns.add<TreeReducePattern>(patterns.getContext());
}