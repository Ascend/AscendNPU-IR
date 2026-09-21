//===- OptimizeScalarTransfers.cpp - Optimize scalar transfers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Analysis/IntermediateValuesAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/ScalarOptimizationsUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "hfusion-optimize-scalar-transfers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_OPTIMIZESCALARTRANSFERS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::detail;

namespace {
using TopSort = SetVector<Operation *>;

class IntermediateValuesDataFlowListener : public RewriterBase::Listener {
public:
  explicit IntermediateValuesDataFlowListener(
      IntermediateValuesAnalyzer &analyzer)
      : analyzer(analyzer) {}

protected:
  void notifyOperationErased(Operation *op) override {
    analyzer.invalidate(op);
  }

  void notifyBlockErased(Block *block) override { analyzer.invalidate(block); }

private:
  IntermediateValuesAnalyzer &analyzer;
};

static bool isNonControlFlowTransfer(Operation *op,
                                     IntermediateValuesAnalyzer &transfers) {
  if (auto extract = dyn_cast<tensor::ExtractOp>(op))
    return isVectorToScalarTransfer(extract) &&
           transfers.isIntermediateTransfer(extract.getResult());

  if (auto insert = dyn_cast<tensor::InsertOp>(op))
    return isScalarInsert(insert) &&
           transfers.isIntermediateTransfer(insert.getScalar());

  if (auto fill = dyn_cast<linalg::FillOp>(op))
    return isStaticRankOneFill(op) &&
           transfers.isIntermediateTransfer(fill.value());

  if (isa<scf::ForOp, scf::IfOp>(op))
    return false;

  return op->getNumRegions() == 0 && op->getNumResults() == 1 &&
         transfers.isIntermediateTransfer(op->getResult(0));
}

static Value createScalarSingletonTensor(Value scalar, OpBuilder &builder) {
  Type type = scalar.getType();
  assert(type.isIntOrFloat());
  auto loc = scalar.getLoc();

  RankedTensorType tensorType = RankedTensorType::get({1}, type);

  auto empty =
      builder.create<tensor::EmptyOp>(loc, tensorType.getShape(), type);

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

  auto insertOp =
      builder.create<tensor::InsertOp>(loc, scalar, empty, ValueRange(zero));

  return insertOp.getResult();
}

static Value insertScalarValue(Value scalar, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(scalar);
  return createScalarSingletonTensor(scalar, builder);
}

static Value extractVectorValue(Value vector, OpBuilder &builder) {
  assert(isSingletonTensorType(vector.getType()));
  auto loc = vector.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(vector);

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

  auto extractOp =
      builder.create<tensor::ExtractOp>(loc, vector, ValueRange(zero));

  return extractOp.getResult();
}

static Value fillSingletonTensorWithScalar(Value scalar, OpBuilder &builder) {
  auto elementType = scalar.getType();
  assert(elementType.isIntOrFloat());
  auto loc = scalar.getLoc();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(scalar);

  RankedTensorType tensorType = RankedTensorType::get({1}, elementType);

  auto empty =
      builder.create<tensor::EmptyOp>(loc, tensorType.getShape(), elementType);

  auto fillOp = builder.create<linalg::FillOp>(loc, scalar, empty.getResult());

  return fillOp.getResult(0);
}

/// Tensor produced by a `linalg.reduce` that reduces all of its input
/// dimensions, the pass rewrites the reduce so that it produces
/// `tensor<1xElemT>` directly instead of inserting a
/// `tensor.expand_shape` on the reduce result:
///
///   %r = linalg.reduce ins(%a : tensor<NxT>) outs(%i : tensor<T>)
///        dimensions = [0]
///   %e = tensor.extract %r[] : tensor<T>
///
/// becomes
///
///   %a' = tensor.expand_shape %a [[0, 1]] : tensor<NxT> into tensor<1xNxT>
///   %i' = tensor.expand_shape %i [] : tensor<T> into tensor<1xT>
///   %r' = linalg.reduce ins(%a' : tensor<1xNxT>) outs(%i' : tensor<1xT>)
///        dimensions = [1]
///   %e' = tensor.extract %r'[%c0] : tensor<1xT>  (then lifted to NoOp)
///
/// FIXME: After tree reduce for 1D reduction will be supported
///        this normalization step probably can be ommited after
///        performance regression testing
static bool normalizeConsumedReduce(linalg::ReduceOp reduceOp,
                                    IntermediateValuesAnalyzer &transfers,
                                    IRRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);

  // Since reduction with index supported only for 1D tensors skip normalization
  if ((reduceOp.getResults().size() != 1) || !reduceOp->hasOneUse() ||
      reduceOp.hasIndexSemantics())
    return false;

  auto extractOp = dyn_cast<tensor::ExtractOp>(*reduceOp->getUsers().begin());
  if (!extractOp || !transfers.isIntermediateTransfer(extractOp.getResult()))
    return false;

  Value tensor = extractOp.getTensor();
  if (!isZeroRankTensorType(tensor.getType()))
    return false;

  auto inputType =
      dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
  auto initType = dyn_cast<RankedTensorType>(reduceOp.getInits()[0].getType());
  if (!inputType || !initType || inputType.getEncoding() ||
      initType.getEncoding())
    return false;

  unsigned rank = inputType.getRank();

  SmallVector<ReassociationIndices> reassociation;
  for (unsigned dim = 0; dim < rank; ++dim) {
    if (dim == 0)
      reassociation.push_back(ReassociationIndices{0, 1});
    else
      reassociation.push_back(ReassociationIndices{dim + 1});
  }

  auto loc = reduceOp.getLoc();
  rewriter.setInsertionPoint(reduceOp);

  // Expand the input and the init with the leading singleton dimension.
  SmallVector<int64_t> expandedShape{1};
  expandedShape.append(inputType.getShape().begin(),
                       inputType.getShape().end());
  Value expandedInput =
      rewriter
          .create<tensor::ExpandShapeOp>(
              loc,
              RankedTensorType::get(expandedShape, inputType.getElementType()),
              reduceOp.getInputs()[0], reassociation)
          .getResult();

  // The rank-0 init is expanded with an empty reassociation.
  SmallVector<ReassociationIndices> emptyReassociation;
  Value expandedInit =
      rewriter
          .create<tensor::ExpandShapeOp>(
              loc, RankedTensorType::get({1}, initType.getElementType()),
              reduceOp.getInits()[0], emptyReassociation)
          .getResult();

  rewriter.modifyOpInPlace(reduceOp, [&] {
    reduceOp.getInputsMutable().assign(ValueRange{expandedInput});
    reduceOp.getInitsMutable().assign(ValueRange{expandedInit});
    SmallVector<int64_t> newDims(reduceOp.getDimensions().begin(),
                                 reduceOp.getDimensions().end());
    for (int64_t &dim : newDims)
      dim += 1;
    reduceOp.setDimensions(newDims);
    reduceOp.getResult(0).setType(
        RankedTensorType::get({1}, initType.getElementType()));
  });

  // The extract now reads the singleton element of the reduced tensor.
  rewriter.setInsertionPoint(extractOp);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  extractOp.getIndicesMutable().assign(ValueRange{zero});

  LDBG("Lifted the singleton dimension into the reduce: "
       << *reduceOp.getOperation());
  return true;
}

/// Converts `tensor.extract` operation with intermediate result.
/// Conversion depends on the operand type:
///
/// (a) Iff extracted operand has `tensor<1xElemT>` type,
///     then operation is converted to NoOp
/// (b) Iff extracted operand has `tensor<ElemT>` type,
///     then extract operation converted to
///     `expandShape: tensor<ElemT> -> tensor<1xElemT>`
static void liftExtractOp(tensor::ExtractOp extractOp, IRRewriter &rewriter,
                          IRMapping &map,
                          DenseSet<Operation *> &deferredRemove) {
  LDBG("Start lifting extractOp: " << *extractOp.getOperation());

  rewriter.setInsertionPoint(extractOp);
  auto tensor = extractOp.getTensor();
  auto tensorType = tensor.getType();
  assert(isa<RankedTensorType>(tensorType) &&
         "scalar-transfer extraction requires a ranked tensor");
  auto loc = tensor.getLoc();

  if (isSingletonTensorType(extractOp.getTensor().getType())) {
    deferredRemove.insert(extractOp);
    map.map(extractOp.getResult(), extractOp.getTensor());
  } else {
    auto resultType =
        RankedTensorType::get({1}, tensor.getType().getElementType());

    // Empty reassociation is needed to expand rank 0 -> rank 1
    SmallVector<ReassociationIndices> reassociation;
    auto expandShape = rewriter.create<tensor::ExpandShapeOp>(
        loc, resultType, tensor, reassociation);

    LDBG("The replacement for operation is: " << *expandShape.getOperation());
    map.map(extractOp.getResult(), expandShape.getResult());
    return;
  }
}

/// Converts `tensor.insert` with intermediate scalar operand to NoOp.
/// Uses of converted operation replaced by lifted scalar value.
static void liftInsertOp(tensor::InsertOp insertOp, IRRewriter &rewriter,
                         IRMapping &map,
                         DenseSet<Operation *> &deferredRemove) {
  LDBG("Converting insert to NoOp: " << *insertOp.getOperation());

  Value scalar = insertOp.getScalar();
  assert(scalar.getType().isIntOrFloat() &&
         "scalar-transfer insertion requires a scalar value");
  assert(map.contains(scalar) &&
         "broken transformation order, the insert argument wasn't mapped");
  Value scalarReplacement = map.lookup(scalar);

  if (Operation *def = scalar.getDefiningOp()) {
    LDBG("The operation is scheduled for removal: " << *def);
    deferredRemove.insert(def);
  }

  rewriter.replaceOp(insertOp, scalarReplacement);
  return;
}

/// Converts `fillOp` with intermediate scalar operand to
/// `extract + fill` pattern that can be optimized in the future.
/// So during the lifting process IR:
///
///   %scalar = scalar_op : ElemT // intermediate value
///   %fill = fillOp (%scalar) (...) : tensor<NxElemT>
///
/// becomes:
///
///   %lifted = vector_op : tensor<1xElemT>
///   %extracted = tensor.extract %lifted[%c0] : tensor<1xElemT>
///   %fillOp = fillOp (%extracted) (...) : tensor<NxElemT>
static void liftFillOp(linalg::FillOp fillOp, IRRewriter &rewriter,
                       IRMapping &map, DenseSet<Operation *> &deferredRemove) {
  LDBG("Started lifting : " << *fillOp);
  rewriter.setInsertionPointAfter(fillOp);
  OpOperand &broadcastScalarOperand = fillOp->getOpOperand(0);
  Value broadcastedScalarValue = broadcastScalarOperand.get();
  assert(map.contains(broadcastedScalarValue) &&
         "broken transformation order, the fill argument wasn't mapped");

  Value vectorReplacement = map.lookup(broadcastedScalarValue);

  Value output = fillOp.output();
  assert(isa<RankedTensorType>(output.getType()) &&
         "fill over memref should not be marked as consumed");

  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  assert(outputType.getRank() == 1 &&
         "the init of linalg.fill should be first rank tensor");
  assert(!ShapedType::isDynamic(outputType.getDimSize(0)) &&
         "dynamic rank-one fills are not scalar-transfer candidates");

  auto extracted = extractVectorValue(vectorReplacement, rewriter);
  broadcastScalarOperand.assign(extracted);

  if (Operation *def = broadcastedScalarValue.getDefiningOp()) {
    deferredRemove.insert(def);
  }
}

/// Generic lift of intermediate scalar operation
/// with only one result and without regions.
/// Operands and result types lifted
/// from ElemT to tensor<1xElemT>.
/// For operations with non-intermediate arguments
/// fillOp adaptor created.
///
/// So after whole rewrite following IR:
///
/// %scalar = op(%1, %2) : ElemT
///
/// becomes:
///
/// %adaptor = fillOp (%2) (...) : tensor<1xElemT>
/// %lifted = op(%lifted_1, adaptor) : tensor<1xElemT>
static Operation *liftOp(Operation *op, IRRewriter &rewriter, IRMapping &map) {
  LDBG("Started lifting of: " << *op);
  assert(op->getNumResults() == 1 &&
         "scalar-transfer lifting requires a single result");
  Type type = op->getResult(0).getType();
  assert(type.isIntOrFloat());
  rewriter.setInsertionPointAfter(op);

  SmallVector<Value> operands;

  for (Value operand : op->getOperands()) {
    if (!map.contains(operand)) {
      // FillOp doesn't block VFFusion,
      // so here we generate linalg.fillOp (operand) : tensor<1xf32>
      // to transfer scalar value to the vector context
      Value fillOp = fillSingletonTensorWithScalar(operand, rewriter);
      LDBG("Adaptor created: " << *fillOp.getDefiningOp());
      map.map(operand, fillOp);
    }
  }

  assert(llvm::all_of(op->getOperands(),
                      [&](Value operand) { return map.contains(operand); }));

  Operation *clonedOp = rewriter.clone(*op, map);
  RankedTensorType tensorType = RankedTensorType::get({1}, type);
  SmallVector<Type, 1> newResultTypes{tensorType};

  rewriter.modifyOpInPlace(clonedOp, [&] {
    for (auto [result, newType] :
         llvm::zip_equal(clonedOp->getResults(), newResultTypes)) {
      result.setType(newType);
    }
  });

  for (auto [originalVal, newVal] :
       llvm::zip_equal(clonedOp->getResults(), op->getResults())) {
    map.map(originalVal, newVal);
  }

  LDBG("replacement created: " << *clonedOp);
  return clonedOp;
}

static bool isAIV(func::FuncOp funcOp) {
  if (auto mixMode = funcOp->getAttrOfType<StringAttr>("mix_mode")) {
    return mixMode.getValue() == "aiv";
  }
  return false;
}

/// Lifts operands and their corresponding result types
/// from `ElemT` to tensor<1xElemT>
///
/// So following IR after conversion:
///
///   %1 = scf.for ... iter_args(%arg1 = %cst_0): {
///     %s = scalar(%arg1) : f32
///     %inserted = tensor.insert %s into %empty[%c0] : tensor<1xf32>
///     %v = vector_ops(%inserted)
///     %extracted = tensor.extract %v[%c0] : tensor<1xf32>
///     %result = scalar(%extracted) : f32
///     scf.yield %result : f32
///   }
///
/// becomes:
///
///   %1 = scf.for ... iter_args(%arg1 = %inserted): {
///     %extracted_arg = tensor.extract %arg1[%c0] : tensor<1xf32>
///     %s = scalar(%extracted_arg) : f32
///     %inserted = tensor.insert %s into %empty[%c0] : tensor<1xf32>
///     %v = vector_ops(%inserted)
///     %extracted = tensor.extract %v[%c0] : tensor<1xf32>
///     %result = scalar(%extracted) : f32
///     %inserted_res = tensor.insert %result into %empty[%c0] : tensor<1xf32>
///     scf.yield %inserted_res : tensor<1xf32>
///   }
///   %extracted_1 = tensor.extract %1[%c0] : tensor<1xf32>
///
/// Expects that @param operands is sorted in increasing order
static SmallVector<Value>
liftForOperands(RewriterBase &rewriter, scf::ForOp forOp,
                ArrayRef<OpOperand *> operands, ArrayRef<Value> replacements,
                IntermediateValuesAnalyzer &transfers) {
  assert(!operands.empty());
  assert(operands.size() == replacements.size());
  assert(llvm::all_of(operands, [&](OpOperand *operand) {
    return operand->getOwner() == forOp;
  }));
  assert(llvm::all_of(replacements, [&](Value val) {
    return isSingletonTensorType(val.getType());
  }));
  assert(llvm::all_of(operands, [&](OpOperand *operand) {
    return operand->get().getType().isIntOrFloat();
  }));

  // 1. Create new iter operands list
  SmallVector<Value> newIterOperands;
  unsigned replacementIdx = 0;
  for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
    if (replacementIdx < operands.size() &&
        opOperand.getOperandNumber() ==
            operands[replacementIdx]->getOperandNumber()) {
      newIterOperands.push_back(replacements[replacementIdx]);
      replacementIdx++;
      continue;
    }
    newIterOperands.push_back(opOperand.get());
  }
  assert(
      replacementIdx == operands.size() &&
      "broken operands/init-arg mapping, a selected channel was not applied");

  // 2. Create new forOp with replaced operands
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(forOp);
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands, /*bodyBuilder=*/nullptr,
      forOp.getUnsignedCmp());
  newForOp->setDiscardableAttrs(
      forOp->getDiscardableAttrDictionary().getValue());

  Block &newBlock = newForOp.getRegion().front();
  SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                             newBlock.getArguments().end());

  // 3. Inject an incoming cast op at the beginning of the block for the bbArg
  // corresponding to the `replacement` value.
  rewriter.setInsertionPointToStart(&newBlock);

  for (auto *operand : operands) {
    BlockArgument newRegionIterArg = newForOp.getTiedLoopRegionIterArg(
        &newForOp->getOpOperand(operand->getOperandNumber()));
    Value extract = extractVectorValue(newRegionIterArg, rewriter);
    transfers.registerNewValue(extract);
    newBlockTransferArgs[newRegionIterArg.getArgNumber()] = extract;
  }

  // 4. Steal the old block ops, mapping to the newBlockTransferArgs.
  Block &oldBlock = forOp.getRegion().front();
  rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

  // 5. Inject an outgoing cast op at the end of the block and yield it instead.
  auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
  SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
  rewriter.setInsertionPoint(clonedYieldOp);

  for (auto *operand : operands) {
    BlockArgument newRegionIterArg = newForOp.getTiedLoopRegionIterArg(
        &newForOp->getOpOperand(operand->getOperandNumber()));
    unsigned yieldIdx =
        newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
    Value insert =
        insertScalarValue(clonedYieldOp.getOperand(yieldIdx), rewriter);
    newYieldOperands[yieldIdx] = insert;
  }

  rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(clonedYieldOp);

  // 6. Inject an outgoing cast op after the forOp.
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> newResults = newForOp.getResults();
  for (auto *operand : operands) {
    BlockArgument newRegionIterArg = newForOp.getTiedLoopRegionIterArg(
        &newForOp->getOpOperand(operand->getOperandNumber()));
    unsigned yieldIdx =
        newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
    newResults[yieldIdx] = extractVectorValue(newResults[yieldIdx], rewriter);
  }

  return newResults;
}

/// Lifts `scf.if` results from `ElemT` to `tensor<1xElemT>`.
///
/// So, after lifting following IR:
///
///   %result = scf.if %condition -> (f32) {
///     scf.yield %cst: f32
///   } else {
///     %s = vector(...)
///     %extracted = tensor.extract %s[%c0] : tensor<1xf32>
///     scf.yield %extracted: f32
///   }
///   %inserted = tensor.insert %result into %empty[%c0] : tensor<1xf32>
///
/// becomes:
///
///   %result = scf.if %condition -> (tensor<1xf32>) {
///     %inserted_cst = tensor.insert %cst into %empty[%c0] : tensor<1xf32>
///     scf.yield %inserted_cst: tensor<1xf32>
///   } else {
///     %s = vector(...)
///     %extracted = tensor.extract %s[%c0] : tensor<1xf32>
///     %inserted = tensor.insert %s into %empty[%c0] : tensor<1xf32>
///     scf.yield %inserted: f32
///   }
///   %extracted_res = tensor.extract %result[%c0] : tensor<1xf32>
///   %inserted_res = tensor.insert %result into %empty[%c0] : tensor<1xf32>
///
/// Expects that @param intermediateResults is sorted in increasing order
static SmallVector<Value> liftIf(RewriterBase &rewriter, scf::IfOp ifOp,
                                 ArrayRef<unsigned> intermediateResults,
                                 IntermediateValuesAnalyzer &transfers) {
  assert(!intermediateResults.empty());
  assert(llvm::is_sorted(intermediateResults));

  SmallVector<Type> newResultTypes(ifOp.getResultTypes());
  for (unsigned resultIndex : intermediateResults) {
    Type resultType = ifOp.getResult(resultIndex).getType();
    assert(resultType.isIntOrFloat());
    newResultTypes[resultIndex] = RankedTensorType::get({1}, resultType);
  }

  rewriter.setInsertionPointAfter(ifOp);
  auto newIf = rewriter.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, ifOp.getCondition(),
      /*addThenBlock=*/false, /*addElseBlock=*/false);
  newIf->setDiscardableAttrs(ifOp->getDiscardableAttrDictionary().getValue());

  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());

  auto rewriteYield = [&](Region &region) {
    auto yieldOp = cast<scf::YieldOp>(region.front().getTerminator());
    SmallVector<Value> newYieldOperands(yieldOp.getOperands());
    rewriter.setInsertionPoint(yieldOp);

    for (unsigned resultIndex : intermediateResults) {
      Value scalar = yieldOp.getOperand(resultIndex);
      assert(scalar.getType().isIntOrFloat());
      Value tensor = createScalarSingletonTensor(scalar, rewriter);
      newYieldOperands[resultIndex] = tensor;
    }

    for (auto [index, replacement] : llvm::enumerate(newYieldOperands))
      yieldOp->getOpOperand(index).assign(replacement);
  };

  rewriteYield(newIf.getThenRegion());
  rewriteYield(newIf.getElseRegion());

  SmallVector<Value> newResults(newIf.getResults());
  rewriter.setInsertionPointAfter(newIf);
  for (unsigned resultIndex : intermediateResults) {
    newResults[resultIndex] =
        extractVectorValue(newResults[resultIndex], rewriter);
  }

  return newResults;
}

/// Selects intermediate results of `scf.if` operation
/// and then lifts their types from `ElemT` to `tensor<1xElemT>`.
static void normalizeIf(scf::IfOp ifOp, IntermediateValuesAnalyzer &transfers,
                        IRRewriter &rewriter) {
  SmallVector<unsigned> intermediateResults;

  for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
    if (!transfers.isIntermediateTransfer(result))
      continue;

    assert(result.getType().isIntOrFloat());
    intermediateResults.push_back(index);
  }

  if (intermediateResults.empty())
    return;

  auto newResults = liftIf(rewriter, ifOp, intermediateResults, transfers);
  rewriter.replaceOp(ifOp, newResults);

  // Refine inserted transfers
  for (unsigned idx : intermediateResults) {
    // The result was marked as before transformation, so its value is consumed
    // by vector context. So newly created extract is intermediate transfer
    transfers.registerNewValue(newResults[idx]);
  }
}

/// Selects intermediate results of `scf.for` operation
/// and then lifts their types from `ElemT` to `tensor<1xElemT>`.
static void normalizeLoop(scf::ForOp forOp,
                          IntermediateValuesAnalyzer &transfers,
                          IRRewriter &rewriter) {

  SmallVector<OpOperand *> intermediateOperands;
  SmallVector<unsigned> intermediateIdxs;
  llvm::SmallSet<unsigned, 3> intermediateResults;
  for (auto blockArg : forOp.getRegionIterArgs()) {
    unsigned idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
    bool isConsumed = transfers.isIntermediateTransfer(forOp.getResult(idx));
    if (transfers.isIntermediateTransfer(blockArg) || isConsumed) {
      intermediateOperands.push_back(forOp.getTiedLoopInit(blockArg));
      intermediateIdxs.push_back(idx);
      if (isConsumed)
        intermediateResults.insert(idx);
    }
  }

  if (intermediateIdxs.empty())
    return;

  SmallVector<Value> newInits;
  for (auto *operand : intermediateOperands) {
    auto insert = insertScalarValue(operand->get(), rewriter);
    newInits.push_back(insert);
  }

  auto newResults = liftForOperands(rewriter, forOp, intermediateOperands,
                                    newInits, transfers);
  rewriter.replaceOp(forOp, newResults);

  for (unsigned idx : intermediateIdxs) {
    if (intermediateResults.contains(idx)) {
      transfers.registerNewValue(newResults[idx]);
    }
  }
}

static void
eliminateProducerConsumerTransfers(TopSort topSort, SetVector<Operation *> ops,
                                   IntermediateValuesAnalyzer &transfers,
                                   IRRewriter &rewriter) {
  IRMapping replacementMap;
  DenseSet<Operation *> deferredRemove;

  if (ops.empty())
    return;

  for (Operation *op : ops) {
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      liftExtractOp(extractOp, rewriter, replacementMap, deferredRemove);
    } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
      liftInsertOp(insertOp, rewriter, replacementMap, deferredRemove);
    } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      liftFillOp(fillOp, rewriter, replacementMap, deferredRemove);
    } else {
      liftOp(op, rewriter, replacementMap);
    }
  }

  for (Operation *currentOp : llvm::reverse(topSort)) {
    if (!(deferredRemove.contains(currentOp) && ops.contains(currentOp)))
      continue;

    LDBG("started removal of: " << *currentOp << " at "
                                << static_cast<void *>(currentOp));

    if (!currentOp->getUses().empty()) {

      for (OpOperand &use : llvm::make_early_inc_range(currentOp->getUses())) {
        LDBG("rewriting use of "
             << *currentOp << " in " << *use.getOwner()
             << ", mapped=" << replacementMap.contains(use.get()));
        assert(!transfers.isIntermediateTransfer(use.getOwner()) &&
               "broken transformation order, trying to remove operation with "
               "intermediate use");
        auto vectorReplacement = replacementMap.lookup(use.get());
        Value replacement = extractVectorValue(vectorReplacement, rewriter);
        use.assign(replacement);
        LDBG("after assign: uses=" << currentOp->getNumResults() << ", "
                                   << currentOp->use_empty());
      }
    }

    assert(currentOp->getUsers().empty() &&
           "tries to remove operation with uses");

    for (Value operand : currentOp->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && ops.contains(def)) {
        LDBG("Operation is scheduled for removal: " << *def);
        deferredRemove.insert(def);
      }
    }

    rewriter.eraseOp(currentOp);
  }
}

// TODO: probably it will be more profitable to use greedy rewriter
void normalize(func::FuncOp funcOp, IntermediateValuesAnalyzer &transfers,
               IRRewriter &rewriter) {
  // Preprocess scf.for operation
  SmallVector<scf::ForOp> loops;
  SmallVector<scf::IfOp> ifs;
  SmallVector<linalg::ReduceOp> reduces;
  funcOp.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      loops.push_back(forOp);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      ifs.push_back(ifOp);
    } else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
      reduces.push_back(reduceOp);
    }
  });

  for (auto loop : loops)
    normalizeLoop(loop, transfers, rewriter);

  for (auto ifOp : ifs)
    normalizeIf(ifOp, transfers, rewriter);

  for (auto reduceOp : reduces)
    normalizeConsumedReduce(reduceOp, transfers, rewriter);
}

/// The optimization of intermediate transfers consists
/// of four main steps:
///
/// 1. Analysis of intermediate values
/// 2.1 Normalization of `scf` operations
/// 2.2 Normalization of `reduce` operations
/// 3. Top-down walk to create lifted operations
/// 4. Bottom-up walk to remove old scalar operations
void optimizeScalarTransfers(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp->getContext());
  IntermediateValuesAnalyzer transfers(funcOp.getOperation());
  IntermediateValuesDataFlowListener listener(transfers);
  rewriter.setListener(&listener);
  normalize(funcOp, transfers, rewriter);

  llvm::SetVector<mlir::Operation *> operations;
  funcOp.walk([&](Operation *op) {
    if (funcOp.getOperation() != op) {
      operations.insert(op);
    }
  });

  TopSort topSort = mlir::topologicalSort(operations);

  SetVector<Operation *> order;

  for (Operation *op : topSort) {
    LDBG("visiting operation: " << *op);
    LDBG("is transfer: " << transfers.isIntermediateTransfer(op) << "\n");

    if (isNonControlFlowTransfer(op, transfers)) {
      assert(op->getNumResults() == 1 &&
             "scalar-transfer lifting requires a single result");
      order.insert(op);
    }
  }

  eliminateProducerConsumerTransfers(topSort, order, transfers, rewriter);
}

struct OptimizeScalarTransfersPass
    : public impl::OptimizeScalarTransfersBase<OptimizeScalarTransfersPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!isAIV(funcOp)) {
      LDBG("The following func is not AIV, so pass is skipped"
           << funcOp.getSymName());
      return;
    }

    auto result = funcOp.walk([&](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      if (dialect == "cf" ||
          (dialect == "scf" && !isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op))) {
        op->emitWarning("[hfusion-optimize-scalar-transfers]: unexpected "
                        "control flow at the early hfusion pipeline");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return;

    LDBG("started the transfers optimizations");
    optimizeScalarTransfers(funcOp);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createOptimizeScalarTransfersPass() {
  return std::make_unique<OptimizeScalarTransfersPass>();
}
