//===- FusibleProducerAnalyzer.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"

#include "llvm/Support/FormatVariadic.h"

#include <functional>

#define DEBUG_TYPE "fusible-producer-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::detail;

namespace {
/// Attribute for marking the reduce op in the kernel function.
inline char kReductionOpIdxFormat[] = "__reduction{0}__";

/// Attribute for marking reduce op's producers for a certain reduction axis.
inline char kReductionFusibleProducerFormat[] =
    "__reduction{0}_axis{1}_fusible_producer__";

/// Attribute for marking store op's producers for a certain reduction axis.
inline char kReturnValueFusibleProducerFormat[] =
    "__result{0}_axis{1}_fusible_producer__";

/// Set a unit attribute named \c attrName to \c op.
void trySetNamedAttr(Operation *op, NamedAttribute attr) {
  assert(op != nullptr);
  if (!op->hasAttr(attr.getName()))
    op->setAttr(attr.getName(), attr.getValue());
}

//===----------------------------------------------------------------------===//
// Utils for tracing producers
//===----------------------------------------------------------------------===//

bool isFusibleOp(Operation *op) {
  return isa_and_nonnull<linalg::LinalgOp, tensor::PadOp, tensor::ConcatOp,
                         hfusion::DeinterleaveOp, hfusion::InterleaveOp>(op);
}

SmallVector<Value> getInputOperands(Operation *op) {
  return TypeSwitch<Operation *, SmallVector<Value>>(op)
      .Case([&](linalg::LinalgOp linalgOp) { return linalgOp.getDpsInputs(); })
      .Case([&](tensor::PadOp padOp) {
        return SmallVector<Value>{padOp.getSource()};
      })
      .Case([&](tensor::ConcatOp concatOp) {
        return SmallVector<Value>{concatOp.getInputs()};
      })
      .Case([&](hfusion::DeinterleaveOp deinterleaveOp) {
        return SmallVector<Value>{deinterleaveOp.getInput()};
      })
      .Case([&](hfusion::InterleaveOp interleaveOp) {
        return SmallVector<Value>{interleaveOp.getInput()};
      })
      .Default([&](Operation *) { return SmallVector<Value>(); });
}

using FusibleProducerTestFn = std::function<bool(Operation *)>;

/// Utility function to trace back consumer op's producers that can be fused
/// to the same containing loop.
void traceBackToFusibleProducersForConsumer(
    Operation *consumer, SetVector<Operation *> &producers,
    const FusibleProducerTestFn &isFusibleProducer) {
  if (!isFusibleOp(consumer))
    return;

  auto workList = getInputOperands(consumer);
  for (Value operand : workList) {
    auto *nextOperation = operand.getDefiningOp();
    if (!nextOperation)
      continue;

    if (!isFusibleOp(nextOperation)) {
      LDBG("nextOperation " << *nextOperation << " is not fusible");
      continue;
    }

    if (producers.contains(nextOperation))
      continue;

    auto isFusible = isFusibleProducer(nextOperation);
    if (!isFusible) {
      LDBG("nextOperation " << *nextOperation
                            << " is not fusible because it failed the test.");
      continue;
    }

    LDBG("nextOperation " << *nextOperation << " is fusible");
    producers.insert(nextOperation);
    traceBackToFusibleProducersForConsumer(nextOperation, producers,
                                           isFusibleProducer);
  }
}

bool isFusibleProducerForConsumerWithReductionAxes(Operation *op,
                                                   DimensionAnalyzer *analyzer,
                                                   int64_t reduceDimInAnchor) {
  if (!isFusibleOp(op))
    return false;

  auto results = op->getResults();
  // TODO: Support multi-output operations.
  if (!llvm::hasSingleElement(results))
    return false;

  auto result = results.front();
  auto maybeShapedType = dyn_cast<ShapedType>(result.getType());
  if (!maybeShapedType)
    return false;

  // Cannot fuse another reduce op
  if (isa<linalg::ReduceOp>(op))
    return false;

  // Check if the current value shares the common reduction axis
  for (auto dimIdx = 0; dimIdx < maybeShapedType.getRank(); ++dimIdx) {
    if (analyzer->isDimensionEqualToAnchor(reduceDimInAnchor, {result, dimIdx},
                                           /*isStrict=*/false))
      return true;
  }
  return false;
}

} // namespace

namespace mlir {
namespace hfusion {
namespace detail {

//===----------------------------------------------------------------------===//
// ConsumerWithReduction
//===----------------------------------------------------------------------===//

ConsumerWithReduction::ConsumerWithReduction(Operation *consumer,
                                             ConsumerType type,
                                             NamedAttribute identifier) {
  this->type = type;
  this->identifier = identifier;
  // tag the consumer so that we can match it later in schedule
  trySetNamedAttr(consumer, *this->identifier);
}

//===----------------------------------------------------------------------===//
// FusibleProducers
//===----------------------------------------------------------------------===//

void FusibleProducers::setProducerInfo(SetVector<Operation *> producers,
                                       NamedAttribute identifier) {
  this->producers = std::move(producers);
  this->identifier = identifier;
  // tag the producers so that we can match them later in schedule
  llvm::for_each(
      this->producers,
      std::bind(trySetNamedAttr, std::placeholders::_1, *this->identifier));
}

// TODO: the analyzer should be const
FusibleProducerAnalysisResult
analyzeProducersForReductionOp(linalg::ReduceOp reduceOp,
                               const ReduceInfo &reduceOpInfo,
                               DimensionAnalyzer *analyzer) {
  MLIRContext *ctx = reduceOp->getContext();
  FusibleProducerAnalysisResult result;
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//";
  LDBG(logLineComment);
  LDBG("Analyzing reduction producers for consumer #" << reduceOpInfo.idx);
  LDBG(*reduceOp);
  LDBG(logLineComment);
#endif
  // Build consumer information
  result.consumerInfo = ConsumerWithReduction(
      /*consumer=*/reduceOp,
      /*type=*/ConsumerType::kReduction,
      /*identifier=*/
      NamedAttribute(
          StringAttr::get(
              ctx,
              llvm::formatv(kReductionOpIdxFormat, reduceOpInfo.idx).str()),
          UnitAttr::get(ctx)));

  // Build producer information for each reduce dim
  assert(!reduceOpInfo.inputsInterchange.empty());
  SmallVector<int64_t> interchange = reduceOpInfo.inputsInterchange.front();
  for (int64_t reduceDim : reduceOpInfo.reductionDims) {
    int64_t reduceDimInAnchor = interchange[reduceDim];
    LDBG("Collecting fusible producers that share reduction axis: "
         << reduceDim << ". W.r.t to the anchor the axis is "
         << reduceDimInAnchor);
    auto isFusibleProducerTestFunc =
        std::bind(isFusibleProducerForConsumerWithReductionAxes,
                  std::placeholders::_1, analyzer, reduceDimInAnchor);
    // Collect the fusible producers
    SetVector<Operation *> producerOps;
    traceBackToFusibleProducersForConsumer(reduceOp, producerOps,
                                           isFusibleProducerTestFunc);
    // Record producer info
    FusibleProducers producers;
    producers.setProducerInfo(
        std::move(producerOps), /*identifier=*/NamedAttribute(
            StringAttr::get(ctx,
                            llvm::formatv(kReductionFusibleProducerFormat,
                                          reduceOpInfo.idx, reduceDimInAnchor)),
            UnitAttr::get(ctx)));
    result.consumer2ProducerMap.insert(
        {std::make_pair(reduceOp, reduceDimInAnchor), producers});
  }
  return result;
}

FailureOr<FusibleProducerAnalysisResult>
analyzeProducersForStoreOp(hfusion::StoreOp storeOp, StoreOpInfo &storeOpInfo,
                           const SetVector<int64_t> &reduceDimsInAnchor,
                           DimensionAnalyzer *analyzer) {
  auto returnNumAttr =
      storeOp->getAttrOfType<IntegerAttr>(hfusion::ReturnOperandNumAttr::name);
  if (!returnNumAttr)
    return FailureOr<FusibleProducerAnalysisResult>();

  MLIRContext *ctx = storeOp->getContext();
  FusibleProducerAnalysisResult result;
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//";
  LDBG(logLineComment);
  LDBG("Analyzing reduction producers for consumer #"
       << returnNumAttr.getInt());
  LDBG(*storeOp);
  LDBG(logLineComment);
#endif
  // Build consumer information
  result.consumerInfo = ConsumerWithReduction(
      /*consumer=*/storeOp,
      /*type=*/ConsumerType::kOutput,
      /*identifier=*/
      NamedAttribute(StringAttr::get(ctx, hfusion::ReturnOperandNumAttr::name),
                     returnNumAttr));

  // We need to check whether each axis in the output is a reduce dimension.
  // For example:
  //   Total Axis    = [d0, d1, d2, d3]
  //   Output        = [d0, d2, d3]
  //   Reduce Input  = [d2, d3], Reduce dim = [0, 1]
  // Then we need to collect the fusible group information for axis d2 and d3.
  SetVector<int64_t> looselyReductionDims;
  SetVector<int64_t> strictlyParallelDims;
  for (int64_t outputDim = 0; outputDim < storeOp.getNumLoops(); ++outputDim) {
    bool outputDimIsReduction = false;
    for (int64_t reduceDimInAnchor : reduceDimsInAnchor) {
      // Only need to consider return values that contains reduction axes.
      if (!analyzer->isDimensionEqualToAnchor(
              reduceDimInAnchor, {storeOp->getResult(0), outputDim}))
        continue;

      outputDimIsReduction = true;
      looselyReductionDims.insert(outputDim);
      LDBG("Output's axis " << outputDim << " is reduction axis (in anchor): "
                            << reduceDimInAnchor);
      LDBG("Collecting fusible producers that share reduction axis");
      auto isFusibleProducerTestFunc =
          std::bind(isFusibleProducerForConsumerWithReductionAxes,
                    std::placeholders::_1, analyzer, reduceDimInAnchor);
      // Collect the fusible producers
      SetVector<Operation *> producerOps;
      traceBackToFusibleProducersForConsumer(storeOp, producerOps,
                                             isFusibleProducerTestFunc);
      // Record producer info
      FusibleProducers producers;
      producers.setProducerInfo(
          std::move(producerOps), /*identifier=*/NamedAttribute(
              StringAttr::get(ctx,
                              llvm::formatv(kReturnValueFusibleProducerFormat,
                                            returnNumAttr.getInt(),
                                            reduceDimInAnchor)),
              UnitAttr::get(ctx)));
      result.consumer2ProducerMap.insert(
          {std::make_pair(storeOp, reduceDimInAnchor), producers});
      break;
    }

    if (!outputDimIsReduction) {
      LDBG("Output's axis " << outputDim << " is strictly parallel axis");
      strictlyParallelDims.insert(outputDim);
    }
  }
#ifndef NDEBUG
  // sanity check to check if two sets are disjoint
  // use std::set since it provides set_intersection
  std::vector<int64_t> diff;
  std::set_intersection(looselyReductionDims.begin(),
                        looselyReductionDims.end(),
                        strictlyParallelDims.begin(),
                        strictlyParallelDims.end(), std::back_inserter(diff));
  LDBG("loosely reduction Dims : "
       << utils::debugger::to_string(looselyReductionDims));
  LDBG("strictly parallel Dims : "
       << utils::debugger::to_string(strictlyParallelDims));
  if (!diff.empty())
    llvm_unreachable("parallel and reduce dims are not disjoint!");
#endif

  bool doesOutputHaveReduceAxes = !looselyReductionDims.empty();
  storeOpInfo.looselyReductionDims = std::move(looselyReductionDims);
  storeOpInfo.strictlyParallelDims = std::move(strictlyParallelDims);
  return doesOutputHaveReduceAxes ? result
                                  : FailureOr<FusibleProducerAnalysisResult>();
}

} // namespace detail
} // namespace hfusion
} // namespace mlir