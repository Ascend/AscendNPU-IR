//===- AnyPBRInfoCollector.cpp - Def. for AnyPBR Info Collector --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the logic of collecting and analyzing kernel
// information.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRKernelInfoCollector.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AnyPBRKernelInfo.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "bishengir/Dialect/Utils/ReachabilityAnalyzer.h"
#include "llvm/Support/Debug.h"

// #include "AutoScheduleAttrDefs.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS()                                                                 \
  (llvm::dbgs() << '[' << DEBUG_TYPE << "] [AnyPBR Info Collector] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

void analyzeProducersForOutputsWithReductionAxes(AnyPBRKernelInfo *kernelInfo) {
  assert(kernelInfo != nullptr);
  if (kernelInfo->reduceOp2Info.empty())
    return;

  for (auto &[storeOp, storeOpInfo] : kernelInfo->storeOp2Info) {
    auto analysisResult = hfusion::detail::analyzeProducersForStoreOp(
        cast<hfusion::StoreOp>(storeOp), storeOpInfo,
        kernelInfo->reduceDimsInAnchor, kernelInfo->getAnalyzer());
    if (failed(analysisResult))
      continue;

    kernelInfo->recordFusibleProducerAnalysisResult(
        std::move(analysisResult.value()));
  }
}

static std::optional<hfusion::AtomicKind>
tryMapReduceToAtomicKind(linalg::ReduceOp reduceOp) {
  // We only support these kind of atomic kind now
  Block &body = reduceOp.getCombiner().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
  if (isa<arith::AddFOp>(bodyOp) || isa<arith::AddIOp>(bodyOp)) {
    return hfusion::AtomicKind::ADD;
  }
  if (isa<arith::MaximumFOp>(bodyOp) || isa<arith::MaxSIOp>(bodyOp) ||
      isa<arith::MaxNumFOp>(bodyOp)) {
    return hfusion::AtomicKind::MAX;
  }
  if (isa<arith::MinimumFOp>(bodyOp) || isa<arith::MinSIOp>(bodyOp) ||
      isa<arith::MinNumFOp>(bodyOp)) {
    return hfusion::AtomicKind::MIN;
  }
  return {};
}

void analyzeMultiCoreReduceInfo(AnyPBRKernelInfo *kernelInfo) {
  // TODO: Support multiple reducesOp bind to multi cores.
  // TODO: Support dynamic shape.
  if (kernelInfo->reduceOp2Info.size() > 1 ||
      !kernelInfo->isPureStaticKernel()) {
    return;
  }

  bool hasValidReduceToBindMultiCore = false;
  for (auto outputValue : kernelInfo->outputValues) {
    auto storeOp = outputValue.getDefiningOp<hfusion::StoreOp>();
    for (const auto &[reduceOp, _] : kernelInfo->reduceOp2Info) {
      // We can only bind multi-core to reduce if it is
      // reduce + store and there isn't any op in between.
      bool reduceImmFollowedByStore = llvm::all_of(
          reduceOp->getResult(0).getUsers(),
          [storeOp](const Operation *op) -> bool { return op == storeOp; });
      // TODO: Support reduce with multiple results too
      if (!reduceImmFollowedByStore || reduceOp->getResults().size() > 1) {
        continue;
      }

      auto maybeAtomicKind =
          tryMapReduceToAtomicKind(cast<linalg::ReduceOp>(reduceOp));
      if (maybeAtomicKind.has_value()) {
        storeOp.setAtomicKind(maybeAtomicKind.value());
        hasValidReduceToBindMultiCore = true;
      }
    }
  }
  kernelInfo->enableMultiCoreReduce =
      (kernelInfo->enableMultiCoreReduce && hasValidReduceToBindMultiCore);
}

//===----------------------------------------------------------------------===//
// AnyPBRKernelInfoCollector
//===----------------------------------------------------------------------===//

LogicalResult AnyPBRKernelInfoCollector::visitLinalgOpImpl(Operation *op) {
  auto *kernelInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getInfo());
  if (!kernelInfo)
    return failure();

  if (!isa<linalg::ReduceOp>(op))
    return success();

  auto reduceOp = cast<linalg::ReduceOp>(op);
  auto reduceInfo = kernelInfo->reduceOp2Info.at(reduceOp);
  // Collect reduce op's producer info
  auto *dimensionAnalyzer = kernelInfo->getAnalyzer();
  auto analysisResult = detail::analyzeProducersForReductionOp(
      reduceOp, reduceInfo, dimensionAnalyzer);
  kernelInfo->recordFusibleProducerAnalysisResult(std::move(analysisResult));
  return success();
}

LogicalResult AnyPBRKernelInfoCollector::postVisitFuncImpl(func::FuncOp f) {
  if (failed(KernelInfoCollector::postVisitFuncImpl(f)))
    return failure();

  auto *kernelInfo = dyn_cast_or_null<AnyPBRKernelInfo>(getInfo());
  assert(kernelInfo != nullptr);
  if (kernelInfo == nullptr) {
    return failure();
  }
  for (const auto &[_, info] : kernelInfo->reduceOp2Info) {
    kernelInfo->reduceDimsInAnchor.clear();
    assert(!info.inputsInterchange.empty());
    auto interchange = info.inputsInterchange.front();
    for (auto reduceDim : info.reductionDims)
      kernelInfo->reduceDimsInAnchor.insert(interchange[reduceDim]);
  }
  LDBG("Reduce Dims are " << utils::debugger::to_string(
           kernelInfo->reduceDimsInAnchor));

  // Collect producer information for hfusion.store op
  // We collect the information here because it depends on reduction op's
  // information.
  analyzeProducersForOutputsWithReductionAxes(kernelInfo);

  // This part of code is for binding reduce to multi core
  if (kernelInfo->enableMultiCoreReduce)
    analyzeMultiCoreReduceInfo(kernelInfo);

  return success();
}
