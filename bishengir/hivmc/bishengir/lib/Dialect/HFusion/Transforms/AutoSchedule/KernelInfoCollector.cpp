//===- KernelInfoCollector.cpp - Def. for Kernel Info Collector --*- C++-*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfoCollector.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "bishengir/Dialect/Utils/ReachabilityAnalyzer.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "AutoScheduleAttrDefs.h"

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS()                                                                 \
  (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Kernel Info Collector] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {

/// Set a unit attribute named \c attrName to \c op.
void setNamedUnitAttr(Operation *op, StringRef attrName) {
  assert(op != nullptr);
  op->setAttr(attrName, UnitAttr::get(op->getContext()));
}

/// Collect shaped type arguments used by reshape op.
SmallVector<Value> getMaybeReshapedInputs(ArrayRef<BlockArgument> inputs) {
  SmallVector<Value> result;
  result.append(inputs.begin(), inputs.end());
  for (auto [idx, arg] : llvm::enumerate(inputs)) {
    Type argType = arg.getType();
    if (!isa<ShapedType>(argType)) {
      continue;
    }
    auto maybeArgReshaped =
        hfusion::traceReshapeOrSliceSingleConsumerOrSelf(arg);
    LDBG("maybeArgReshaped [" << idx << "]: " << maybeArgReshaped);
    if (!hfusion::isReshapeOrSliceOp(maybeArgReshaped.getDefiningOp())) {
      continue;
    }
    result[idx] = maybeArgReshaped;
  }
  return result;
}

hfusion::detail::MatmulInfo collectMatmulInfo(Operation *op,
                                              const KernelInfo &kernelInfo) {
  hfusion::detail::MatmulInfo info;
  info.idx = kernelInfo.matmulOp2Info.size();
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    info.numParallel = matmulOp.getNumParallelLoops();
    info.numReduction = matmulOp.getNumReductionLoops();
  }
  if (auto matmulOp = dyn_cast<linalg::MatmulTransposeAOp>(op)) {
    info.transposeA = true;
    info.numParallel = matmulOp.getNumParallelLoops();
    info.numReduction = matmulOp.getNumReductionLoops();
  }
  if (auto matmulOp = dyn_cast<linalg::MatmulTransposeBOp>(op)) {
    info.transposeB = true;
    info.numParallel = matmulOp.getNumParallelLoops();
    info.numReduction = matmulOp.getNumReductionLoops();
  }
  if (auto operandA = dyn_cast<BlockArgument>(op->getOperand(0))) {
    info.tensorAId = operandA.getArgNumber();
  }
  if (auto operandB = dyn_cast<BlockArgument>(op->getOperand(1))) {
    info.tensorBId = operandB.getArgNumber();
  }
  return info;
}

hfusion::detail::ReduceInfo collectReduceInfo(linalg::ReduceOp reduceOp,
                                              const KernelInfo &kernelInfo) {
  hfusion::detail::ReduceInfo info;
  info.idx = kernelInfo.reduceOp2Info.size();
  info.numLoops = reduceOp.getNumLoops();
  info.numResults = reduceOp->getNumResults();
  info.reductionDims.insert(reduceOp.getDimensions().begin(),
                            reduceOp.getDimensions().end());
  return info;
}

hfusion::detail::BroadcastInfo
collectBroadcastInfo(linalg::BroadcastOp brcOp,
                     [[maybe_unused]] const KernelInfo &) {
  hfusion::detail::BroadcastInfo info;
  info.numLoops = brcOp.getNumLoops();
  info.broadcastDims.insert(brcOp.getDimensions().begin(),
                            brcOp.getDimensions().end());
  return info;
}

hfusion::detail::TransposeInfo
collectTransposeInfo(linalg::TransposeOp transposeOp,
                     [[maybe_unused]] const KernelInfo &) {
  hfusion::detail::TransposeInfo info;
  info.numLoops = transposeOp.getNumLoops();
  ShapedType inputTy = transposeOp.getInput().getType();
  int64_t rank = inputTy.getRank();
  info.elemBitwidth = inputTy.getElementTypeBitWidth();

  ArrayRef<int64_t> permutation = transposeOp.getPermutation();
  info.transposeLastDim = (permutation.back() != rank - 1);

  DenseMap<int64_t, int64_t> permDims;
  for (int64_t permIdx : permutation) {
    if (permIdx != permutation[permIdx]) {
      // current implementation only considers general scenarios of transpose
      // TODO: optimize `1xA -> Ax1` or `1x1` transpose for better performance
      permDims[permIdx] = permutation[permIdx];
    }
  }
  if (permDims.size() > 2) {
    // should guarantee to be binary transpose
    transposeOp->emitError() << "contains more than 1 pair of permuted dims!"
                                " Should be decomposed before Auto-Schedule";
    return {};
  }
  auto [idx0, idx1] = *permDims.begin();
  info.permuteDims = std::make_pair(std::min(idx0, idx1), std::max(idx0, idx1));
  LDBG("permuteDims: " << std::get<0>(info.permuteDims) << ", "
                       << std::get<1>(info.permuteDims));
  return info;
}

hfusion::detail::CastInfo collectCastInfo(hfusion::CastOp castOp) {
  hfusion::detail::CastInfo info;
  info.srcType = castOp.getInputs()[0].getType();
  info.dstType = castOp.getOutputs()[0].getType();
  info.srcElemType = getElementTypeOrSelf(info.srcType);
  info.dstElemType = getElementTypeOrSelf(info.dstType);
  if (auto shapedType = dyn_cast<ShapedType>(info.srcType)) {
    ArrayRef<int64_t> shape = shapedType.getShape();
    info.shape = SmallVector<int64_t>{shape.begin(), shape.end()};
  }
  if (auto shapedType = dyn_cast<ShapedType>(info.dstType)) {
    info.rank = shapedType.getRank();
  } else {
    info.rank = 0;
  }
  return info;
}

hfusion::detail::ConcatInfo collectConcatInfo(tensor::ConcatOp concatOp) {
  hfusion::detail::ConcatInfo info;
  info.rank = concatOp.getRank();
  info.concatDim = static_cast<int64_t>(concatOp.getDim());
  info.elemBitwidth = concatOp.getResultType().getElementTypeBitWidth();
  return info;
}

void recordAnchorInfoForDpsOp(
    DestinationStyleOpInterface op, hfusion::detail::OpInfo &info,
    hfusion::detail::DimensionAnalyzer &dimensionAnalyzer) {
  for (auto input : op.getDpsInputs()) {
    info.inputsAnchorDimension.push_back(
        dimensionAnalyzer.getCommonAxis(input));
    info.inputsInterchange.push_back(
        dimensionAnalyzer.getNormalizedInterchange(input));
  }
  for (auto result : op->getResults()) {
    info.resultsAnchorDimension.push_back(
        dimensionAnalyzer.getCommonAxis(result));
    info.resultsInterchange.push_back(
        dimensionAnalyzer.getNormalizedInterchange(result));
  }
}

void recordAnchorInfo(Operation *op, hfusion::detail::OpInfo &info,
                      hfusion::detail::DimensionAnalyzer &dimensionAnalyzer) {
  for (auto result : op->getResults()) {
    info.resultsAnchorDimension.push_back(
        dimensionAnalyzer.getCommonAxis(result));
    info.resultsInterchange.push_back(
        dimensionAnalyzer.getNormalizedInterchange(result));
  }
}

bool hasProducerOperand(Operation *op) {
  return llvm::any_of(op->getOperands(), [](Value operand) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) {
      return false;
    }
    return defOp->hasAttr(kIntermediateProducerTagName);
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelInfoCollector
//===----------------------------------------------------------------------===//

KernelInfo *KernelInfoCollector::getInfo() {
  assert(info_ != nullptr);
  return info_;
}

KernelInfo *KernelInfoCollector::getInfo() const {
  assert(info_ != nullptr);
  return info_;
}

BitVector KernelInfoCollector::getAnchorCommonAxis(Value value) const {
  assert(info_ != nullptr);
  assert(info_->getAnalyzer() != nullptr);
  return info_->getAnalyzer()->getCommonAxis(value);
}

LogicalResult KernelInfoCollector::run() {
  func::FuncOp f = this->getInfo()->originalKernel;
  if (failed(visitFuncImpl(f)))
    return f->emitError() << "failed to visit func";

  return postVisitFuncImpl(f);
}

LogicalResult KernelInfoCollector::visitFuncImpl(func::FuncOp f) {
  auto walkResult = f.getOperation()->walk([&](Operation *op) {
    // TransposeOp is also an LinalgOp, so has to come first
    auto visitStatus = TypeSwitch<Operation *, LogicalResult>(op)
                           .Case<linalg::LinalgOp>(
                               [&](Operation *op) { return visitLinalgOp(op); })
                           .Case<tensor::ExtractOp>([&](Operation *op) {
                             return visitTensorExtractOp(op);
                           })
                           .Case<tensor::PadOp>([&](Operation *op) {
                             return visitTensorPadOp(op);
                           })
                           .Case<tensor::ConcatOp>([&](Operation *op) {
                             return visitTensorConcatOp(op);
                           })
                           .Case<tensor::InsertSliceOp>([&](Operation *op) {
                             return visitTensorInsertSliceOp(op);
                           })
                           .Case<tensor::ExtractSliceOp>([&](Operation *op) {
                             return visitTensorExtractSliceOp(op);
                           })
                           .Case<hfusion::DeinterleaveOp>([&](Operation *op) {
                             return visitDeinterleaveOp(op);
                           })
                           .Case<hfusion::InterleaveOp>([&](Operation *op) {
                             return visitInterleaveOp(op);
                           })
                           .Default([&](Operation *op) { return success(); });
    if (failed(visitStatus)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  // arith op can be marked as intermediate producer only if any of its operand
  // is intermediate producer
  f->walk([&](Operation *op) {
    if (utils::isArithOp(op) && hasProducerOperand(op)) {
      setNamedUnitAttr(op, kIntermediateProducerTagName);
    }
  });
  return success();
}

LogicalResult KernelInfoCollector::postVisitFuncImpl(func::FuncOp f) {
  // Mark multi buffer
  auto kernelInputs = getMaybeReshapedInputs(f.getArguments());
  utils::BufferAnalysisOptions options;
  if (getScheduleOptions().enableAutoMultiBuffer) {
    for (auto ioValues : kernelInputs) {
      // get IOValues copied result
      auto copyOpUsers =
          llvm::make_filter_range(ioValues.getUsers(), [](Operation *user) {
            return (isa<hfusion::StoreOp>(user) || isa<hfusion::LoadOp>(user));
          });
      if (llvm::hasSingleElement(copyOpUsers)) {
        Operation *copyOp = *(copyOpUsers.begin());
        LDBG("Auto multi-buffer value: " << copyOp->getResult(0));
        options.multiBufferCount[copyOp->getResult(0)] = 2;
      }
    }
  }
  // Count max buffer
  options.enableDmaOpt = getScheduleOptions().enableCountBufferDmaOpt;
  if (failed(countMaxBuffer(options)))
    return failure();

  return success();
}

LogicalResult KernelInfoCollector::visitLinalgOp(Operation *op) {
  KernelInfo *kernelInfo = getInfo();
  if (!kernelInfo)
    return failure();

  //===--------------------------------------------------------------------===//
  // Collect Op-agnostic information
  //===--------------------------------------------------------------------===//
  // Collect the element type with smallest bits
  auto elementTypes = llvm::map_to_vector(
      llvm::make_filter_range(op->getOperandTypes(), utils::hasRank),
      [](const Type &t) { return cast<ShapedType>(t).getElementType(); });
  auto localMinType = utils::getSmallestElementType(elementTypes);
  auto globalMinType = kernelInfo->smallestElementType;
  if (globalMinType == Type() || localMinType.getIntOrFloatBitWidth() <
                                     globalMinType.getIntOrFloatBitWidth()) {
    kernelInfo->smallestElementType = localMinType;
  } else {
    kernelInfo->smallestElementType = globalMinType;
  }
  //===--------------------------------------------------------------------===//
  // Collect Op-specific information
  //===--------------------------------------------------------------------===//
  if (isMatmulOps(op)) {
    auto info = collectMatmulInfo(op, *kernelInfo);
    kernelInfo->matmulOp2Info[op] = std::move(info);
  }
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    // Collect reduce op info.
    auto info = collectReduceInfo(reduceOp, *kernelInfo);
    recordAnchorInfoForDpsOp(cast<DestinationStyleOpInterface>(op), info,
                             *kernelInfo->getAnalyzer());
    kernelInfo->reduceOp2Info.insert({reduceOp, info});
  }
  if (auto brcOp = dyn_cast<linalg::BroadcastOp>(op)) {
    auto info = collectBroadcastInfo(brcOp, *kernelInfo);
    kernelInfo->broadcastOp2Info.insert({brcOp, info});
  }
  if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
    auto info = collectTransposeInfo(transposeOp, *kernelInfo);
    recordAnchorInfoForDpsOp(cast<DestinationStyleOpInterface>(op), info,
                             *kernelInfo->getAnalyzer());
    kernelInfo->transposeOp2Info.insert({transposeOp, info});
  }
  if (auto castOp = dyn_cast<hfusion::CastOp>(op)) {
    auto info = collectCastInfo(castOp);
    recordAnchorInfoForDpsOp(cast<DestinationStyleOpInterface>(op), info,
                             *kernelInfo->getAnalyzer());
    kernelInfo->castOp2Info.insert({castOp, info});
  }
  if (auto storeOp = dyn_cast<hfusion::StoreOp>(op)) {
    detail::StoreOpInfo info(storeOp.getNumLoops());
    recordAnchorInfoForDpsOp(cast<DestinationStyleOpInterface>(op), info,
                             *kernelInfo->getAnalyzer());
    kernelInfo->storeOp2Info.insert({storeOp, info});
  } else {
    // Tag all linalg ops (except for hfusion::StoreOp) as intermediate
    // producers so that they can be matched during scheduling.
    setNamedUnitAttr(op, kIntermediateProducerTagName);
  }
  return visitLinalgOpImpl(op);
}

LogicalResult KernelInfoCollector::visitTensorExtractOp(Operation *op) {
  if (llvm::any_of(op->getUsers(), [](Operation *user) {
        return isa<linalg::LinalgOp>(user) ||
               (utils::isArithOp(user) && !isa<arith::ConstantOp>(user));
      })) {
    // Tag `tensor.extract` op that is used by a `linalg.op` or non constant
    // `arith op` as intermediate producers so that they can be matched during
    // scheduling.
    setNamedUnitAttr(op, kIntermediateProducerTagName);
  }
  return visitTensorExtractOpImpl(op);
}

LogicalResult KernelInfoCollector::visitTensorPadOp(Operation *op) {
  // Tag `tensor.pad` op to be intermediate producer as default.
  // TODO: check when `tensor.pad` op should not be fused
  setNamedUnitAttr(op, kIntermediateProducerTagName);
  return visitTensorPadOpImpl(op);
}

LogicalResult KernelInfoCollector::visitTensorConcatOp(Operation *op) {
  // Tag `tensor.concat` op to be intermediate producer as default.
  // TODO: check when `tensor.concat` op should not be fused
  setNamedUnitAttr(op, kIntermediateProducerTagName);
  // collect info and update anchor for concat op
  KernelInfo *kernelInfo = getInfo();
  if (!kernelInfo)
    return failure();
  if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    auto info = collectConcatInfo(concatOp);
    recordAnchorInfo(concatOp, info, *kernelInfo->getAnalyzer());
    kernelInfo->concatOp2Info.insert({concatOp, info});
  }
  return visitTensorConcatOpImpl(op);
}

/// Check if op has sources marked as producer
bool hasProducerSources(Operation *op) {
  std::queue<Operation *> q;
  q.push(op);
  DenseSet<Operation *> visited;

  while (!q.empty()) {
    Operation *curOp = q.front();
    q.pop();
    if (visited.contains(curOp)) {
      continue;
    }
    visited.insert(curOp);
    if (curOp->getAttr(kIntermediateProducerTagName)) {
      return true;
    }
    for (const Value &src : curOp->getOperands()) {
      Operation *defOp = src.getDefiningOp();
      if (defOp != nullptr) {
        q.push(defOp);
      }
    }
  }
  return false;
}

LogicalResult KernelInfoCollector::visitTensorInsertSliceOp(Operation *op) {
  if (hasProducerSources(op)) {
    // Tag `tensor.insert_slice` op with intermediate producer sources as
    // intermediate producers, so that they can be matched during scheduling.
    setNamedUnitAttr(op, kIntermediateProducerTagName);
  }
  return visitTensorInsertSliceOpImpl(op);
}

LogicalResult KernelInfoCollector::visitTensorExtractSliceOp(Operation *op) {
  auto sliceOp = cast<tensor::ExtractSliceOp>(op);
  detail::ExtractSliceInfo info;
  info.resultsAnchorDimension.push_back(
      getAnchorCommonAxis(sliceOp.getResult()));

  RankedTensorType srcType = sliceOp.getSourceType();
  int64_t rank = srcType.getRank();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> sliceShape = sliceOp.getStaticSizes();
  SmallVector<OpFoldResult> mixSrcShape = sliceOp.getMixedSizes();
  SmallVector<OpFoldResult> mixSliceShape = sliceOp.getMixedSizes();

  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(srcShape[i]) &&
        ShapedType::isDynamic(sliceShape[i])) {
      // src shape and slice size are both dynamic value
      Value srcV = mixSrcShape[i].get<Value>();
      Value sliceV = mixSliceShape[i].get<Value>();
      if (srcV == sliceV) {
        info.fullSliceDims.insert(i);
      } else {
        info.partialSlicedDims.insert(i);
      }
      continue;
    }

    if (srcShape[i] == sliceShape[i]) {
      // src shape and slice size are same static constant
      info.fullSliceDims.insert(i);
    } else {
      // src shape and slice size are different static constant,
      // or either of them is dynamic value
      info.partialSlicedDims.insert(i);
    }
  }

  getInfo()->extractSliceOp2Info[sliceOp] = std::move(info);
  return visitTensorExtractSliceOpImpl(op);
}

LogicalResult KernelInfoCollector::visitDeinterleaveOp(Operation *op) {
  setNamedUnitAttr(op, kIntermediateProducerTagName);
  return visitDeinterleaveOpImpl(op);
}

LogicalResult KernelInfoCollector::visitInterleaveOp(Operation *op) {
  setNamedUnitAttr(op, kIntermediateProducerTagName);
  return visitInterleaveOpImpl(op);
}

LogicalResult KernelInfoCollector::countMaxBuffer(
    const utils::BufferAnalysisOptions &options) {
  KernelInfo *info = getInfo();
  OpBuilder opBuilder(info->originalKernel.getContext());

  // Broadcast and reduction ops should be aligned.
  auto alignmentAttr =
      hivm::AlignKindAttr::get(opBuilder.getContext(), hivm::AlignKind::ALIGN);

  SetVector<Operation *> reductionOps;
  for (auto [key, _] : info->reduceOp2Info)
    reductionOps.insert(key);

  SetVector<Operation *> broadcastOps;
  for (auto [key, _] : info->broadcastOp2Info)
    broadcastOps.insert(key);

  SmallVector<Operation *> markOps;
  for (Operation *op : llvm::concat<Operation *>(
           llvm::to_vector(broadcastOps), llvm::to_vector(reductionOps))) {
    opBuilder.setInsertionPointAfter(op);
    // Extra buffer size is inferred from broadcast/reduction op's result.
    auto markOp =
        opBuilder.create<annotation::MarkOp>(op->getLoc(), op->getResult(0));
    markOp->setAttr(hivm::AlignKindAttr::name, alignmentAttr);
    markOps.push_back(markOp);
  }
  LDBG("---Counting max buffer ...");
  LDBG("-----Func after annotation: \n" << info->originalKernel);
  std::optional<int64_t> maxBufferCnt =
      utils::countMaxBuffer(info->originalKernel, options);
  // Erase marks after counting max buffer because it keeps tensor's from being
  // dce.
  for (auto *op : llvm::reverse(markOps))
    op->erase();

  if (maxBufferCnt.has_value()) {
    int64_t maxBufferCntInitVal = maxBufferCnt.value();
    assert(maxBufferCntInitVal > 0 && "max buffer count should be positive");

    int tuningDelta = getScheduleOptions().maxBufferCntTuning;
    int64_t maxBufferCntAfterTuning = maxBufferCntInitVal + tuningDelta;
    maxBufferCntAfterTuning = std::max((int64_t)1, maxBufferCntAfterTuning);
    info->maxBufferCnt = maxBufferCntAfterTuning;
    LDBG("-----Max buffer count is: " << info->maxBufferCnt);
    return success();
  }
  return info->originalKernel.emitError("Max buffer count is zero!");
}
