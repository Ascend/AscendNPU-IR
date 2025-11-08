//===--- EnableStrideAlign.cpp ---- enable stride_align marks -------------===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>
#include <numeric>

#define DEBUG_TYPE "hivm-enable-stride-align"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_ENABLESTRIDEALIGN
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

struct EnableStrideAlignPass
    : public impl::EnableStrideAlignBase<EnableStrideAlignPass> {
public:
  void runOnOperation() override;
};
} // namespace

struct NormalizeAlignInfoPattern : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override {
    if (!utils::isAnnotationWithAttr(markOp, StrideAlignDimsAttr::name)) {
      return failure();
    }

    // check the validity of align info
    auto alignDims = markOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignDimsAttr::name);
    auto alignBytes = markOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignValueInByteAttr::name);
    if (alignDims == nullptr || alignBytes == nullptr) {
      return markOp.emitError()
             << "align info must contain align dims and align bytes";
    }

    if (alignDims.size() != alignBytes.size()) {
      return markOp.emitError()
             << "size of align bytes should be matched with size of align dims";
    }

    // TODO : merge same dimension setting

    // sort the align dims of mark storage align attr
    bool isSorted = llvm::is_sorted(alignDims.asArrayRef());
    if (isSorted) {
      return failure();
    }

    auto sortedInfo = sortAlignInfo(alignDims, alignBytes);
    SmallVector<int32_t> sortedAlignDims;
    SmallVector<int32_t> sortedAlignBytes;
    for (auto [sortedAlignDim, sortedAlignByte] : sortedInfo) {
      sortedAlignDims.push_back(sortedAlignDim);
      sortedAlignBytes.push_back(sortedAlignByte);
    }
    rewriter.modifyOpInPlace(markOp, [&]() {
      markOp->setAttr(hivm::StrideAlignDimsAttr::name,
                      DenseI32ArrayAttr::get(markOp.getContext(),
                                             ArrayRef(sortedAlignDims)));
      markOp->setAttr(hivm::StrideAlignValueInByteAttr::name,
                      DenseI32ArrayAttr::get(markOp.getContext(),
                                             ArrayRef(sortedAlignBytes)));
    });

    return success();
  }
};

bool isContain(ArrayRef<int32_t> alignDims, ArrayRef<int32_t> alignBytes,
               ArrayRef<int32_t> otherAlignDims,
               ArrayRef<int32_t> otherAlignBytes) {
  if (alignDims.size() < otherAlignDims.size()) {
    return false;
  }

  size_t dimSize = alignDims.size();
  size_t otherDimSize = otherAlignDims.size();
  for (size_t i = 0; i < dimSize; i++) {
    for (size_t j = 0; j < otherDimSize; j++) {
      if (alignDims[i] == otherAlignDims[j] &&
          alignBytes[i] != otherAlignBytes[j]) {
        return false;
      }
    }
  }

  return true;
}

struct AddAlignAnnotationMarkForAlloc
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp->hasAttr(hivm::StrideAlignDimsAttr::name)) {
      return failure();
    }
    auto mayMarkOp = utils::getAnnotateOpWithAttr(
        allocOp.getResult(), hivm::StrideAlignDimsAttr::name);
    if (mayMarkOp.has_value()) {
      // already added
      return failure();
    }

    auto alignDims = allocOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignDimsAttr::name);
    auto alignBytes = allocOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignValueInByteAttr::name);
    createAlignMarkOp(rewriter, allocOp.getLoc(), allocOp.getResult(),
                      alignDims, alignBytes);
    return success();
  }
};

bool markAlignInfoForNoAlignOperand(PatternRewriter &rewriter, Operation *op,
                                    Value tobeStrideAlignOperand,
                                    const ArrayRef<int32_t> &alignDims,
                                    const ArrayRef<int32_t> &alignBytes) {
  // no storage align info, just create new one to add info
  auto [adjustedAlignDims, adjustedAlignBytes] =
      adjustAlignInfo(op, tobeStrideAlignOperand, alignDims, alignBytes);
  if (adjustedAlignDims.empty()) {
    return false;
  }

  LDBG("The propagate operand " << tobeStrideAlignOperand << " of the hivmOp "
                                << *op << " is aligned as");
  LLVM_DEBUG(dump(adjustedAlignDims, adjustedAlignBytes, DEBUG_TYPE));
  createAlignMarkOp(rewriter, tobeStrideAlignOperand.getLoc(),
                    tobeStrideAlignOperand, adjustedAlignDims,
                    adjustedAlignBytes);
  return true;
}

bool markAlignInfoForAlignedOperand(PatternRewriter &rewriter, Operation *op,
                                    Value tobeStrideAlignOperand,
                                    Operation *origAlignMarkedOp,
                                    const ArrayRef<int32_t> &alignDims,
                                    const ArrayRef<int32_t> &alignBytes) {
  auto itAlignDims =
      origAlignMarkedOp->template getAttrOfType<DenseI32ArrayAttr>(
          hivm::StrideAlignDimsAttr::name);
  auto itAlignBytes =
      origAlignMarkedOp->template getAttrOfType<DenseI32ArrayAttr>(
          hivm::StrideAlignValueInByteAttr::name);
  assert(itAlignDims != nullptr && itAlignBytes != nullptr);
  if (!isContain(itAlignDims, itAlignBytes, ArrayRef(alignDims),
                 ArrayRef(alignBytes))) {
    // update to unioned storage align info
    auto [adjustedAlignDims, adjustedAlignBytes] =
        adjustAlignInfo(op, tobeStrideAlignOperand, alignDims, alignBytes);
    if (adjustedAlignDims.empty() ||
        isContain(itAlignDims, itAlignBytes, adjustedAlignDims,
                  adjustedAlignBytes)) {
      return false;
    }
    LDBG("The propagate operand " << tobeStrideAlignOperand << " of the hivmOp "
                                  << *op << " is aligned as");
    LLVM_DEBUG(dump(adjustedAlignDims, adjustedAlignBytes, DEBUG_TYPE));
    createAlignMarkOp(rewriter, tobeStrideAlignOperand.getLoc(),
                      tobeStrideAlignOperand, adjustedAlignDims,
                      adjustedAlignBytes);
    return true;
  }
  return false;
}

LogicalResult processAlignPropagationAmongOperationOperands(
    Operation *op, llvm::SmallVectorImpl<Value> &tobeStrideAlignOperands,
    PatternRewriter &rewriter) {
  auto maybeMarkOps = utils::getAnnotateOpWithAttrForEachOperand(
      tobeStrideAlignOperands, hivm::StrideAlignDimsAttr::name);
  if (llvm::none_of(maybeMarkOps, [](std::optional<Operation *> maybeMarkOp) {
        return maybeMarkOp.has_value();
      })) {
    return failure();
  }

  // union stride align information
  llvm::SmallVector<int32_t> unionAlignDims;
  llvm::SmallVector<int32_t> unionAlignBytes;
  auto markdOps = llvm::to_vector<6>(llvm::make_filter_range(
      maybeMarkOps, [](std::optional<Operation *> maybeMarkOp) {
        return maybeMarkOp.has_value();
      }));
  for (const std::optional<Operation *> &markOp : markdOps) {
    auto itAlignDims = markOp.value()->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignDimsAttr::name);
    auto itAlignBytes = markOp.value()->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignValueInByteAttr::name);
    assert(itAlignDims != nullptr && itAlignBytes != nullptr);
    auto [newAlignDims, newAlignBytes] =
        unionAlignInfo(ArrayRef(unionAlignDims), ArrayRef(unionAlignBytes),
                       itAlignDims, itAlignBytes);
    unionAlignDims = newAlignDims;
    unionAlignBytes = newAlignBytes;
  }

  bool isChanged = false;
  for (auto [maybeMarkOp, tobeStrideAlignOperand] :
       llvm::zip(maybeMarkOps, tobeStrideAlignOperands)) {
    if (!maybeMarkOp.has_value()) {
      // no storage align info, just create new one to add info
      isChanged =
          markAlignInfoForNoAlignOperand(rewriter, op, tobeStrideAlignOperand,
                                         unionAlignDims, unionAlignBytes);
      continue;
    }
    isChanged = markAlignInfoForAlignedOperand(
        rewriter, op, tobeStrideAlignOperand, maybeMarkOp.value(),
        unionAlignDims, unionAlignBytes);
  }
  return isChanged ? success() : failure();
}

template <typename OPTy>
struct PropagateAlignAmongOperationOperands : public OpRewritePattern<OPTy> {
  using OpRewritePattern<OPTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OPTy op,
                                PatternRewriter &rewriter) const override {
    auto hivmOp = dyn_cast<hivm::HIVMStructuredOp>(op.getOperation());
    if (!hivmOp) {
      return failure();
    }

    if (isa<hivm::LoadOp>(op.getOperation()) ||
        isa<hivm::StoreOp>(op.getOperation())) {
      // no need to do stride align for gm
      return failure();
    }

    auto unTempUBOperands = hivmOp.getTargetSpaceOperands(
        hivm::AddressSpace::UB, false /*includeTmpBuffer*/);
    llvm::SmallVector<Value> tobeStrideAlignOperands;
    for (size_t i = 0; i < unTempUBOperands.size(); i++) {
      auto operType = unTempUBOperands[i].getType();
      if (operType.isIntOrFloat()) {
        continue;
      }
      assert(isa<MemRefType>(operType));
      if (cast<MemRefType>(operType).getRank() == 0) {
        continue;
      }
      tobeStrideAlignOperands.push_back(unTempUBOperands[i]);
    }

    return processAlignPropagationAmongOperationOperands(
        (Operation *)op.getOperation(), tobeStrideAlignOperands, rewriter);
  }
};

struct RemoveAlignMarkPattern : public OpRewritePattern<annotation::MarkOp> {
  using OpRewritePattern<annotation::MarkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(annotation::MarkOp markOp,
                                PatternRewriter &rewriter) const override {
    if (!utils::isAnnotationWithAttr(markOp,
                                     hivm::StrideAlignValueInByteAttr::name)) {
      return failure();
    }

    removeMarkOpAttr(markOp, hivm::StrideAlignDimsAttr::name, rewriter);
    removeMarkOpAttr(markOp, hivm::StrideAlignValueInByteAttr::name, rewriter);
    return success();
  }
};

template <typename AllocLikeOp>
struct EnableAlignAllocation : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto alignDimsAttr = allocOp->getAttr(hivm::StrideAlignDimsAttr::name);
    auto alignBytesAttr =
        allocOp->getAttr(hivm::StrideAlignValueInByteAttr::name);
    if (alignDimsAttr == nullptr && alignBytesAttr == nullptr)
      return failure();

    auto alignDims = cast<DenseI32ArrayAttr>(alignDimsAttr);
    auto alignBytes = cast<DenseI32ArrayAttr>(alignBytesAttr);
    auto unalignedTy = allocOp.getType();
    SmallVector<int> alignUnits = utils::collectAlignUnits<MemRefType>(
        alignDims.asArrayRef(), alignBytes.asArrayRef(), unalignedTy);

    SmallVector<OpFoldResult> shape = allocOp.getMixedSizes();
    auto [alignedShape, subShape] =
        calculateAlignedShape(rewriter, allocOp.getLoc(), shape, alignUnits);

    if (!isEqualConstantIntOrValueArray(alignedShape, subShape)) {
      SmallVector<Value> dynSizes;
      SmallVector<int64_t> staticSizes;
      dispatchIndexOpFoldResults(alignedShape, dynSizes, staticSizes);
      MemRefType alignedTy = MemRefType::Builder(unalignedTy)
                                 .setShape(staticSizes)
                                 .setLayout(MemRefLayoutAttrInterface());
      auto alignedAlloc =
          rewriter.create<AllocLikeOp>(allocOp.getLoc(), alignedTy, dynSizes);

      SmallVector<OpFoldResult> offsets(alignedTy.getRank(),
                                        rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(alignedTy.getRank(),
                                        rewriter.getIndexAttr(1));
      MemRefType resTy =
          cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              unalignedTy.getShape(), alignedTy, offsets, subShape, strides));
      Value alignedMemRef = rewriter.create<memref::SubViewOp>(
          allocOp.getLoc(), resTy, alignedAlloc, offsets, subShape, strides);
      if (failed(replaceAndPropagateMemRefType(rewriter, allocOp.getLoc(),
                                               allocOp, alignedMemRef))) {
        LDBG("Cannot replace with aligned memref " << (Value)alignedMemRef);
        return failure();
      }
      rewriter.eraseOp(allocOp);
    } else {
      LDBG("no need to do alignment");
      // When no new alloc op is created, success should still be returned
      // because marks are removed.
      rewriter.modifyOpInPlace(allocOp, [&]() {
        allocOp->removeAttr(hivm::StrideAlignDimsAttr::name);
        allocOp->removeAttr(hivm::StrideAlignValueInByteAttr::name);
      });
    }
    return success();
  }
};

template <typename PATTERN, typename... PATTERNS>
void addPattern(RewritePatternSet &patterns) {
  patterns.add<PATTERN>(patterns.getContext());
  if constexpr (sizeof...(PATTERNS) > 0) {
    addPattern<PATTERNS...>(patterns);
  }
}

template <typename... PATTERNS>
LogicalResult applyPatterns(MLIRContext *context, GreedyRewriteConfig config,
                            Operation *op, bool *changed = nullptr) {
  RewritePatternSet patterns(context);
  addPattern<PATTERNS...>(patterns);
  return applyPatternsGreedily(op, std::move(patterns), config, changed);
}

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<PropagateAlignAmongOperationOperands<OpType>>(
      patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populatePropagateAlignAmongOpOperandsPatterns(
    RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
  registerOne<::mlir::hivm::CopyOp>(patterns);
}

bool isSame(std::map<Operation *, std::unique_ptr<AlignInfo>> *lhs,
            std::map<Operation *, std::unique_ptr<AlignInfo>> *rhs) {
  if (lhs->size() != rhs->size()) {
    return false;
  }

  for (auto &lIt : *lhs) {
    auto rIt = rhs->find(lIt.first);
    if (rIt == rhs->end()) {
      return false;
    }

    // compare the align Info
    if ((*lIt.second) != (*rIt->second)) {
      return false;
    }
  }

  return true;
}

void dump(std::map<Operation *, std::unique_ptr<AlignInfo>> *alignInfoMap) {
#ifndef NDEBUG
  LDBG("dump align info map");
  for (auto &it : *alignInfoMap) {
    LDBG("operation " << *it.first);
    LLVM_DEBUG(it.second->dump());
  }
#endif
}

void collectAlignInfo(
    Operation *funcOp,
    std::map<Operation *, std::unique_ptr<AlignInfo>> *alignInfoMap) {
  funcOp->walk([&](Operation *op) {
    if (!utils::isAllocLikeOp(op)) {
      return WalkResult::advance();
    }

    if (!op->hasAttr(hivm::StrideAlignDimsAttr::name)) {
      return WalkResult::advance();
    }

    auto alignInfo = std::make_unique<AlignInfo>(
        op->getAttrOfType<DenseI32ArrayAttr>(hivm::StrideAlignDimsAttr::name),
        op->getAttrOfType<DenseI32ArrayAttr>(
            hivm::StrideAlignValueInByteAttr::name));

    (*alignInfoMap)[op] = std::move(alignInfo);
    return WalkResult::advance();
  });
}

LogicalResult propagteAlignInfoUpToAlloc(MLIRContext *context,
                                         Operation *funcOp,
                                         bool enableNormalize = false) {
  GreedyRewriteConfig config = GreedyRewriteConfig();
  if (enableNormalize) {
    // Normalize the mark op with storage align
    if (failed(applyPatterns<NormalizeAlignInfoPattern>(context, config,
                                                        funcOp))) {
      return failure();
    }
  }

  // Propagate align marks up to root allocations
  RewritePatternSet patterns(context);
  populatePropagateAlignUpToRootAllocationPattern(
      patterns, hivm::StrideAlignDimsAttr::name.str(),
      hivm::StrideAlignValueInByteAttr::name.str());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    LDBG("propagating up failed");
    return failure();
  }
  LDBG("IR after propagating up");
  LDBG(*funcOp);
  return success();
}

LogicalResult propagateDownAlignInfo(MLIRContext *context, Operation *funcOp) {
  GreedyRewriteConfig config = GreedyRewriteConfig();
  if (failed(applyPatterns<AddAlignAnnotationMarkForAlloc,
                           PropagateAlignDownToLeafOperandsPattern>(
          context, config, funcOp))) {
    LDBG("propagating down failed");
    return failure();
  }
  LDBG("IR after propagating down");
  LDBG(*funcOp);
  return success();
}

LogicalResult populatePropagateAlignAmongOpOperands(MLIRContext *context,
                                                    Operation *funcOp) {
  RewritePatternSet patterns(context);
  populatePropagateAlignAmongOpOperandsPatterns(patterns);
  GreedyRewriteConfig config = GreedyRewriteConfig();
  config.maxIterations = 10000;
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    LDBG("propagating in operation failed");
    return failure();
  }
  LDBG("IR after propagating in operation");
  LDBG(*funcOp);
  return success();
}

LogicalResult propagateAlignInfoToOperands(MLIRContext *context,
                                           Operation *funcOp) {
  // collect the align info on allocation
  std::map<Operation *, std::unique_ptr<AlignInfo>> prevAlignInfoMap;
  collectAlignInfo(funcOp, &prevAlignInfoMap);

  bool isChanged;
  int64_t iterationCount = 0;
  int64_t maxIterations = 10;
  do {
    isChanged = false;
    LDBG("iteration : " << iterationCount << "\n");
    // Propagate align marks of root allocations down to leaf Operand
    if (failed(propagateDownAlignInfo(context, funcOp))) {
      return failure();
    }

    // Step 2: Propagate align marks among the Operation Operands
    if (failed(populatePropagateAlignAmongOpOperands(context, funcOp))) {
      return failure();
    }

    // Propagate align marks up to root allocations again
    if (failed(propagteAlignInfoUpToAlloc(context, funcOp))) {
      return failure();
    }

    // collect the align info on allocation
    std::map<Operation *, std::unique_ptr<AlignInfo>> curAlignInfoMap;
    collectAlignInfo(funcOp, &curAlignInfoMap);
    isChanged = !isSame(&prevAlignInfoMap, &curAlignInfoMap);
    LDBG("isChanged : " << isChanged);
    prevAlignInfoMap = std::move(curAlignInfoMap);
    iterationCount++;
  } while (isChanged && iterationCount < maxIterations);

  if (isChanged && iterationCount >= maxIterations) {
    funcOp->emitError()
        << "propagate storage align failure and match the max iterations"
        << maxIterations;
    return failure();
  }
  return success();
}

void EnableStrideAlignPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  auto *context = &getContext();
  // Propagate align info up to alloc op
  if (failed(propagteAlignInfoUpToAlloc(context, funcOp,
                                        true /*enableNormalize*/))) {
    return signalPassFailure();
  }

  // Propagate align info to all operands
  if (failed(propagateAlignInfoToOperands(context, funcOp))) {
    return signalPassFailure();
  }

  GreedyRewriteConfig config = GreedyRewriteConfig();
  // Remove annotation.mark with storage align
  if (failed(applyPatterns<RemoveAlignMarkPattern>(context, config, funcOp))) {
    LDBG("remove align mark failed");
    return signalPassFailure();
  }
  LDBG("IR after propagating align marks");
  LDBG(funcOp);

  // Rewrite all allocations with align marks
  RewritePatternSet patterns(context);
  ConversionTarget target(getContext());
  patterns.add<EnableAlignAllocation<memref::AllocOp>,
               EnableAlignAllocation<memref::AllocaOp>>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    LDBG("enable align allocation failed");
    return signalPassFailure();
  }
  // Add metadata.
  funcOp->setAttr(hivm::StorageAlignedAttr::name, UnitAttr::get(context));
}

std::unique_ptr<Pass> mlir::hivm::createEnableStrideAlignPass() {
  return std::make_unique<EnableStrideAlignPass>();
}
