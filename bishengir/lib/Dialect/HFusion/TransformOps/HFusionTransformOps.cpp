//===- HFusionTransformOps.cpp - Implementation of HFusion transform ops --===//
//
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

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::hfusion;

namespace {
static constexpr llvm::StringLiteral kBufferSizeInByteAttr =
    "buffer_size_in_byte";
} // namespace

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheReadOp::apply(TransformRewriter &rewriter,
                   TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    hfusion::LoadOp cachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      auto *definingOp = opResult.getOwner();
      rewriter.setInsertionPointAfter(definingOp);
      cachedOp = createCacheRead(rewriter, opResult, definingOp->getLoc());
    } else if (auto blockArgument = dyn_cast_or_null<BlockArgument>(target)) {
      auto *insertPoint = &(blockArgument.getParentBlock()->front());
      rewriter.setInsertionPoint(insertPoint);
      cachedOp =
          createCacheRead(rewriter, blockArgument, insertPoint->getLoc());
    } else {
      llvm::report_fatal_error("unsupported type");
    }
    cachedOps.push_back(cachedOp.getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheReadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheWriteOp::apply(TransformRewriter &rewriter,
                    TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    FailureOr<hfusion::StoreOp> maybeCachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      CacheWriteOptions options = {
          /*outputOnly=*/getOutputOnly(),
          /*cacheWriteToOutputInit=*/getCacheWriteToOutputInit(),
          /*reshapeTrace=*/std::nullopt};
      maybeCachedOp = createCacheWrite(rewriter, opResult, options);
    } else {
      llvm::report_fatal_error("unsupported type");
    }
    if (failed(maybeCachedOp))
      return DiagnosedSilenceableFailure::definiteFailure();
    cachedOps.push_back((*maybeCachedOp).getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheWriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp
//===----------------------------------------------------------------------===//
void transform::ExtendedFuseIntoContainingOp::build(OpBuilder &builder,
                                                    OperationState &result,
                                                    Value producerOp,
                                                    Value containingOp) {
  result.addOperands({producerOp, containingOp});
  auto resultType = transform::AnyOpType::get(builder.getContext());
  result.addTypes({resultType, resultType});
}

bool transform::ExtendedFuseIntoContainingOp::allowsRepeatedHandleOperands() {
  // Allow repeated handles since we are fusing everything anyway.
  return true;
}

DiagnosedSilenceableFailure
transform::ExtendedFuseIntoContainingOp::fuseIntoOneContaining(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state,
    size_t index, Operation *containingOp) {
  assert(index < getFusedOp().size());
  assert(index < getNewContainingOp().size());

  SmallVector<Operation *> fusedOps;
  auto producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (std::empty(producerOps)) {
    results.set(cast<OpResult>(getFusedOp()[index]),
                SmallVector<mlir::Operation *>{});
    results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
    return DiagnosedSilenceableFailure::success();
  }

  SetVector<Operation *> remainingProducers(producerOps.begin(),
                                            producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<std::pair<Operation *, size_t>> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isAncestor(op);
          });
      LLVM_DEBUG(DBGS() << "producerOp: " << *producerOp << "\n");
      LLVM_DEBUG(DBGS() << "numUsesInContainingOp: " << numUsesInContainingOp
                        << "\n");
      if (numUsesInContainingOp > 0) {
        return std::make_pair(producerOp, it.index());
      }
    }
    return failure();
  };

  // Helper function to erase producerOp from eraseRemainingProducer if no
  // users.
  auto eraseRemainingProducer = [&](Operation *producerOp, size_t pos) {
    int64_t numUsesInContainingOp =
        llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
          return containingOp->isAncestor(op);
        });
    if (numUsesInContainingOp == 0) {
      remainingProducers.erase(remainingProducers.begin() + pos);
    }
  };

  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      auto diag = mlir::emitSilenceableFailure(getLoc())
                  << "could not find next producer to fuse into container";
      diag.attachNote(containingOp->getLoc()) << "containing op";
      return diag;
    }

    Operation *producerOp;
    size_t producerIndex;
    std::tie(producerOp, producerIndex) = *nextProducer;

    // Default diagnostic, to be complemented with more failure information.
    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << " into " << *containingOp;

    // Union the multiple consumers in containing op.
    bishengir::unionProducerUsers(rewriter, diag, producerOp, containingOp);

    auto [tiledOps, newContainingOp] = bishengir::tileAndFuseFirstExtractUse(
        rewriter, diag, producerOp, containingOp, getDuplicateProducer());
    if (!tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused a direct extract use\n"
                        << *containingOp << "\n");
      fusedOps.append(tiledOps);
      if (newContainingOp) {
        // Update handles associated with the containing op so we don't need
        // to invalidate them. This is a hack to support better composability
        // between tiling and fusion while a proper mechanism is being
        // investigated.
        //
        // DO NOT replicate this elsewhere unless you understand what you are
        // doing.
        LogicalResult replacementStatus =
            rewriter.notifyPayloadOperationReplaced(containingOp,
                                                    newContainingOp);
        (void)replacementStatus;
        assert(succeeded(replacementStatus) &&
               "unable to update transform state mapping");
        rewriter.eraseOp(containingOp);
        containingOp = newContainingOp;
      }
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }

    SmallVector<Operation *> tiledContainingOpOperand =
        bishengir::tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, producerOp, containingOp);
    if (!tiledContainingOpOperand.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused an extract use through block argument\n"
                        << *containingOp);
      fusedOps.append(tiledContainingOpOperand);
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }

    Operation *cloned = bishengir::cloneAndFuseFirstUse(
        rewriter, diag, producerOp, containingOp);
    if (cloned) {
      LLVM_DEBUG(DBGS() << "\nFused an use by cloning\n" << *containingOp);
      fusedOps.push_back(cloned);
      eraseRemainingProducer(producerOp, producerIndex);
      continue;
    }
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  results.set(cast<OpResult>(getFusedOp()[index]), fusedOps);
  results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::ExtendedFuseIntoContainingOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto containingOps = getContainingOp();
  for (auto it : llvm::enumerate(containingOps)) {
    auto containingOpPayloads = state.getPayloadOps(it.value());
    if (!llvm::hasSingleElement(containingOpPayloads)) {
      return emitDefiniteFailure()
             << "requires exactly one containing_op handle (got "
             << llvm::range_size(containingOpPayloads) << ")";
    }
    Operation *currentOp = *containingOpPayloads.begin();
    auto status =
        fuseIntoOneContaining(rewriter, results, state, it.index(), currentOp);
    if (!status.succeeded())
      return status;
  }
  return DiagnosedSilenceableFailure::success();
}

ParseResult ExtendedFuseIntoContainingOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  OpAsmParser::UnresolvedOperand producer;
  SmallVector<OpAsmParser::UnresolvedOperand> containingOps;
  FunctionType functionalType;
  llvm::SMLoc producerLoc;
  llvm::SMLoc containingOpsLoc;

  if (parser.getCurrentLocation(&producerLoc) || parser.parseOperand(producer))
    return ParseResult::failure();

  if (parser.parseKeyword("into"))
    return ParseResult::failure();

  if (parser.getCurrentLocation(&containingOpsLoc) ||
      parser.parseOperandList(containingOps))
    return ParseResult::failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();

  if (result.propertiesAttr) {
    NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
    attrs.append("resultSegmentSizes",
                 parser.getBuilder().getDenseI32ArrayAttr(
                     {static_cast<int32_t>(containingOps.size()),
                      static_cast<int32_t>(containingOps.size())}));
    result.propertiesAttr = attrs.getDictionary(parser.getContext());
  } else {
    result.addAttribute("resultSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(containingOps.size()),
                             static_cast<int32_t>(containingOps.size())}));
  }

  if (parser.parseColonType(functionalType))
    return ParseResult::failure();

  if (parser.resolveOperand(producer, functionalType.getInputs().front(),
                            result.operands) ||
      parser.resolveOperands(containingOps,
                             functionalType.getInputs().drop_front(),
                             containingOpsLoc, result.operands)) {
    return ParseResult::failure();
  }

  result.addTypes(functionalType.getResults());
  return ParseResult::success();
}

void ExtendedFuseIntoContainingOp::print(OpAsmPrinter &p) {
  p << ' ' << getProducerOp();
  p << ' ' << "into";
  p << ' ';
  p.printOperands(getContainingOp());
  p.printOptionalAttrDict((*this)->getAttrs(), {"resultSegmentSizes"});
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(), getResults().getTypes());
}

void transform::ExtendedFuseIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// SetBufferSizeOp
//===----------------------------------------------------------------------===//

struct SetBufferSizeResult {
  DiagnosedSilenceableFailure diag{DiagnosedSilenceableFailure::success()};
  int64_t bufferSizeInBytes;
};

SetBufferSizeResult
calculateBufferSize(int64_t bufferSize, SetBufferSizeMode unitMode,
                    Type elementType, std::optional<Type> referenceElementType,
                    Location loc) {
  SetBufferSizeResult result;
  result.bufferSizeInBytes = bufferSize;
  // Adjust size for element mode by multiplying byte size.
  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (unitMode == SetBufferSizeMode::kPerElement) {
    int perElementByte = static_cast<int>(
        llvm::divideCeil(elementBitWidth, mlir::utils::kBitsToByte));
    result.bufferSizeInBytes *= perElementByte;
  }
  if (!referenceElementType.has_value())
    return result;

  // Adjust size by reference type.
  if (!(*referenceElementType).isIntOrFloat()) {
    result.diag = emitDefiniteFailure(
        loc, "reference type must be an int or float type!");
    return result;
  }
  auto referenceTypeWidth = referenceElementType->getIntOrFloatBitWidth();
  if (referenceTypeWidth > elementBitWidth) {
    result.diag = emitDefiniteFailure(
        loc, "Reference type's bit width should be less than or equal to the "
             "current element type!");
    return result;
  }
  if (referenceTypeWidth == 0) {
    llvm::report_fatal_error("Reference type's width should be positive");
    result.diag =
        emitDefiniteFailure(loc, "reference type's width should be positive");
    return result;
  }
  auto factor = elementBitWidth / referenceTypeWidth;
  if (referenceTypeWidth == 0)
    llvm::report_fatal_error("Reference type's with should be positive");
  if (elementBitWidth % referenceTypeWidth != 0)
    factor = (elementBitWidth + referenceTypeWidth - 1) / referenceTypeWidth;
  result.bufferSizeInBytes *= static_cast<int>(factor);
  return result;
}

template <typename AllocOpTy>
void setBufferSizeForAllocLikeOp(AllocOpTy op, int64_t bufferSize,
                                 transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  MemRefType oldType = op.getType();
  // Create new alloc with static size.
  auto newType = MemRefType::get({bufferSize}, rewriter.getI8Type(),
                                 mlir::AffineMap{}, oldType.getMemorySpace());
  auto newAllocOp = rewriter.create<AllocOpTy>(loc, newType);
  // Create view from new alloc to old alloc's sizes and replace its use.
  auto startOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto viewOp = rewriter.create<memref::ViewOp>(
      loc, oldType, newAllocOp.getResult(), startOffset, op->getOperands());
  rewriter.replaceOp(op, viewOp);
}

void setBufferSizeForOpResult(Operation *op, int64_t resultNumber,
                              int64_t bufferSize,
                              transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto mark = rewriter.create<annotation::MarkOp>(op->getLoc(),
                                                  op->getResult(resultNumber));
  mark->setAttr(kBufferSizeInByteAttr, rewriter.getI64IntegerAttr(bufferSize));
}

DiagnosedSilenceableFailure
SetBufferSizeOp::apply(transform::TransformRewriter &rewriter,
                       transform::TransformResults &transformResults,
                       transform::TransformState &state) {
  auto staticBufferSizes = getStaticBufferSizes();
  if (getTarget().size() != staticBufferSizes.size())
    return emitDefiniteFailure(
        "Number of operands to set does not match buffer size count!");

  SetBufferSizeMode unitMode = getUnitMode();
  std::optional<Type> maybeReferenceType = getReferenceType();
  for (const auto &targetHandle : llvm::enumerate(getTarget())) {
    auto payloadOps = state.getPayloadOps(targetHandle.value());
    for (Operation *payloadOp : payloadOps) {
      auto staticBufferSize = staticBufferSizes[targetHandle.index()];
      if (staticBufferSize < 0)
        return emitDefiniteFailure("buffer size should be greater than 0!");

      for (OpResult result : payloadOp->getResults()) {
        auto maybeShapedType = dyn_cast<ShapedType>(result.getType());
        // If the op result is not a shaped type, or has static shape type, do
        // nothing.
        if (!maybeShapedType || maybeShapedType.hasStaticShape())
          continue;

        auto calculationResult = calculateBufferSize(
            staticBufferSize, unitMode,
            /*elementType=*/maybeShapedType.getElementType(),
            /*referenceElementType=*/maybeReferenceType, result.getLoc());
        if (!calculationResult.diag.succeeded())
          return std::move(calculationResult.diag);

        TypeSwitch<Operation *>(payloadOp)
            .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
              setBufferSizeForAllocLikeOp(
                  allocaOp, calculationResult.bufferSizeInBytes, rewriter);
            })
            .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
              setBufferSizeForAllocLikeOp(
                  allocOp, calculationResult.bufferSizeInBytes, rewriter);
            })
            .Default([&](Operation *) {
              setBufferSizeForOpResult(payloadOp, result.getResultNumber(),
                                       calculationResult.bufferSizeInBytes,
                                       rewriter);
            });
      }
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void SetBufferSizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// MultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MultiBufferOp::apply(transform::TransformRewriter &rewriter,
                     transform::TransformResults &transformResults,
                     transform::TransformState &state) {
  auto factor = getFactor();
  if (factor < 1) {
    emitError("factor should be >= 1.");
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  for (const auto &targetHandle : getTarget()) {
    auto payloadOps = state.getPayloadOps(targetHandle);
    for (Operation *definingOp : payloadOps) {
      assert(definingOp && "definingOp shouldn't be null.");
      if (!definingOp->getResults().empty()) {
        rewriter.setInsertionPointAfter(definingOp);
        for (auto res : definingOp->getResults()) {
          auto markOp =
              rewriter.create<annotation::MarkOp>(definingOp->getLoc(), res);
          markOp->setAttr(hfusion::MultiBufferAttr::name,
                          rewriter.getI32IntegerAttr(factor));
        }
      }
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void MultiBufferOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOpsEnums.cpp.inc"
#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.cpp.inc"
