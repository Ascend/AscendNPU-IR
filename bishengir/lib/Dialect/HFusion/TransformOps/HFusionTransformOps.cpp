//===- HFusionTransformOps.cpp - Implementation of HFusion transform ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/Dialect/Transform/IR/TransformOps.h"
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
// GetFuncArgumentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncArgumentOp::apply(TransformRewriter &rewriter,
                         TransformResults &transformResults,
                         TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  Region::BlockArgListType funcArgs = func.getArguments();
  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumArguments(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedArgs = llvm::map_to_vector(
      operandPositions, [&](int64_t pos) { return Value(funcArgs[pos]); });
  if (getFindReshapeConsumer()) {
    for (auto [idx, v] : llvm::enumerate(selectedArgs)) {
      auto maybeResult = hfusion::traceReshapeOrSliceSingleConsumer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape consumer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()), selectedArgs);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncArgumentOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// GetFuncResultOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncResultOp::apply(TransformRewriter &rewriter,
                       TransformResults &transformResults,
                       TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  func::ReturnOp returnOp = nullptr;
  func->walk([&returnOp](func::ReturnOp op) { returnOp = op; });
  if (!returnOp)
    return emitDefiniteFailure() << "cannot find return op in func!";

  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumResults(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedResult =
      llvm::map_to_vector(operandPositions, [&](int64_t pos) {
        return returnOp->getOpOperand(pos).get();
      });
  if (getFindReshapeProducer()) {
    for (auto [idx, v] : llvm::enumerate(selectedResult)) {
      auto maybeResult = hfusion::traceReshapeOrSliceSingleProducer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape producer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()),
                             selectedResult);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncResultOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

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
      llvm_unreachable("unsupported type");
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
      llvm_unreachable("unsupported type");
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
// ReverseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure ReverseOp::apply(TransformRewriter &rewriter,
                                             TransformResults &transformResults,
                                             TransformState &state) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<Operation *> reversedOperations = {targets.rbegin(),
                                                 targets.rend()};
  transformResults.set(cast<OpResult>(getResult()), reversedOperations);
  return DiagnosedSilenceableFailure::success();
}

void ReverseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
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
    llvm_unreachable("Reference type's width should be positive");
    result.diag = emitDefiniteFailure(
        loc, "reference type's width should be positive");
    return result;
  }
  auto factor = elementBitWidth / referenceTypeWidth;
  if (referenceTypeWidth == 0)
    llvm_unreachable("Reference type's with should be positive");
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

//===---------------------------------------------------------------------===//
// MatchAncestorOfOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchAncestorOfOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps)) {
    return emitDefiniteFailure("requires exactly one target handle");
  }

  auto childOps = state.getPayloadOps(getChild());
  if (!llvm::hasSingleElement(childOps)) {
    return emitDefiniteFailure("requires exactly one child handle");
  }
  Operation *childOp = *childOps.begin();
  // Build dominance info from enclosing function
  func::FuncOp enclosingFunc = childOp->getParentOfType<func::FuncOp>();
  DominanceInfo domInfo(enclosingFunc);

  SmallVector<Operation *> res;
  bool incorrectNumOperandTypes = false;
  auto matchFun = [&](Operation *op) {
    if (getOps().has_value() && !strs.contains(op->getName().getStringRef()))
      return;

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().has_value()) {
      auto iface = getInterface().value();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<linalg::LinalgOp>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          !isa<TilingInterface>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::LoopLikeInterface &&
          !isa<LoopLikeOpInterface>(op))
        return;
    }

    // Check if all specified attributes match.
    if (getOpAttrs().has_value()) {
      DictionaryAttr opAttrs = getOpAttrs().value();
      for (NamedAttribute attr : opAttrs) {
        if (attr.getName() == getInterfaceAttrName() ||
            attr.getName() == getOpsAttrName())
          continue;
        if (!op->hasAttr(attr.getName()))
          return;
        if (op->getAttr(attr.getName()) != attr.getValue())
          return;
      }
    }

    // Check if at least one of the optional attributes match.
    if (getOptionalOpAttrs().has_value() &&
        !getOptionalOpAttrs().value().empty()) {
      DictionaryAttr optionalOpAttrs = getOptionalOpAttrs().value();
      if (llvm::none_of(optionalOpAttrs, [&](NamedAttribute attr) {
            if (!op->hasAttr(attr.getName()))
              return false;

            if (op->getAttr(attr.getName()) != attr.getValue())
              return false;

            return true;
          }))
        return;
    }

    if (getFilterResultType().has_value()) {
      Type t = getFilterResultType().value();
      if (op->getNumResults() != 1 || op->getResultTypes().front() != t)
        return;
    }

    if (getFilterOperandTypes().has_value()) {
      ArrayAttr types = getFilterOperandTypes().value();
      auto operandTypes = op->getOperandTypes();

      if (types.size() == 1) {
        // All the operands must be equal to the specified type
        auto typeattr = dyn_cast<TypeAttr>(getFilterOperandTypes().value()[0]);
        auto t = cast<Type>(typeattr.getValue());
        if (!llvm::all_of(op->getOperandTypes(),
                          [&](Type operandType) { return operandType == t; }))
          return;
      } else {
        // The operand types must match all the types in the list (in the same
        // order in with they are specified)
        if (types.size() != operandTypes.size()) {
          incorrectNumOperandTypes = true;
          return;
        }

        for (auto [attr, operandType] :
             llvm::zip_equal(getFilterOperandTypes().value(), operandTypes)) {
          auto typeattr = cast<TypeAttr>(attr);
          auto type = cast<Type>(typeattr.getValue());
          if (type != operandType)
            return;
        }
      }
    }

    if (!domInfo.properlyDominates(op, childOp))
      return;

    // All constraints are satisfied.
    res.push_back(op);
    return;
  };

  (*payloadOps.begin())->walk(matchFun);
  if (incorrectNumOperandTypes)
    return emitDefiniteFailure("If filter_operand_types contains more than a "
                               "type, then it must contain as much types as "
                               "the number of operands in the target ops");
  results.set(cast<OpResult>(getResult()), res);
  return DiagnosedSilenceableFailure::success();
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result,
                                         TypeRange resultTypes, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(resultTypes);
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child, ArrayAttr ops,
                                         DictionaryAttr op_attrs,
                                         DictionaryAttr optional_op_attrs) {
  result.addOperands(target);
  result.addOperands(child);
  if (ops)
    result.addAttribute(MatchAncestorOfOp::getOpsAttrName(result.name), ops);
  if (op_attrs)
    result.addAttribute(MatchAncestorOfOp::getOpAttrsAttrName(result.name),
                        op_attrs);
  if (optional_op_attrs)
    result.addAttribute(
        MatchAncestorOfOp::getOptionalOpAttrsAttrName(result.name),
        optional_op_attrs);
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child, ArrayAttr ops,
                                         DictionaryAttr op_attrs) {
  result.addOperands(target);
  result.addOperands(child);
  if (ops)
    result.addAttribute(MatchAncestorOfOp::getOpsAttrName(result.name), ops);
  if (op_attrs)
    result.addAttribute(MatchAncestorOfOp::getOpAttrsAttrName(result.name),
                        op_attrs);
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

//===----------------------------------------------------------------------===//
// ExtendedLoopOutlineOp
//===----------------------------------------------------------------------===//

namespace {

inline SmallVector<scf::ForOp> collectLoops(const SmallVector<Value> &targets,
                                            transform::TransformState &state) {
  SmallVector<scf::ForOp> loops;
  DominanceInfo domInfo;
  for (Value target : targets) {
    if (state.getPayloadOps(target).empty()) {
      assert(false && "payload op is empty.");
    }
    Operation *loop = *state.getPayloadOps(target).begin();
    assert(llvm::hasSingleElement(state.getPayloadOps(target)) &&
           "expect single element.");
    loops.push_back(dyn_cast<scf::ForOp>(loop));
  }
  llvm::sort(loops, [&domInfo](scf::ForOp a, scf::ForOp b) {
    return domInfo.properlyDominates(a, b, false);
  });
  return loops;
}

inline void getResultsUsedBelow(const SmallVector<Operation *> &ops,
                                Operation *below, SmallVector<Value> &results) {
  DominanceInfo domInfo;
  for (Operation *op : ops) {
    for (OpOperand &use : op->getUses()) {
      if (domInfo.properlyDominates(below, use.getOwner(), false)) {
        results.push_back(use.get());
        break;
      }
    }
  }
}

inline void duplicateReUsedIterArgs(const SmallVector<scf::ForOp> &loops,
                                    transform::TransformRewriter &rewriter) {
  DenseSet<Value> set;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *loop : loops) {
    rewriter.setInsertionPoint(loop);
    for (OpOperand &iterArg : dyn_cast<scf::ForOp>(loop).getInitArgsMutable()) {
      auto value = iterArg.get();
      Operation *defOp = value.getDefiningOp();
      if (!set.contains(value)) {
        set.insert(value);
      } else {
        assert(defOp != nullptr);
        Operation *newOp = rewriter.clone(*defOp);
        iterArg.assign(newOp->getResult(0));
      }
    }
  }
}

inline SmallVector<Operation *>
collectAllDefiningOpsOfForOp(scf::ForOp consumer) {
  SmallVector<Operation *> definingOps;
  for (Value val : consumer.getInitArgs()) {
    if (val.getDefiningOp())
      definingOps.push_back(val.getDefiningOp());
  }
  for (Value val : consumer.getOperands()) {
    if (val.getDefiningOp())
      definingOps.push_back(val.getDefiningOp());
  }
  visitUsedValuesDefinedAbove(consumer->getRegion(0), consumer->getRegion(0),
                              [&definingOps](OpOperand *operand) {
                                definingOps.push_back(
                                    operand->get().getDefiningOp());
                              });
  return definingOps;
}

// For pair of loops: {producer_loop, consumer_loop},
// 1. recursivly collect users of producer_loop until meet consumer_loop,
//    into prevUsers Set.
// 2. recursivly collect consumer_loop definingOp, until meet producer_loop,
//    if exists in prevUsers, push into intermediateOps vector.
// 3. sort them with dominance order.
inline void
collectAndGroupProducerUsers(scf::ForOp producer, scf::ForOp consumer,
                             SmallVector<Operation *> &intermediateOps,
                             SmallVector<Operation *> &nonIntermediateOps) {
  DominanceInfo domInfo;
  if (!domInfo.properlyDominates(producer, consumer, false)) {
    return;
  }

  SmallVector<Operation *> stack = {producer};
  DenseSet<Operation *> users;
  DenseSet<Operation *> definingOps;

  while (!stack.empty()) {
    Operation *cur = stack.pop_back_val();
    for (Operation *curUser : cur->getUsers()) {
      if (users.find(curUser) == users.end() &&
          domInfo.properlyDominates(curUser, consumer, false)) {
        users.insert(curUser);
        stack.push_back(curUser);
      }
    }
  }

  stack.append(collectAllDefiningOpsOfForOp(consumer));

  while (!stack.empty()) {
    Operation *cur = stack.pop_back_val();
    if (users.find(cur) != users.end()) {
      definingOps.insert(cur);
      for (Value operand : cur->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          stack.push_back(defOp);
        }
      }
    }
  }

  intermediateOps = {definingOps.begin(), definingOps.end()};
  llvm::stable_sort(intermediateOps, [&domInfo](Operation *a, Operation *b) {
    return domInfo.properlyDominates(a, b, false);
  });

  for (Operation *item : users) {
    if (definingOps.find(item) == definingOps.end()) {
      nonIntermediateOps.push_back(item);
    }
  }
  llvm::stable_sort(nonIntermediateOps, [&domInfo](Operation *a, Operation *b) {
    return domInfo.properlyDominates(a, b, false);
  });
}

inline void
collectAndGroupOperations(const SmallVector<scf::ForOp> &loops,
                          transform::TransformState &state,
                          SmallVector<Operation *> &opsToOutline,
                          SmallVector<Operation *> &opsToAdjustPositon) {
  scf::ForOp prev = nullptr;
  for (auto cur : loops) {
    if (prev) {
      SmallVector<Operation *> intermediateOps;
      collectAndGroupProducerUsers(prev, cur, intermediateOps,
                                   opsToAdjustPositon);
      opsToOutline.append(intermediateOps);
    }

    opsToOutline.push_back(cur);
    prev = cur;
  }
}

inline void adjustOperationPositionsAfter(
    const SmallVector<Operation *> &opsToAdjustPositon, Operation *after,
    transform::TransformRewriter &rewriter) {
  DominanceInfo domInfo;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *op : llvm::reverse(opsToAdjustPositon)) {
    rewriter.setInsertionPointAfter(after);
    Operation *newOp = rewriter.clone(*op);
    bool _;
    rewriter.replaceUsesWithIf(
        op->getResults(), newOp->getResults(),
        [&after, &domInfo](OpOperand &operand) -> bool {
          return domInfo.properlyDominates(after, operand.getOwner(), false);
        },
        &_);
    rewriter.eraseOp(op);
  }
}

inline void initAndOutlineOpsIntoRegion(
    const SmallVector<Operation *> &ops, const SmallVector<Value> &origResults,
    const SmallVector<Type> &yieldResultsTypes,
    SmallVector<Value> &yieldResults, scf::ExecuteRegionOp &executeRegionOp,
    Operation *&symbolTableOp, transform::TransformRewriter &rewriter) {
  IRMapping mapper;
  OpBuilder::InsertionGuard g(rewriter);
  for (auto riter = ops.rbegin(); riter != ops.rend(); riter++) {
    Operation *op = *riter;
    if (!executeRegionOp) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(op);
      executeRegionOp = rewriter.create<scf::ExecuteRegionOp>(
          op->getLoc(), yieldResultsTypes);
      symbolTableOp = SymbolTable::getNearestSymbolTable(op);
      executeRegionOp.getRegion().emplaceBlock();
    }

    rewriter.setInsertionPointToStart(&executeRegionOp.getRegion().back());

    Operation *clonedOp = nullptr;
    if (isa<scf::ForOp>(op)) {
      clonedOp = rewriter.cloneWithoutRegions(*op);
      Region &clonedRegion = clonedOp->getRegions().front();
      assert(clonedRegion.empty() && "expected empty region");

      rewriter.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                                  clonedRegion.end());
    } else {
      clonedOp = rewriter.clone(*op);
    }

    mapper.map(op->getResults(), clonedOp->getResults());

    bool _;
    rewriter.replaceOpUsesWithIf(
        op, clonedOp->getResults(),
        [&](OpOperand &use) {
          return executeRegionOp->isAncestor(use.getOwner());
        },
        &_);
  }

  for (auto item : origResults) {
    yieldResults.push_back(mapper.lookup(item));
  }

  rewriter.setInsertionPointToEnd(&executeRegionOp.getRegion().back());
  rewriter.create<scf::YieldOp>(executeRegionOp.getLoc(), yieldResults);

  rewriter.replaceAllUsesWith(origResults, executeRegionOp->getResults());
}

} // namespace

// Extract all result types for return.
DiagnosedSilenceableFailure
transform::ExtendedLoopOutlineOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  SmallVector<Operation *> functions;
  SmallVector<Operation *> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;

  SmallVector<Value> targets = getTargets();
  SmallVector<scf::ForOp> loops = collectLoops(targets, state);

  duplicateReUsedIterArgs(loops, rewriter);

  SmallVector<Operation *> ops;
  SmallVector<Operation *> opsToAdjustPositon;
  collectAndGroupOperations(loops, state, ops, opsToAdjustPositon);
  adjustOperationPositionsAfter(opsToAdjustPositon, loops.back(), rewriter);

  scf::ExecuteRegionOp executeRegionOp = nullptr;
  Operation *symbolTableOp = nullptr;

  SmallVector<Type> yieldResultsTypes;
  SmallVector<Value> yieldResults;
  SmallVector<Value> origResults;

  getResultsUsedBelow(ops, loops.back(), origResults);
  for (Value origResult : origResults) {
    yieldResultsTypes.push_back(origResult.getType());
  }

  initAndOutlineOpsIntoRegion(ops, origResults, yieldResultsTypes, yieldResults,
                              executeRegionOp, symbolTableOp, rewriter);

  for (Operation *op : llvm::reverse(ops)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }

  if (!executeRegionOp) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to outline";
    return diag;
  }

  func::CallOp call;
  FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(
      rewriter, executeRegionOp->getLoc(), executeRegionOp.getRegion(),
      getFuncName(), &call);

  if (failed(outlined)) {
    return emitDefaultDefiniteFailure(executeRegionOp);
  }

  if (symbolTableOp) {
    SymbolTable &symbolTable =
        symbolTables.try_emplace(symbolTableOp, symbolTableOp)
            .first->getSecond();
    symbolTable.insert(*outlined);
    call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
  }

  functions.push_back(*outlined);
  calls.push_back(call);
  results.set(cast<OpResult>(getFunction()), functions);
  results.set(cast<OpResult>(getCall()), calls);

  return DiagnosedSilenceableFailure::success();
}

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOpsEnums.cpp.inc"
#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.cpp.inc"
