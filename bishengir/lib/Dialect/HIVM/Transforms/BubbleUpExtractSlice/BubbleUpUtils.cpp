#include "bishengir/Dialect/HIVM/Transforms/BubbleUpExtractSlice/BubbleUpUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/TileUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::hivm::detail {

const llvm::StringLiteral kBubbleUpPropagateUp = "bubble_up_propagate_up";
const llvm::StringLiteral kBubbleUpPropagateDown = "bubble_up_propagate_down";


static UnrealizedConversionCastOp
createBubblePropagationCast(Value input, Type outputType,
                            StringRef directionAttrName,
                            PatternRewriter &rewriter) {
  auto castOp = rewriter.create<UnrealizedConversionCastOp>(input.getLoc(),
                                                            outputType, input);
  castOp->setAttr(directionAttrName, rewriter.getUnitAttr());
  return castOp;
}

/// propagate_up: old (full) value -> sliced type; consumed by the new halved op
/// created one level below in the chain (e.g. new to_tensor or new_cast
/// source).
UnrealizedConversionCastOp
createBubblePropagatorUpLink(Value oldValue, Type slicedType,
                             PatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(oldValue);
  auto propagateOp = createBubblePropagationCast(
      oldValue, slicedType, kBubbleUpPropagateUp, rewriter);
  return propagateOp;
}


std::optional<UnrealizedConversionCastOp>
createBubblePropagatorDown(Value oldValue, Value newValue,
                           ArrayRef<Operation *> excludedUsers,
                           PatternRewriter &rewriter) {
  SmallVector<OpOperand *> uses;
  for (auto &use : oldValue.getUses()) {
    Operation *owner = use.getOwner();
    if (llvm::is_contained(excludedUsers, owner))
      continue;
    if (owner->hasAttr(kBubbleUpPropagateUp) ||
        owner->hasAttr(kBubbleUpPropagateDown))
      continue;
    uses.push_back(&use);
  }
  if (uses.empty())
    return std::nullopt;

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(newValue);
  auto propagateOp = createBubblePropagationCast(
      newValue, oldValue.getType(), kBubbleUpPropagateDown, rewriter);
  for (OpOperand *use : uses)
    use->set(propagateOp.getResult(0));
  return propagateOp;
}

std::optional<UnrealizedConversionCastOp>
createBubblePropagatorDown(Value value, ArrayRef<Operation *> excludedUsers,
                           PatternRewriter &rewriter) {
  return createBubblePropagatorDown(value, value, excludedUsers, rewriter);
}


static void resolveUpPropagator(UnrealizedConversionCastOp upPropagator,
                                Value newValue, PatternRewriter &rewriter) {
  if (upPropagator->use_empty())
    return;
  rewriter.replaceAllUsesWith(upPropagator.getResult(0), newValue);
}


/// Collect path from to_tensor memref toward alloc (bottom-first order).
static LogicalResult
collectBufferizationPath(bufferization::ToTensorOp toTensorOp,
                         SmallVectorImpl<Operation *> &pathOps,
                         memref::AllocOp &allocOp) {
  Value memref = toTensorOp.getMemref();
  llvm::SmallDenseSet<Value, 4> visited;
  while (memref) {
    if (!visited.insert(memref).second)
      return failure();
    if (auto foundAlloc = memref.getDefiningOp<memref::AllocOp>()) {
      allocOp = foundAlloc;
      return success();
    }
    if (auto castOp = memref.getDefiningOp<memref::MemorySpaceCastOp>()) {
      if (utils::getAnnotateOpWithAttr(castOp.getResult(),
                                       kMayImplicitTransposeWithLastAxis))
        return failure();
      pathOps.push_back(castOp.getOperation());
      memref = castOp.getSource();
      continue;
    }
    if (auto subViewOp = memref.getDefiningOp<memref::SubViewOp>()) {
      pathOps.push_back(subViewOp.getOperation());
      memref = subViewOp.getSource();
      continue;
    }
    return failure();
  }
  return failure();
}

void clearBubblePropagatorAttrs(Operation *op, PatternRewriter &rewriter) {
  rewriter.modifyOpInPlace(op, [&]() {
    op->removeAttr(kBubbleUpPropagateUp);
    op->removeAttr(kBubbleUpPropagateDown);
  });
}

void resolveUpLinksForOldValue(Value oldValue, Value newValue,
                               PatternRewriter &rewriter) {
  SmallVector<UnrealizedConversionCastOp> upPropagators;
  for (Operation *user : oldValue.getUsers()) {
    auto upPropagator = dyn_cast<UnrealizedConversionCastOp>(user);
    if (!upPropagator || !upPropagator->hasAttr(kBubbleUpPropagateUp))
      continue;
    if (upPropagator.getInputs()[0] != oldValue)
      continue;
    upPropagators.push_back(upPropagator);
  }

  for (UnrealizedConversionCastOp upPropagator : upPropagators) {
    resolveUpPropagator(upPropagator, newValue, rewriter);
    if (upPropagator->use_empty())
      clearBubblePropagatorAttrs(upPropagator, rewriter);
  }
}

static void cleanupResolvedBufferizationPropagatorsImpl(
    func::FuncOp funcOp, RewriterBase &rewriter) {
  bool progress = true;
  while (progress) {
    progress = false;
    SmallVector<Operation *> toErase;
    funcOp.walk([&](UnrealizedConversionCastOp ucc) {
      if (ucc->hasAttr(kBubbleUpPropagateUp) ||
          ucc->hasAttr(kBubbleUpPropagateDown))
        return;
      if (ucc->use_empty())
        toErase.push_back(ucc.getOperation());
    });
    funcOp.walk([&](Operation *op) {
      if (!op->use_empty())
        return;
      if (isa<memref::AllocOp, memref::MemorySpaceCastOp, memref::SubViewOp,
              bufferization::ToTensorOp>(op))
        toErase.push_back(op);
    });
    llvm::sort(toErase);
    toErase.erase(std::unique(toErase.begin(), toErase.end()), toErase.end());
    for (Operation *op : toErase) {
      if (op->getBlock() && op->use_empty()) {
        rewriter.eraseOp(op);
        progress = true;
      }
    }
  }
}

void cleanupResolvedBufferizationPropagators(func::FuncOp funcOp,
                                             PatternRewriter &rewriter) {
  cleanupResolvedBufferizationPropagatorsImpl(funcOp, rewriter);
}

void cleanupResolvedBufferizationPropagators(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  if (!funcOp.getBody().empty())
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  cleanupResolvedBufferizationPropagatorsImpl(funcOp, rewriter);
}

MemRefType getSlicedMemRefType(MemRefType oldType,
                               RankedTensorType slicedTensorType) {
  return MemRefType::get(slicedTensorType.getShape(),
                         slicedTensorType.getElementType(), oldType.getLayout(),
                         oldType.getMemorySpace());
}

void markTiledTightlyCoupledAllocIfNeeded(RewriterBase &rewriter,
                                          Value memrefValue) {
  auto maybeAlloc = mlir::utils::tracebackMemRefToAlloc(memrefValue);
  if (!maybeAlloc)
    return;
  Value allocResult = maybeAlloc->getResult();
  auto maybeMark = mlir::utils::getAnnotateOpWithAttr(
      allocResult, hivm::HIVMTightlyCoupledBufferAttr::name);
  if (!maybeMark.has_value())
    return;
  auto markOp = dyn_cast<annotation::MarkOp>(maybeMark.value());
  if (!markOp)
    return;
  auto attr = markOp->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
      hivm::HIVMTightlyCoupledBufferAttr::name);
  if (!attr || !attr.getId().has_value())
    return;
  rewriter.modifyOpInPlace(markOp, [&]() {
    markOp->setAttr(kTiledTightlyCoupledAlloc,
                    UnitAttr::get(rewriter.getContext()));
  });
}

LogicalResult
checkBufferizationBubbleUpPath(bufferization::ToTensorOp toTensorOp) {
  SmallVector<Operation *, 4> pathOps;
  memref::AllocOp allocOp;
  return collectBufferizationPath(toTensorOp, pathOps, allocOp);
}


FailureOr<memref::AllocOp>
createSlicedAllocLike(MemRefType slicedMemRefType, memref::AllocOp oldAllocOp,
                      PatternRewriter &rewriter) {
  auto originalType = dyn_cast<MemRefType>(oldAllocOp.getResult().getType());
  if (!originalType)
    return failure();

  auto memrefType = MemRefType::get(
      slicedMemRefType.getShape(), slicedMemRefType.getElementType(),
      originalType.getLayout(), originalType.getMemorySpace());
  if (!memrefType.hasStaticShape())
    return failure();

  rewriter.setInsertionPoint(oldAllocOp);
  return rewriter.create<memref::AllocOp>(oldAllocOp.getLoc(), memrefType);
}

void cleanupBufferizationPropagators(func::FuncOp funcOp,
                                     bufferization::ToTensorOp toTensorOp,
                                     BufferizationPropagationState &state,
                                     PatternRewriter &rewriter) {
  SmallVector<Operation *> toErase;
  auto scheduleErase = [&](Operation *op) {
    if (op && op->getBlock())
      toErase.push_back(op);
  };

  if (state.newAllocOp) {
    Value newAlloc = state.newAllocOp.getResult();
    SmallVector<OpOperand *> allocUses;
    for (auto &use : state.allocOp.getResult().getUses())
      allocUses.push_back(&use);
    for (OpOperand *use : allocUses) {
      if (auto markOp = dyn_cast<annotation::MarkOp>(use->getOwner())) {
        rewriter.modifyOpInPlace(markOp, [&]() { use->set(newAlloc); });
        continue;
      }
      if (auto ucc = dyn_cast<UnrealizedConversionCastOp>(use->getOwner())) {
        if (ucc->hasAttr(kBubbleUpPropagateUp) ||
            ucc->hasAttr(kBubbleUpPropagateDown))
          rewriter.replaceAllUsesWith(ucc.getResult(0), newAlloc);
      }
    }
  }

  scheduleErase(toTensorOp);
  for (Operation *op : state.pathOps)
    scheduleErase(op);
  scheduleErase(state.allocOp);

  funcOp.walk([&](UnrealizedConversionCastOp ucc) {
    if (!ucc->hasAttr(kBubbleUpPropagateUp) &&
        !ucc->hasAttr(kBubbleUpPropagateDown))
      return;
    if (ucc->use_empty())
      scheduleErase(ucc);
  });

  llvm::sort(toErase);
  toErase.erase(std::unique(toErase.begin(), toErase.end()), toErase.end());
  for (Operation *op : toErase) {
    if (op->getBlock() && op->use_empty())
      rewriter.eraseOp(op);
  }
}

} // namespace mlir::hivm::detail
