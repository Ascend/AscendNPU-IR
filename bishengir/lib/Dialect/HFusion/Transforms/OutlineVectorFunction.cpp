//===--------------------- OutlineVectorFunction.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/Transform/IR/TransformOps.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
#define GEN_PASS_DEF_OUTLINEVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "outline-vector-function"

using namespace mlir;
using namespace mlir::hfusion;

namespace {
struct OutlineVectorFunctionPass
    : public impl::OutlineVectorFunctionBase<OutlineVectorFunctionPass> {
public:
  void runOnOperation() override;
};

static Value getOpTransformHandle(std::string label, OpBuilder &builder,
                                  transform::SequenceOp seqOp) {
  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value opHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();
  return opHandle;
}

static bool loopHasOutlinedLoopAttr(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("outlined-loop-target-")) {
      return true;
    }
  }
  return false;
}

/// `isFill` marks a vector function outlined around a fill, so that a later
/// pass can look for a constant to forward to its readers without having to
/// hunt for fills by inspecting every function body. Whether the fill really
/// does stamp a compile-time constant is for that pass to establish.
static void outlineLoopsIntoVF(SmallVector<Value> &loopHandles, Location loc,
                               OpBuilder &builder, std::string vfName,
                               bool isFill = false) {
  if (loopHandles.size() == 0)
    return;
  auto outlineOp = builder.create<transform::ExtendedLoopOutlineOp>(
      loc,
      SmallVector<Type>{builder.getType<transform::AnyOpType>(),
                        builder.getType<transform::AnyOpType>()},
      loopHandles, vfName);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        mlir::hivm::VectorFunctionAttr::name,
                                        nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                        "no_inline", nullptr);
  if (isFill)
    builder.create<transform::AnnotateOp>(loc, outlineOp.getFunction(),
                                          kFillVFAttrName, nullptr);
  builder.create<transform::AnnotateOp>(
      loc, outlineOp.getCall(), mlir::hivm::VectorFunctionAttr::name, nullptr);
  builder.create<transform::AnnotateOp>(loc, outlineOp.getCall(), "no_inline",
                                        nullptr);
  loopHandles.clear();
}

static void removeLabelsForOutlinedLoops(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().contains("outlined")) {
      op->removeAttr(attr.getName().strref());
    }
  }
}

static void duplicateReusedValuesForInnerSCFForOp(scf::ForOp loop,
                                                  OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  DominanceInfo domInfo;
  auto loc = loop.getLoc();
  DenseSet<Value> set;
  loop.getBody()->walk([&](scf::ForOp innerLoop) {
    for (OpOperand &iterArg : innerLoop.getInitArgsMutable()) {
      Value src = iterArg.get();
      if (!src.getDefiningOp())
        continue;
      auto srcExtractOp = dyn_cast<tensor::ExtractSliceOp>(src.getDefiningOp());
      if (srcExtractOp)
        src = srcExtractOp.getSource();
      if (!src.getDefiningOp() || !domInfo.properlyDominates(src, loop))
        continue;
      if (!set.contains(src)) {
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        if (!ty || !ty.hasStaticShape())
          continue;
        set.insert(src);
      } else {
        builder.setInsertionPoint(loop);
        auto ty = dyn_cast<RankedTensorType>(src.getType());
        Value duplicated = builder.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                           ty.getElementType());
        if (!isEmptyLikeTensor(src)) {
          duplicated =
              builder.create<linalg::CopyOp>(loc, src, duplicated).getResult(0);
        }
        if (srcExtractOp) {
          builder.setInsertionPoint(srcExtractOp);
          auto newSrcExtractOp = builder.create<tensor::ExtractSliceOp>(
              loc, srcExtractOp.getType(), duplicated,
              srcExtractOp.getMixedOffsets(), srcExtractOp.getMixedSizes(),
              srcExtractOp.getMixedStrides());
          iterArg.assign(newSrcExtractOp.getResult());
        } else {
          iterArg.assign(duplicated);
        }
      }
    }
  });
}

/// The ccec VF launch parameter block packs the trip count of each outlined
/// vector loop into a 16-bit field (INT_HIVM_INSERT_REG ..., 0, 15). Split such
/// loops into consecutive chunks of at most kMaxVFLoopTrip iterations; each
/// chunk stays in the same VF, with a representable hardware loop count.
static constexpr int64_t kMaxVFLoopTrip = 65535;

/// Per-source-loop code size budget. Splitting clones the body; cap expansion
/// independently of the VF's total loop count or later buffer allocation.
static constexpr int64_t kMaxChunks = 32;

/// Split a constant-bound, signed-comparison loop into sequential loops of at
/// most maxTripCount iterations. The caller must prove lb/ub/step match the
/// loop's operands. Bounds and step are materialized in the original type;
/// absolute indices, attributes and loop-carried values are preserved.
/// Small or empty loops are left unchanged. Exceeding maxChunks emits a
/// diagnostic and fails without changing IR. Splitting erases forOp and its
/// nested operations. The caller filters unsigned loops and nonpositive steps.
static LogicalResult splitForOpByTripCount(OpBuilder &builder, scf::ForOp forOp,
                                           int64_t lb, int64_t ub, int64_t step,
                                           uint64_t maxTripCount,
                                           uint64_t maxChunks) {
  assert(maxTripCount && maxChunks && step > 0 &&
         "expected positive splitting limits and step");
  // Widen before subtracting or multiplying: signed 64-bit bounds can span
  // more than INT64_MAX iterations even when each individual bound fits.
  llvm::APInt lower(128, lb, true), upper(128, ub, true);
  llvm::APInt stride(128, step, true);
  if (upper.sle(lower))
    return success();
  const uint64_t trip =
      (upper - lower + stride - 1).udiv(stride).getZExtValue();
  if (trip <= maxTripCount)
    return success();
  const uint64_t numChunks = trip / maxTripCount + (trip % maxTripCount != 0);
  if (numChunks > maxChunks) {
    forOp.emitError("loop with trip count ")
        << trip << " needs " << numChunks << " chunks (limit " << maxChunks
        << "); refusing to exceed the loop splitting budget";
    return failure();
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  scf::YieldOp origYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  // Chunk bounds keep the source loop's bound type: scf.for requires
  // lowerBound/upperBound/step to share one type (AllTypesMatch), and the
  // cloned body consumes the original bound type. Interior chunk bounds are
  // strictly below ub, so every value fits the source type's range.
  Type boundType = forOp.getLowerBound().getType();
  auto createBound = [&](int64_t value) -> Value {
    if (isa<IndexType>(boundType))
      return builder.create<arith::ConstantIndexOp>(loc, value);
    return builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(boundType, value));
  };

  SmallVector<Value> inits(forOp.getInitArgs());
  scf::ForOp lastChunk = nullptr;
  // Materialize the original upper-bound value for the last chunk. Rounding
  // it up to lb + trip * step could expose an out-of-range bound to later
  // analyses.
  SmallVector<std::pair<Value, Value>> chunkBounds;
  for (uint64_t loIt = 0; loIt < trip;) {
    const uint64_t hiIt = loIt + std::min(maxTripCount, trip - loIt);
    const int64_t lo = (lower + llvm::APInt(128, loIt) * stride).getSExtValue();
    const int64_t hi =
        hiIt == trip ? ub
                     : (lower + llvm::APInt(128, hiIt) * stride).getSExtValue();
    chunkBounds.emplace_back(createBound(lo), createBound(hi));
    loIt = hiIt;
  }
  Value constantStep = createBound(step);
  for (auto &[loVal, hiVal] : chunkBounds) {
    scf::ForOp chunk = builder.create<scf::ForOp>(
        loc, loVal, hiVal, constantStep, inits,
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          IRMapping mapping;
          mapping.map(forOp.getStep(), constantStep);
          mapping.map(forOp.getInductionVar(), iv);
          for (auto [oldArg, newArg] :
               llvm::zip(forOp.getRegionIterArgs(), iterArgs))
            mapping.map(oldArg, newArg);
          for (Operation &op : forOp.getBody()->without_terminator())
            b.clone(op, mapping);
          SmallVector<Value> mappedYields;
          for (Value v : origYield.getOperands())
            mappedYields.push_back(mapping.lookupOrDefault(v));
          b.create<scf::YieldOp>(bodyLoc, mappedYields);
        });
    for (NamedAttribute attr : forOp->getAttrs())
      chunk->setAttr(attr.getName(), attr.getValue());
    inits = SmallVector<Value>(chunk.getResults());
    lastChunk = chunk;
  }
  assert(lastChunk && "large trip count guarantees at least one chunk");
  for (auto [oldRes, newRes] :
       llvm::zip(forOp.getResults(), lastChunk->getResults()))
    oldRes.replaceAllUsesWith(newRes);
  forOp.erase();
  return success();
}

/// Record the unique direct call to each VF. A null entry means multiple
/// call sites; missing entries have no direct calls. Neither is handled here.
static DenseMap<Operation *, func::CallOp> collectVFCalls(ModuleOp module) {
  DenseMap<Operation *, func::CallOp> calls;
  SymbolTableCollection symbols;
  module.walk([&](func::CallOp call) {
    auto func = symbols.lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!func || func.isDeclaration() || !hivm::isVF(func))
      return;
    auto [it, inserted] = calls.try_emplace(func, call);
    if (!inserted)
      it->second = nullptr;
  });
  return calls;
}

static LogicalResult splitLargeTripOutlinedLoops(func::FuncOp func,
                                                 func::CallOp call) {
  DenseMap<Value, int64_t> constants;
  for (auto [arg, operand] : llvm::zip(func.getArguments(), call.getOperands()))
    if (auto constant = getConstantIntValue(operand))
      constants[arg] = *constant;
  auto resolve = [&](Value value) -> std::optional<int64_t> {
    if (auto constant = getConstantIntValue(value))
      return constant;
    auto it = constants.find(value);
    if (it != constants.end())
      return it->second;
    return std::nullopt;
  };
  // Post-order processes inner loops before cloning their parent. No nested
  // operation pointers survive a parent rewrite and no fixpoint scan is needed.
  SmallVector<scf::ForOp> loops;
  func.walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  OpBuilder builder(func.getContext());
  for (scf::ForOp loop : loops) {
    auto lb = resolve(loop.getLowerBound());
    auto ub = resolve(loop.getUpperBound());
    auto step = resolve(loop.getStep());
    if (!lb || !ub || !step) {
      loop.emitRemark("dynamic VF loop count was not checked against the ")
          << "16-bit launch limit; runtime chunking may be required";
      continue;
    }
    if (loop->hasAttr("unsignedCmp")) {
      loop.emitRemark("unsigned VF loop count legalization is unsupported");
      continue;
    }
    if (*step <= 0) {
      loop.emitRemark("non-positive VF loop step; ")
          << "16-bit launch count was not checked";
      continue;
    }
    if (failed(splitForOpByTripCount(builder, loop, *lb, *ub, *step,
                                     kMaxVFLoopTrip, kMaxChunks)))
      return failure();
  }
  return success();
}

void OutlineVectorFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());
  module.walk([&](func::FuncOp func) {
    func.walk([&](scf::ForOp forOp) {
      if (!loopHasOutlinedLoopAttr(forOp))
        return;
      duplicateReusedValuesForInnerSCFForOp(forOp, builder);
    });
    std::string vfNamePrefix = func.getSymName().str() + "_outlined_vf_";
    int vfIdx = -1;
    builder.setInsertionPointAfter(func);
    transform::SequenceOp seqOp = builder.create<transform::SequenceOp>(
        func.getLoc(), TypeRange(),
        transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [](OpBuilder &b, Location nested, Value rootH) {
          b.create<transform::YieldOp>(nested, ValueRange());
        });
    auto loc = seqOp.getLoc();
    builder.setInsertionPointToStart(seqOp.getBodyBlock());
    func.walk([&](Block *block) {
      int loopIdxInVf = -1;
      Operation *prevLoop = nullptr;
      SmallVector<Value> loopHandles;
      for (Operation &operation : block->getOperations()) {
        Operation *op = &operation;
        if (op->hasAttr("outlinedLoopWithFill")) {
          assert(isa<scf::ForOp>(op));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          vfIdx++;
          std::string loopLabel = vfNamePrefix + std::to_string(vfIdx);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(
              getOpTransformHandle(loopLabel, builder, seqOp));
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx),
                             /*isFill=*/true);
          prevLoop = nullptr;
        } else if (loopHasOutlinedLoopAttr(op)) {
          assert(isa<scf::ForOp>(op));
          if (!prevLoop || op->getPrevNode() != prevLoop) {
            vfIdx++;
            loopIdxInVf = -1;
          }
          loopIdxInVf++;
          std::string loopLabel = vfNamePrefix + std::to_string(vfIdx) + "_" +
                                  std::to_string(loopIdxInVf);
          op->setAttr(loopLabel, builder.getUnitAttr());
          loopHandles.push_back(
              getOpTransformHandle(loopLabel, builder, seqOp));
          prevLoop = op;
        } else {
          outlineLoopsIntoVF(loopHandles, loc, builder,
                             vfNamePrefix + std::to_string(vfIdx));
          prevLoop = nullptr;
        }
      }
    });
    transform::TransformOptions options;
    options.enableExpensiveChecks(false);
    LogicalResult result = transform::applyTransformNamedSequence(
        func, seqOp, func->getParentOfType<ModuleOp>(), options);
    seqOp->erase();
    if (failed(result))
      func->emitError(
          "Failed to outline vector function from loops of current function.");
  });

  // Preserve the existing VF boundaries first, then legalize loop counts
  // inside those functions. No chunk labels or regrouping are needed.
  const auto calls = collectVFCalls(module);
  module.walk([&](func::FuncOp func) {
    if (!hivm::isVF(func))
      return;
    auto it = calls.find(func);
    if (it == calls.end() || !it->second) {
      func.emitRemark("VF does not have exactly one direct call site; ")
          << "16-bit loop trip limit was not checked";
      return;
    }
    if (failed(splitLargeTripOutlinedLoops(func, it->second)))
      signalPassFailure();
  });

  module.walk([&](scf::ForOp forOp) { removeLabelsForOutlinedLoops(forOp); });
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createOutlineVectorFunctionPass() {
  return std::make_unique<OutlineVectorFunctionPass>();
}
