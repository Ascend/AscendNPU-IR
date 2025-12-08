//===- TileCubeVectorLoop.cpp - Tile Cube and Vector Loop on Local Buffer -===//
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

#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "tile-cube-vector-loop"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_TILECUBEVECTORLOOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
const static llvm::StringLiteral kOpToTile = "op_to_tile_{0}_{1}";
const static llvm::StringLiteral kCubeProducerToFuse =
    "cube_producer_to_fuse_{0}";
const static llvm::StringLiteral kVectorProducerToFuse =
    "vector_producer_to_fuse_{0}";
const static llvm::StringLiteral kDummyStore = "dummy_store";

//===----------------------------------------------------------------------===//
// Utils for constructing transform sequence.
//===----------------------------------------------------------------------===//

namespace transform_utils {
using SequenceBodyBuilderFn =
    std::function<void(OpBuilder &, Location, BlockArgument)>;

void applyCleanUpPatterns(OpBuilder &builder, Location loc, Value target) {
  auto bodyBuilderFn = [](OpBuilder &p, Location loc) {
    p.create<transform::ApplyCanonicalizationPatternsOp>(loc);
  };
  auto applyPatternsOp =
      builder.create<transform::ApplyPatternsOp>(loc,
                                                 /*target=*/target,
                                                 /*bodyBuilder=*/bodyBuilderFn);
  // Enable CSE because loop fuse will check of the the exact same loop bound
  // values.
  applyPatternsOp.setApplyCse(true);
  // Disable simplify trivial loops pattern because we might tile a loop of
  // trip count 1
  SmallVector<Attribute> disabledPatterns = {
      builder.getStringAttr("SimplifyTrivialLoops")};
  applyPatternsOp.setDisablePatternsAttr(
      builder.getArrayAttr(disabledPatterns));
}

Value getFuncHandle(OpBuilder &builder, Location loc, Value transformHandle) {
  return builder.create<transform::MatchOp>(
      loc, transformHandle,
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
}

Value getMatchHandle(OpBuilder &builder, Location loc, Value target,
                     StringRef attrName, bool reverse = false) {
  auto matchedValue =
      builder
          .create<transform::MatchOp>(
              loc, target, /*ops=*/ArrayAttr{},
              /*op_attrs=*/
              builder.getDictionaryAttr(ArrayRef<NamedAttribute>{
                  builder.getNamedAttr(attrName, builder.getUnitAttr())}))
          .getResults();

  if (!reverse)
    return matchedValue;

  return builder.create<transform::ReverseOp>(
      loc,
      /*result=*/TypeRange{builder.getType<transform::AnyOpType>()},
      /*target=*/matchedValue);
}

Value fuseLoops(OpBuilder &builder, Location loc,
                const SmallVector<Value> &loops) {
  if (loops.empty())
    return Value();

  auto fusedLoop = loops.front();
  for (auto nextLoop : llvm::drop_begin(loops)) {
    fusedLoop = builder
                    .create<transform::LoopFuseSiblingOp>(
                        loc,
                        /*fused_loop=*/builder.getType<transform::AnyOpType>(),
                        /*target=*/fusedLoop,
                        /*source=*/nextLoop)
                    .getFusedLoop();
  }
  return fusedLoop;
}

transform::ExtendedFuseIntoContainingOp
createFuseIntoContainingOp(OpBuilder &builder, Location loc, Value producerOp,
                           const SmallVector<Value> &containingLoopValues,
                           size_t numContainingLoop) {
  return builder.create<transform::ExtendedFuseIntoContainingOp>(
      loc,
      /*fused_op=*/
      std::vector<Type>(numContainingLoop,
                        builder.getType<transform::AnyOpType>()),
      /*new_containing_op=*/
      std::vector<Type>(numContainingLoop,
                        builder.getType<transform::AnyOpType>()),
      /*producer_op=*/producerOp,
      /*containing_op=*/containingLoopValues,
      /*duplicate_producer=*/
      BoolAttr::get(builder.getContext(), true));
}

transform::TileUsingForOp
createTileUsingForOp(OpBuilder &builder, Location loc, Value target,
                     ArrayRef<int64_t> staticTileSizes,
                     ArrayRef<int64_t> interchange = {}) {
  return builder.create<transform::TileUsingForOp>(
      loc, /*loops=*/TypeRange{builder.getType<transform::AnyOpType>()},
      /*target=*/target, staticTileSizes,
      /*interchange=*/interchange, /*scalable_sizes=*/std::nullopt);
}
} // namespace transform_utils

inline bool isRankedTensor(Type t) { return isa<RankedTensorType>(t); }

template <typename OpType>
Operation *traceConsumerTo(Operation *definingOp,
                           SmallVector<Operation *> &trace) {
  if (!definingOp)
    return nullptr;

  if (isa<OpType>(definingOp))
    return definingOp;

  for (auto result : definingOp->getResults()) {
    for (auto *resultUser : result.getUsers()) {
      trace.push_back(resultUser);
      if (auto tracedUser = traceConsumerTo<OpType>(resultUser, trace))
        return tracedUser;

      trace.pop_back();
    }
  }
  return nullptr;
}

template <typename OpType>
Operation *traceProducerTo(Operation *definingOp,
                           SmallVector<Operation *> &trace) {
  // This guarantee that we will not cross block boundaries.
  if (!definingOp)
    return nullptr;

  if (isa<OpType>(definingOp))
    return definingOp;

  for (auto operand : definingOp->getOperands()) {
    if (isa<BlockArgument>(operand))
      return nullptr;

    auto *operandDefiningOp = operand.getDefiningOp();
    trace.push_back(operandDefiningOp);
    if (auto tracedProducer = traceProducerTo<OpType>(operandDefiningOp, trace))
      return tracedProducer;

    trace.pop_back();
  }
  return nullptr;
}

inline bool hasHIVMUser(Operation *op) {
  if (!isa_and_nonnull<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
    return false;

  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<HIVMStructuredOp>(user);
  });
}

/// Pattern to lift memref loads to tensor loads.
///
/// Convert:
/// ```mlir
///  hivm.hir.load ins(%a: memref<?xf16>) outs(%b: memref<?xf16>)
///  %b_t = bufferization.to_tensor %b : memref<?xf16>
///  some_user(%b_t)
/// ```
/// To:
/// ```mlir
///  %a_t = bufferization.to_tensor %a: memref<?xf16>
///  %b_t = bufferization.to_tensor %b: memref<?xf16>
///  %res = hivm.hir.load ins(%a_t: tensor<?xf16>)
///                       outs(%b_t: tensor<?xf16>) -> tensor<?xf16>
///  some_user(%res)
/// ```
///
/// Restriction:
///   - the memref load's dst operand must have one user that is a
///   `bufferization.to_tensor` op
class LiftToTensor : public OpRewritePattern<hivm::LoadOp> {
public:
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Value dst = loadOp.getTarget();
    auto maybeDstMemRefType = dyn_cast<MemRefType>(dst.getType());
    if (!maybeDstMemRefType)
      return rewriter.notifyMatchFailure(
          loadOp, "No need to lift load that has tensor outs");

    bufferization::ToTensorOp toTensorUser = nullptr;
    for (Operation *user : dst.getUsers()) {
      if (user == loadOp)
        continue;

      auto userDefOp = dyn_cast_if_present<bufferization::ToTensorOp>(user);
      if (!userDefOp)
        continue;

      if (toTensorUser != nullptr)
        return rewriter.notifyMatchFailure(
            loadOp,
            "dst's must only have a single, bufferization.to_tensor user");

      toTensorUser = userDefOp;
    }
    if (!toTensorUser)
      return rewriter.notifyMatchFailure(
          loadOp, "dst's must have a bufferization.to_tensor user");

    Location loc = loadOp->getLoc();
    Value src = loadOp.getSource();
    rewriter.setInsertionPoint(loadOp);
    if (isa<BaseMemRefType>(src.getType()))
      src = rewriter.create<bufferization::ToTensorOp>(loc, src,
                                                       /*restrict=*/true,
                                                       /*writable=*/true);

    rewriter.setInsertionPointAfter(toTensorUser);
    auto tensorType = RankedTensorType::get(
        maybeDstMemRefType.getShape(), maybeDstMemRefType.getElementType());
    // Create a tensor load, using the tensorized source/dst values
    auto tensorLoadOp = rewriter.create<hivm::LoadOp>(
        loc, SmallVector<Type>{tensorType},
        SmallVector<Value>{src, toTensorUser.getResult()}, loadOp->getAttrs());
    // Replace the users of `bufferization.to_tensor` by the new load
    // Except it's use in the newly created load op
    rewriter.replaceAllUsesExcept(toTensorUser.getResult(),
                                  tensorLoadOp.getResult(0), {tensorLoadOp});

    // Erase the memref load
    rewriter.eraseOp(loadOp);
    return success();
  }
};

LogicalResult liftMemRefLoadsInLoop(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LiftToTensor>(ctx);
  return applyPatternsGreedily(module, std::move(patterns));
}

/// Pattern to shrink memref alloc's size.
///
/// Convert:
/// ```mlir
///  %alloc = memref.alloc() : memref<128xf32>
///  %subviewed = memref.subview %alloc : memeref<64xf32>
///  some_user(%subviewed)
/// ```
/// To:
/// ```mlir
///  %alloc = memref.alloc() : memref<64xf32>
///  some_user(%alloc)
/// ```
///
/// Restriction:
///   - the alloc is directly subviewed to use, and has no other users.
class ShrinkAlloc : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto users = allocOp.getResult().getUsers();
    if (!llvm::hasSingleElement(users))
      return rewriter.notifyMatchFailure(allocOp,
                                         "alloc has more than one user");

    Operation *singleUser = *users.begin();
    auto subview = dyn_cast_if_present<memref::SubViewOp>(singleUser);
    if (!subview)
      return failure();

    rewriter.setInsertionPoint(subview);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        subview, subview.getMixedSizes(),
        allocOp.getMemref().getType().getElementType());
    return success();
  }
};

LogicalResult shrinkAlloc(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ShrinkAlloc>(ctx);
  return applyPatternsGreedily(module, std::move(patterns));
}

/// Pattern to remove the dummy store.
class RemoveDummyStore : public OpRewritePattern<hivm::StoreOp> {
public:
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (!storeOp->hasAttr(kDummyStore))
      return failure();

    Value source = storeOp.getSource();
    auto hivmSource =
        dyn_cast_if_present<HIVMStructuredOp>(source.getDefiningOp());
    if (!hivmSource)
      return storeOp->emitError("Dummy store's source is not a HIVM op");

    // Clone the source op right before the store op because we want to
    // replace its init with the dummy store's init. But the defining op of
    // the init operand might come after the source op.
    //
    // We can do this easily because it's guaranteed that there is no other
    // users of the stored op because we already replaced all of its users by
    // the dummy store. So the IR looks like:
    // ```mlir
    //   %a = hivm.hir.op
    //   ..                                 // there is no other users of %a
    //   %store = hivm.hir.store ins(%a)
    //   ...
    //   other_users(%store)
    // ```
    rewriter.setInsertionPoint(storeOp);
    IRMapping mapping;
    mapping.map(hivmSource.getDpsInitOperand(0)->get(),
                storeOp.getDpsInitOperand(0)->get());
    Operation *newOp = rewriter.clone(*hivmSource.getOperation(), mapping);
    rewriter.replaceOp(hivmSource, newOp);
    rewriter.replaceAllUsesWith(storeOp.getResult(0), storeOp.getSrc());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// OpToTile
//===----------------------------------------------------------------------===//

/// Structure for holding parameters necessary for tiling.
struct TilingParams {
  SmallVector<int64_t> tiledDims;
  SmallVector<int64_t> tileSizes;
  SmallVector<int64_t> tileInterchange;
};

/// Structure for holding information on an op to-be-tiled.
struct OpToTile {
  /// Order by the ops appearance in the IR.
  bool operator<(const OpToTile &other) {
    assert(this->op);
    assert(other.op);
    assert(this->op->getBlock() == other.op->getBlock() &&
           "operations must be in the same block");
    return this->op->isBeforeInBlock(other.op);
  }

#ifndef NDEBUG
  void dump() {
    LDBG("Dumping OpToTile info: ");
    llvm::outs() << "  op tag: " << tag << "\n";
    llvm::outs() << "  tiling dimensions: "
                 << utils::debugger::to_string(tilingParams.tiledDims) << "\n";
    llvm::outs() << "  tile size: "
                 << utils::debugger::to_string(tilingParams.tileSizes) << "\n";
    llvm::outs() << "  tiling interchange: "
                 << utils::debugger::to_string(tilingParams.tileInterchange)
                 << "\n";
  }
#endif

  /// Pointer to the original operation to tile in the payload IR.
  /// \note Be careful to access this because it's like to become a dangling
  /// pointer.
  Operation *op{nullptr};

  /// The unique identifier given to the op.
  std::string tag{};

  /// Tiling params.
  TilingParams tilingParams;
};

//===----------------------------------------------------------------------===//
// Base Loop Information.
//===----------------------------------------------------------------------===//

/// Base class for representing a loop that needs sub-tiling.
class LoopInfo {
public:
  explicit LoopInfo(size_t idxIn, int64_t targetTripCount,
                    hivm::TCoreType loopCoreTypeIn)
      : idx(idxIn), loopCoreType(loopCoreTypeIn), tripCount(targetTripCount) {}

  virtual ~LoopInfo() = default;

  static bool classof(const LoopInfo *) { return true; }

public:
  /// Perform post transformation action to the loop.
  virtual LogicalResult
  performPostTransformationAction(ModuleOp /*module*/) const {
    return success();
  }

  /// Get the core type of the loop.
  hivm::TCoreType getLoopType() const { return loopCoreType; }

  /// Get the unique index of the loop.
  size_t getIdx() const { return idx; };

  /// Get or set the target trip count of the tiled loop.
  int64_t getTripCount() const { return tripCount; }
  void setTripCount(int64_t newTripCount) {
    if (newTripCount < 0)
      return;

    tripCount = newTripCount;
  }

  /// Get all the ops to tile.
  ArrayRef<OpToTile> getOpTileInfo() const { return opsToTileAndFuse; }

  /// Record the tiling params for the `op`.
  void recordOpToTile(Operation *op, TilingParams &&tilingParams) {
    LDBG("Try to record op to tile...");
    if (!op)
      return;

    if (!isa<TilingInterface>(op) || !isa<DestinationStyleOpInterface>(op)) {
      LDBG("op is not a tileable op and destination style op");
      return;
    }

    // Only handle stores with single return value
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    if (dpsOp.getNumDpsInits() != 1) {
      LDBG("op has more than one dps inits");
      return;
    }

    OpToTile opTileInfo;
    opTileInfo.op = op;
    opTileInfo.tag = generateOpToTileTag();
    opTileInfo.tilingParams = std::move(tilingParams);
    lazySetAttr(op, opTileInfo.tag, UnitAttr::get(op->getContext()));

    // Sort the ops by their topological ordering in the payload.
    // We should try our best to minimize the changes in the op ordering.
    opsToTileAndFuse.push_back(opTileInfo);
    llvm::sort(opsToTileAndFuse.begin(), opsToTileAndFuse.end());

    LDBG("Successfully recorded op to tile: " << *op);
    LLVM_DEBUG(opTileInfo.dump());
  };

  /// Generate a producer tag for the current loop so that we can match the
  /// producer ops later.
  std::string generateProducerTag() const {
    return llvm::formatv((loopCoreType == hivm::TCoreType::CUBE
                              ? kCubeProducerToFuse
                              : kVectorProducerToFuse)
                             .data(),
                         getIdx())
        .str();
  }

  /// Create the transform sequence to tile the loop.
  transform::NamedSequenceOp createTransformSequence(OpBuilder &builder,
                                                     Location loc) {
    return builder.create<transform::NamedSequenceOp>(
        loc,
        /*symName=*/transform::TransformDialect::kTransformEntryPointSymbolName,
        /*rootType=*/
        builder.getType<transform::AnyOpType>(),
        /*resultType=*/TypeRange{},
        /*bodyBuilder=*/getBodyBuilder());
  }

  /// Lazily set an attribute to the op.
  void lazySetAttr(Operation *op, const std::string &attrName,
                   Attribute value) {
    recordLazyAction([op, attrName, value]() {
      if (!op)
        llvm_unreachable("corrupted operation");

      op->setAttr(attrName, value);
    });
  }

  /// Commit all recorded lazy actions at once.
  ///
  /// During info collection, it's possible that we're uncertain whether we
  /// need to tile the loop or not. This will be apparent after the info
  /// collection process.
  /// However, there is an overlap between the info collection process and
  /// the actions that we want to do **AFTER** we decide to do tiling.
  /// (For example, adding attributes to operations.)
  /// On one hand, we can choose to unify the two processes, but for most of the
  /// times we don't want to directly commit the actions during info collection
  /// because this might mess up with internal states.
  /// Therefore, we provide this mechanism so that we an record the actions and
  /// then apply afterwards.
  void commitLazyActions() {
    for (auto &action : lazyActions)
      action();

    lazyActions.clear();
  }

private:
  std::string generateOpToTileTag() const {
    return llvm::formatv(kOpToTile.data(), getIdx(), opsToTileAndFuse.size())
        .str();
  }

  void recordLazyAction(std::function<void()> action) {
    lazyActions.push_back(std::move(action));
  }

  //===--------------------------------------------------------------------===//
  // Utils for constructing transform sequence.
  //===--------------------------------------------------------------------===//

  /// Define the common body builder to perform loop tiling, fusing, and fuse
  /// into.
  transform_utils::SequenceBodyBuilderFn getBodyBuilder() {
    return [this](OpBuilder &builder, Location loc, BlockArgument ba) {
      this->funcHandle = transform_utils::getFuncHandle(builder, loc, ba);

      SmallVector<Value> tiledLoopHandles;
      for (const OpToTile &info : getOpTileInfo()) {
        // Step 1(a): Match the ops to tile
        auto opHandle =
            transform_utils::getMatchHandle(builder, loc, ba, info.tag);
        // Step 1(b): Tile the op
        auto tiledHandle = transform_utils::createTileUsingForOp(
            builder, loc, opHandle, info.tilingParams.tileSizes);

        // There should only be one tiling dimension
        tiledLoopHandles.emplace_back(tiledHandle.getLoops().front());
      }
      transform_utils::applyCleanUpPatterns(builder, loc, this->funcHandle);
      // Step 2: Fuse independent loops together
      Value fusedLoop =
          transform_utils::fuseLoops(builder, loc, tiledLoopHandles);

      // Step 3: Fuse producers into tiled loop
      auto producersToFuse = transform_utils::getMatchHandle(
          builder, loc, ba, generateProducerTag(), /*reverse=*/true);
      createForEachFuseIntoBlock(builder, loc, producersToFuse, fusedLoop);
      builder.create<transform::YieldOp>(loc);
    };
  }

  /// Create a `for_each` loop to fuse producers into loop one-by-one, while
  /// performing canonicalization.
  ///
  /// \note this is a member function because we want to get access to to global
  /// function handle of the transform sequence, which is stored in `funcHandle`
  void createForEachFuseIntoBlock(OpBuilder &builder, Location loc,
                                  Value producer, Value loop) const {
    ImplicitLocOpBuilder::InsertionGuard g(builder);
    size_t kNumContainingLoops = 1;
    auto forEachRegionBuilderFn = [&kNumContainingLoops, &loc, &loop, &builder,
                                   this](ImplicitLocOpBuilder &forEachBuilder,
                                         Block &block) -> void {
      auto blockArg = block.getArgument(0);
      transform_utils::applyCleanUpPatterns(builder, loc, this->funcHandle);
      auto op = transform_utils::createFuseIntoContainingOp(
          forEachBuilder, loc, blockArg, SmallVector<Value>{loop},
          kNumContainingLoops);
      forEachBuilder.create<transform::YieldOp>(loc, op->getResults());
    };
    SmallVector<Type> forEachResultTypes(
        kNumContainingLoops * 2, builder.getType<transform::AnyOpType>());
    auto foreach =
        builder.create<transform::ForeachOp>(loc,
                                             /*results=*/forEachResultTypes,
                                             /*target=*/ValueRange{producer},
                                             /*with_zip_shortest=*/false);
    Region &body = foreach.getBody();
    Block *block = builder.createBlock(
        &body, /*insertPt=*/{}, {builder.getType<transform::AnyOpType>()},
        {foreach.getLoc()});
    ImplicitLocOpBuilder b(loc, builder);
    forEachRegionBuilderFn(b, *block);
    transform::ForeachOp::ensureTerminator(body, builder, foreach.getLoc());
  }

private:
  /// Unique ID of the loop.
  size_t idx{0};

  /// Loop core type.
  hivm::TCoreType loopCoreType{hivm::TCoreType::CUBE_OR_VECTOR};

  /// Ops to tile and fuse in this loop.
  SmallVector<OpToTile> opsToTileAndFuse{};

  /// The target trip count of the tiled loop.
  int64_t tripCount{0};

  /// A collection of lazy actions to be applied.
  std::vector<std::function<void()>> lazyActions;

  /// Function handle in the transform sequence.
  Value funcHandle{};
};

//===----------------------------------------------------------------------===//
// Cube Loop Information.
//===----------------------------------------------------------------------===//

class CubeLoopInfo : public LoopInfo {
public:
  CubeLoopInfo(size_t idx, int64_t targetTripCount)
      : LoopInfo(idx, targetTripCount, hivm::TCoreType::CUBE) {}

  static bool classof(const LoopInfo *T) {
    return T->getLoopType() == hivm::TCoreType::CUBE;
  }

  void setFixpipeOp(hivm::FixpipeOp op) { fixpipeOp = op; }
  hivm::FixpipeOp getFixpipeOp() { return fixpipeOp; }

private:
  hivm::FixpipeOp fixpipeOp{nullptr};
};

//===----------------------------------------------------------------------===//
// Vector Loop Information.
//===----------------------------------------------------------------------===//

class VectorLoopInfo : public LoopInfo {
public:
  VectorLoopInfo(size_t idx, int64_t targetTripCount)
      : LoopInfo(idx, targetTripCount, hivm::TCoreType::VECTOR) {}

  static bool classof(const LoopInfo *T) {
    return T->getLoopType() == hivm::TCoreType::VECTOR;
  }

  LogicalResult
  performPostTransformationAction(ModuleOp module) const override {
    MLIRContext *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveDummyStore>(ctx);
    return applyPatternsGreedily(module, std::move(patterns));
  }
};

class TileCubeVectorLoopPass
    : public impl::TileCubeVectorLoopBase<TileCubeVectorLoopPass> {
public:
  explicit TileCubeVectorLoopPass(TileCubeVectorLoopOptions options)
      : TileCubeVectorLoopBase(options) {}
  TileCubeVectorLoopPass(const TileCubeVectorLoopPass &other)
      : TileCubeVectorLoopBase(other) {
    this->tiledMixCubeLoopNumber = other.tiledMixCubeLoopNumber;
    this->tiledMixVectorLoopNumber = other.tiledMixVectorLoopNumber;
  }
  TileCubeVectorLoopPass &
  operator=(const TileCubeVectorLoopPass &other) = delete;

  /// Main entry point to the pass.
  void runOnOperation() final;

private:
  /// Main entry to collect loop information.
  void collectLoopInfo(ModuleOp topLevelModule);
  LogicalResult collectCubeLoopInfo(scf::ForOp cubeLoop);
  LogicalResult collectVectorLoopInfo(scf::ForOp vectorLoop);
  std::optional<TilingParams> calculateFixpipeTiling(CubeLoopInfo &info) const;

  /// Main entry to apply transformation
  LogicalResult applyTransformation(ModuleOp topLevelModule, LoopInfo &info);

  /// Loops to perform sub-tiling.
  SmallVector<std::unique_ptr<LoopInfo>> loopsToTile;

  /// Device spec
  hacc::HACCTargetDeviceSpecInterface spec{};
};

/// Generate tiling params for vector ops based on shape, dimension, and the
/// target trip count.
TilingParams getVectorTiling(ShapedType tilingShape, int64_t tilingDim,
                             int64_t tripCount) {
  TilingParams result;
  int64_t rank = tilingShape.getRank();
  result.tiledDims = {tilingDim};
  result.tileSizes = SmallVector<int64_t>(rank, 0);
  result.tileSizes[tilingDim] =
      llvm::divideCeilSigned(tilingShape.getDimSize(tilingDim), tripCount);
  return result;
}

/// Try to collect tiling information for store op.
///
/// Return failure if dimension analyzer failed to analyze tiling dimension.
LogicalResult
tryCollectTilingInfoForStore(hivm::StoreOp storeOp, VectorLoopInfo &info,
                             hivm::detail::DimensionAnalyzer &analyzer) {
  LDBG("Store op: " << *storeOp);
  auto tilingDimForStore = analyzer.getTilingDim(storeOp.getSrc());
  if (tilingDimForStore == -1) {
    LDBG("Cannot determine the tiling dim of Store Op!");
    return failure();
  }

  ShapedType dstType = storeOp.getDstOperandType();
  info.recordOpToTile(storeOp, getVectorTiling(dstType, tilingDimForStore,
                                               info.getTripCount()));
  return success();
}

/// Try to collect tiling information for yield op by iterating over the yielded
/// values.
///
/// Return failure if:
///   a) the yielded value does not have tensor type.
///   b) the yielded value is not produced by a HIVM Structured Op.
///   c) dimension analyzer failed to analyze tiling dimension.
LogicalResult
tryCollectTilingInfoForYield(scf::YieldOp yieldOp, Value yieldedVal,
                             VectorLoopInfo &info,
                             hivm::detail::DimensionAnalyzer &analyzer) {
  if (!isRankedTensor(yieldedVal.getType())) {
    LDBG("Yielding non-tensor values, skip");
    return failure();
  }

  // Try to find the immediate HIVM producer
  // Note: Currently we assume that all vector computations are done in
  // tensor.
  SmallVector<Operation *> trace = {yieldOp, yieldedVal.getDefiningOp()};
  Operation *tracedProducer =
      traceProducerTo<HIVMStructuredOp>(yieldedVal.getDefiningOp(), trace);
  if (!tracedProducer) {
    LDBG("Cannot find HIVM producer");
    return failure();
  }

  auto hivmOp = cast<HIVMStructuredOp>(tracedProducer);
  if (hivmOp->getNumResults() != 1) {
    LDBG("HIVM Op result number is not one");
    return failure();
  }

  Value singleResult = hivmOp->getResult(0);
  int64_t maybeTilingDim = analyzer.getTilingDim(singleResult);
  if (maybeTilingDim == -1) {
    LDBG("Cannot determine the tiling dim of HIVM Op " << singleResult);
    return failure();
  }

  OpBuilder builder(tracedProducer->getContext());
  Location loc = hivmOp->getLoc();
  builder.setInsertionPointAfter(tracedProducer);
  // If the to-be-tiled hivm op is not a store, it may have data dependency
  // issues with other ops, which makes it difficult to do tiling and loop
  // fusion.
  // Therefore, we create a dummy store op and tile it instead.
  Value originalInit = hivmOp.getDpsInitOperand(0)->get();
  auto dummyStore = builder.create<hivm::StoreOp>(loc, singleResult.getType(),
                                                  singleResult, originalInit);

  // Modify the original hivm op's init operand to `tensor.empty`.
  // This is to prevent creating a "multiple-producer" situation during
  // fuse into.
  builder.setInsertionPoint(tracedProducer);
  Value newInit =
      utils::createTmpBufferOrTensorWithTargetType(builder, loc, originalInit);
  hivmOp.setDpsInitOperand(0, newInit);

  // Only replace the user by the dummy store on the chain to the scf.yield.
  // For example:
  // ```mlir
  // %val = ...                         (original value)
  // %store = hivm.hir.store ins(%val)
  // user1(%val)
  // scf.yield %val                     (only replace this!)
  // ```
  SetVector<Operation *> chainOfUsersToYield = {trace.begin(), trace.end()};
  singleResult.replaceUsesWithIf(
      dummyStore.getResult(0),
      /*shouldReplace=*/[&chainOfUsersToYield](OpOperand &operand) -> bool {
        return chainOfUsersToYield.contains(operand.getOwner());
      });

  ShapedType dstType = dummyStore.getDstOperandType();
  info.recordOpToTile(dummyStore, getVectorTiling(dstType, maybeTilingDim,
                                                  info.getTripCount()));
  // This store is a dummy one, mark it so that we can erase it later.
  info.lazySetAttr(dummyStore, kDummyStore.str(),
                   UnitAttr::get(dummyStore->getContext()));
  return success();
}

/// Trace down from the LoadOp to the MmadL1Op while marking intermediate ops as
/// producers to fuse-into.
void markOpTouchingMMAD(hivm::LoadOp load, CubeLoopInfo &info) {
  // Ignore memref loads because the targeted loads should be tensor or
  // has been converted to tensor.
  bool isMemRefLoad = load.hasPureBufferSemantics();
  if (isMemRefLoad)
    return;

  // Record all operations that operated on the inputs to mmad
  SmallVector<Operation *> trace{load};
  Operation *maybeMmadL1Op =
      traceConsumerTo<hivm::MmadL1Op>(load.getResult(0).getDefiningOp(), trace);

  if (!maybeMmadL1Op)
    return;

  // Also consider the loaded operands as potential producers
  trace.append(
      {load.getSource().getDefiningOp(), load.getTarget().getDefiningOp()});

  // Tag the producers so that we can fuse into later.
  llvm::for_each(trace, [&load, &info](Operation *op) {
    if (!op)
      return;

    info.lazySetAttr(op, info.generateProducerTag(),
                     UnitAttr::get(load.getContext()));
  });
}

/// Calculate the tiling parms for the fixpipe op in the cube loop.
///
/// Rules:
///   1) Select the largest axis in [M, N] to tile.
///   2) If L0C fits, we will not tile it. This might not always be the case
///      because L1 may not fit.
std::optional<TilingParams>
TileCubeVectorLoopPass::calculateFixpipeTiling(CubeLoopInfo &info) const {
  auto fixpipeOp = info.getFixpipeOp();
  auto dstType = fixpipeOp.getDstOperandType();
  if (!dstType.hasStaticShape()) {
    LDBG("Fixpipe dst doesn't have static shape");
    return std::nullopt;
  }

  auto maybeStaticTotalSize = mlir::utils::getStaticTotalSizeInBits(
      dstType.getShape(), dstType.getElementType());
  if (!maybeStaticTotalSize.has_value()) {
    LDBG("Failed to calculate static total size for fixpipe dst");
    return std::nullopt;
  }

  LDBG("Total size required in L0C: " << maybeStaticTotalSize.value());
  const int64_t kL0CSizeInBits = hacc::utils::getIntegerSpecValue(
      this->spec.getSpecForIdentifierEnum(hacc::DeviceSpec::L0C_SIZE));
  if (maybeStaticTotalSize.value() <= kL0CSizeInBits &&
      info.getTripCount() != 1) {
    LDBG("No need to tile because the data can fit on L0C");
    fixpipeOp.emitWarning(
        "Ignoring candidate cube loop trip count because it's suboptimal");
    info.setTripCount(1);
    return std::nullopt;
  }

  [[maybe_unused]] int64_t calculatedTripCount =
      llvm::divideCeilSigned(maybeStaticTotalSize.value(), kL0CSizeInBits);
  LDBG("Calculated minimum trip count: " << calculatedTripCount);

  TilingParams result;
  assert(dstType.getRank() == 2 && "MmadL1 operand rank must be two");
  // Tile the larger dimension
  int64_t singleTileDim = dstType.getDimSize(0) > dstType.getDimSize(1) ? 0 : 1;
  result.tiledDims = {singleTileDim};
  result.tileSizes = SmallVector<int64_t>(dstType.getRank(), 0);
  result.tileSizes[singleTileDim] = llvm::divideCeilSigned(
      dstType.getDimSize(singleTileDim), info.getTripCount());
  return result;
}

void TileCubeVectorLoopPass::collectLoopInfo(ModuleOp topLevelModule) {
  topLevelModule->walk([this](scf::ForOp candidateLoop) {
    auto maybeLoopCoreType = candidateLoop->getAttrOfType<hivm::TCoreTypeAttr>(
        hivm::kPipelinedLoopCoreTypeAttrName);

    if (!maybeLoopCoreType)
      return WalkResult::advance();

    // No need to walk inside loop.
    if (maybeLoopCoreType.getTcoretype() == hivm::TCoreType::CUBE) {
      LDBG("Collecting cube loop info");
      (void)collectCubeLoopInfo(candidateLoop);
      return WalkResult::skip();
    }
    if (maybeLoopCoreType.getTcoretype() == hivm::TCoreType::VECTOR) {
      LDBG("Collecting vector loop info");
      (void)collectVectorLoopInfo(candidateLoop);
      return WalkResult::skip();
    }

    return WalkResult::advance();
  });
}



static bool
areValuesAlignedAfterTiling(ValueRange valueRange,
                            mlir::hivm::detail::DimensionAnalyzer &analyzer,
                            int64_t tilingFactor, int64_t alignSize) {
  for (auto value : valueRange) {
    auto tilingDim = analyzer.getTilingDim(value);
    // If there is no tiling dim, we are not tiling it.
    if (tilingDim == -1)
      return true;
    auto resultType = dyn_cast<RankedTensorType>(value.getType());
    if (!resultType || ShapedType::isDynamicShape(resultType.getShape()))
      continue;
    int bitUsed = 1;
    for (auto dim = 0; dim < resultType.getRank(); dim++) {
      if (dim == tilingDim) {
        bitUsed = bitUsed * resultType.getDimSize(dim) / tilingFactor;
      } else {
        bitUsed = bitUsed * resultType.getDimSize(dim);
      }
    }
    bitUsed = bitUsed * resultType.getElementTypeBitWidth();
    if (bitUsed % alignSize != 0)
      return false;
  }
  return true;
}

/// Collect vector loop information.
///
/// We assume that the vector loop always have the following structure:
/// ```mlir
/// for ... {
//    %a = ...
///   hivm.hir.store
//    %b = ...
//    %c = ...
///   yield %a, %b, %c
/// }
/// ```
/// Note that the yielded values may not be stored. So we have to tile
/// both the store and the yielded values.
LogicalResult
TileCubeVectorLoopPass::collectVectorLoopInfo(scf::ForOp vectorLoop) {
  VectorLoopInfo info(loopsToTile.size(),
                      static_cast<int64_t>(this->tiledMixVectorLoopNumber));
  if (info.getTripCount() == 1)
    return success();

  hivm::detail::DimensionAnalyzer analyzer(vectorLoop);
  if (failed(analyzer.initialize()))
    return vectorLoop->emitOpError("Failed to analyze vector loop");

  // Compute vector ops' tiling first
  analyzer.computeTilingDim();
  auto ubAlignSize = hacc::utils::getIntegerSpecValue(
      this->spec.getSpecForIdentifierEnum(hacc::DeviceSpec::UB_ALIGN_SIZE));

  // Visit all store ops
  auto walkResult =
      vectorLoop->walk([&analyzer, &info, &ubAlignSize](Operation *op) {
        if (auto storeOp = dyn_cast<hivm::StoreOp>(op)) {
          if (failed(tryCollectTilingInfoForStore(storeOp, info, analyzer)))
            return WalkResult::interrupt();

          return WalkResult::advance();
        }

	// TODO:: Use MarkStrideAlign to annotate the unaligned axis and
	// rely on StrideAlign to make it aligned
        if (!areValuesAlignedAfterTiling(op->getResults(), analyzer,
                                         info.getTripCount(), ubAlignSize) ||
            !areValuesAlignedAfterTiling(op->getOperands(), analyzer,
                                         info.getTripCount(), ubAlignSize)) {
          return WalkResult::interrupt();
        }

        // Mark ops as intermediate producers for fuse into.
        // Currently automatically consider all HIVM/Collapse/Expand ops as
        // producers
        if (isa<HIVMStructuredOp>(op) || hasHIVMUser(op))
          info.lazySetAttr(op, info.generateProducerTag(),
                           UnitAttr::get(op->getContext()));

        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return vectorLoop.emitOpError("Failed to collect vector loop tiling info");

  // Visit yield op next because it will generate dummy store op
  auto yieldOp = dyn_cast<scf::YieldOp>(vectorLoop.getBody()->getTerminator());
  if (!yieldOp)
    llvm_unreachable("scf.for must have a scf.yield terminator");

  for (auto yieldedVal : yieldOp.getOperands())
    if (failed(
            tryCollectTilingInfoForYield(yieldOp, yieldedVal, info, analyzer)))
      return vectorLoop.emitOpError(
          "Failed to collect vector loop tiling info");

  if (info.getOpTileInfo().empty())
    return failure();

  // Finish collecting all info
  info.commitLazyActions();
  loopsToTile.emplace_back(std::make_unique<VectorLoopInfo>(info));
  return success();
}

/// Collect cube loop information.
///
/// We assume that the cube loop always have the following structure:
/// ```mlir
/// for ... {
///   %a = memref.alloc
///   hivm.hir.load ins(%arg_a) outs(%a)
///   %b = hivm.hir.load ins(%arg_b) outs(...)
///   %result = hivm.hir.mmadL1 ins(%a, %b)
///   hivm.hir.fixpipe ins(%result)
/// }
/// ```
/// Note that the load of matrix A/B are not necessarily in the loop.
LogicalResult TileCubeVectorLoopPass::collectCubeLoopInfo(scf::ForOp cubeLoop) {
  CubeLoopInfo info(loopsToTile.size(),
                    static_cast<int64_t>(this->tiledMixCubeLoopNumber));
  if (info.getTripCount() == 1)
    return success();

  // Locate the single fixpipe op
  hivm::FixpipeOp singleFixpipeOp = nullptr;
  auto walkResult =
      cubeLoop->walk([&singleFixpipeOp](hivm::FixpipeOp fixpipeOp) {
        if (!singleFixpipeOp)
          singleFixpipeOp = fixpipeOp;
        else
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return cubeLoop.emitOpError(
        "Currently don't support multiple fixpipe in cube loop");

  if (!singleFixpipeOp)
    return cubeLoop.emitOpError("Cannot find fixpipe in cube loop");

  info.setFixpipeOp(singleFixpipeOp);

  // Calculate fixpipe tiling
  auto maybeTilingResult = calculateFixpipeTiling(info);
  if (!maybeTilingResult.has_value())
    return singleFixpipeOp.emitOpError("Failed to calculate tiling");

  info.recordOpToTile(singleFixpipeOp, std::move(*maybeTilingResult));

  std::optional<Operation *> maybeMmadL1Op =
      traceDefOp<hivm::MmadL1Op>(singleFixpipeOp.getSource());
  if (!maybeMmadL1Op.has_value() || !isa<hivm::MmadL1Op>(maybeMmadL1Op.value()))
    return cubeLoop.emitOpError("Cannot find matmul in cube loop");

  // Only tile the load ops inside the for loop.
  cubeLoop.walk([&info](hivm::LoadOp load) { markOpTouchingMMAD(load, info); });

  // Finish collecting all info
  info.commitLazyActions();
  loopsToTile.emplace_back(std::make_unique<CubeLoopInfo>(info));
  return success();
}

LogicalResult
TileCubeVectorLoopPass::applyTransformation(ModuleOp topLevelModule,
                                            LoopInfo &info) {
  MLIRContext &context = getContext();
  auto builder = OpBuilder(&context);

  LDBG("Creating transform sequence...");
  // Assuming we have a flat module
  builder.setInsertionPointToEnd(topLevelModule.getBody());
  transform::NamedSequenceOp transformSequenceOp =
      info.createTransformSequence(builder, topLevelModule->getLoc());
  if (!transformSequenceOp)
    return topLevelModule.emitError("Failed to create transform sequence");

  LDBG("Before applying transform sequence \n" << *topLevelModule);
  auto entryPoint =
      cast<transform::TransformOpInterface>(transformSequenceOp.getOperation());
  if (!entryPoint)
    return topLevelModule.emitError("Failed to get entry point");

  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(
          topLevelModule, entryPoint, /*transformModule=*/{}, options))) {
    entryPoint.erase();
    LDBG("Failed to apply transform");
    return topLevelModule.emitError("Failed to apply transform");
  }
  // Erase the transform library
  entryPoint.erase();

  LDBG("after applying transformations " << *topLevelModule);
  return success();
}

void TileCubeVectorLoopPass::runOnOperation() {
  ModuleOp topLevelModule = getOperation();
  std::optional<hacc::HACCTargetDeviceSpecInterface> maybeSpecInterface =
      hacc::utils::getNPUTargetSpec(topLevelModule);
  if (!maybeSpecInterface.has_value()) {
    signalPassFailure();
    return;
  }
  this->spec = maybeSpecInterface.value();

  // Preprocessing step: lift copy from memref dialect to tensor dialect
  // because tiling is so much easier in tensor-land.
  if (failed(liftMemRefLoadsInLoop(topLevelModule))) {
    signalPassFailure();
    return;
  }

  // Step 1: collect all loop information
  collectLoopInfo(topLevelModule);

  topLevelModule->setAttr(
      transform::TransformDialect::kWithNamedSequenceAttrName,
      UnitAttr::get(&getContext()));
  // Step 2: for each candidate loop, construct transform library and apply
  // transformation
  for (auto &loopInfo : loopsToTile) {
    LDBG("Try to apply " << (isa<CubeLoopInfo>(*loopInfo) ? "cube" : "vector")
                         << " tiling");
    if (loopInfo->getTripCount() == 1) {
      LDBG("Trip count is 1, skip");
      continue;
    }

    // Roll back transformation if not successful.
    ModuleOp cloned = topLevelModule.clone();
    if (failed(applyTransformation(cloned, *loopInfo))) {
      cloned->emitWarning(
          "Failed to apply loop transformation, reverting back operation");
      cloned->erase();
    } else {
      topLevelModule.getBodyRegion().getBlocks().clear();
      IRMapping map;
      cloned.getBodyRegion().cloneInto(&topLevelModule.getBodyRegion(),
                                       topLevelModule.getBodyRegion().begin(),
                                       map);
      cloned->erase();
    }
    if (failed(loopInfo->performPostTransformationAction(topLevelModule)))
      topLevelModule.emitError("Failed to apply post transformation action");
  }
  topLevelModule->removeAttr(
      transform::TransformDialect::kWithNamedSequenceAttrName);

  // Post processing step: try to shrink alloc's size.
  // This is needed because for copy that is lifted to tensor, it possible that
  // the dst is a full-sized alloc defined out of the loop. If we don't shrink
  // it, there is going to be two problems:
  //   1) there is a waste of space on the local buffer
  //   2) currently our Mmad implementation assumes that the data is contiguous
  //      on cbuf, so a full alloc + subview will have precision error
  if (failed(shrinkAlloc(topLevelModule)))
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createTileCubeVectorLoopPass(
    const TileCubeVectorLoopOptions &options) {
  return std::make_unique<TileCubeVectorLoopPass>(options);
}
