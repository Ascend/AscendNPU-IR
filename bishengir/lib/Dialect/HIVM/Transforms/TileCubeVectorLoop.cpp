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
#include "bishengir/Dialect/Scope/IR/Scope.h"
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
/// All optional operands (pad_value, left/right_padding_num, init_condition)
/// and attributes from the original memref load are preserved.
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
    auto tensorLoadOp = rewriter.create<hivm::LoadOp>(
        loc, TypeRange{tensorType}, loadOp->getOperands(), loadOp->getAttrs());
    tensorLoadOp.getSrcMutable().set(src);
    tensorLoadOp.getDstMutable().set(toTensorUser.getResult());
    // Replace the users of `bufferization.to_tensor` by the new load
    // Except it's use in the newly created load op
    rewriter.replaceAllUsesExcept(toTensorUser.getResult(),
                                  tensorLoadOp.getResult(0), {tensorLoadOp});

    // Erase the memref load
    rewriter.eraseOp(loadOp);
    return success();
  }
};

struct CanonicalizeAllocToTensor : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return failure();
    auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(*op->user_begin());
    if (!toTensorOp)
      return failure();
    auto tensorType = toTensorOp.getType();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        toTensorOp, tensorType.getShape(), tensorType.getElementType());
    rewriter.eraseOp(op);
    return success();
  }
};

LogicalResult liftMemRefLoadsInLoop(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LiftToTensor>(ctx);
  patterns.add<CanonicalizeAllocToTensor>(ctx);
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

    // Dynamic offsets or non-zero offsets are not supported.
    auto offsets = subview.getStaticOffsets();
    if (llvm::any_of(offsets, [](int64_t offset) {
          return ShapedType::isDynamic(offset) || offset != 0;
        }))
      return failure();

    // Dynamic strides or non-unit strides are not supported.
    auto strides = subview.getStaticStrides();
    if (llvm::any_of(strides, [](int64_t stride) {
          return ShapedType::isDynamic(stride) || stride != 1;
        }))
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
  bool operator<(const OpToTile &other) const {
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

  void recordOpToTile(Operation *op, TilingParams &&tilingParams) {
    recordOpToTile(op, generateOpToTileTag(), std::move(tilingParams));
  };

  /// Record the tiling params for the `op`.
  void recordOpToTile(Operation *op, const std::string &customTag,
                      TilingParams &&tilingParams) {
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
    opTileInfo.tag = customTag;
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
  virtual transform::NamedSequenceOp createTransformSequence(OpBuilder &builder,
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
        llvm::report_fatal_error("corrupted operation");

      op->setAttr(attrName, value);
    });
  }

  /// Record & rollback while failed with vector loop info collecting.
  void recordUndoAction(std::function<void()> action) {
    undoActions.push_back(std::move(action));
  }

  void rollback() {
    for (auto &action : llvm::reverse(undoActions))
      action();
    undoActions.clear();
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

protected:
  //===--------------------------------------------------------------------===//
  // Utils for constructing transform sequence.
  //===--------------------------------------------------------------------===//

  /// Define the common body builder to perform loop tiling, fusing, and fuse
  /// into.
  virtual transform_utils::SequenceBodyBuilderFn getBodyBuilder() { return {}; }

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

  /// Function handle in the transform sequence.
  Value funcHandle{};

private:
  std::string generateOpToTileTag() const {
    return llvm::formatv(kOpToTile.data(), getIdx(), opsToTileAndFuse.size())
        .str();
  }

  void recordLazyAction(std::function<void()> action) {
    lazyActions.push_back(std::move(action));
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

  /// A collection of undo actions for rollback.
  std::vector<std::function<void()>> undoActions;
};

//===----------------------------------------------------------------------===//
// Cube Loop Information.
//===----------------------------------------------------------------------===//

class CubeBranch {
public:
  CubeBranch(size_t branchIdx, hivm::FixpipeOp fixpipe, hivm::MmadL1Op mmad)
      : branchIdx(branchIdx), fixpipe(fixpipe), mmad(mmad) {}

  size_t getBranchIdx() const { return branchIdx; }
  hivm::FixpipeOp getFixpipe() const { return fixpipe; }
  hivm::MmadL1Op getMmad() const { return mmad; }
  ArrayRef<Operation *> getProducers() const { return producers; }

  void addProducer(Operation *op) { producers.push_back(op); }

  std::string getTileTag(size_t loopIdx) const {
    return llvm::formatv("op_to_tile_{0}_branch_{1}", loopIdx, branchIdx).str();
  }

private:
  size_t branchIdx;
  hivm::FixpipeOp fixpipe;
  hivm::MmadL1Op mmad;
  SmallVector<Operation *, 4> producers;
};

class CubeLoopGroup {
public:
  CubeLoopGroup(size_t groupIdx) : groupIdx(groupIdx) {}

  ArrayRef<CubeBranch> getBranches() const { return branches; }
  MutableArrayRef<CubeBranch> getBranchesMutable() { return branches; }
  const TilingParams &getGroupTilingParams() const { return groupTilingParams; }

  void addBranch(CubeBranch branch) { branches.push_back(std::move(branch)); }
  void setGroupTilingParams(TilingParams params) {
    groupTilingParams = std::move(params);
  }

  std::string getGroupProducerTag(size_t loopIdx) const {
    return llvm::formatv("cube_producer_to_fuse_{0}_group_{1}", loopIdx,
                         groupIdx)
        .str();
  }

private:
  size_t groupIdx;
  SmallVector<CubeBranch, 2> branches;
  TilingParams groupTilingParams;
};

class CubeLoopInfo : public LoopInfo {
public:
  CubeLoopInfo(size_t idx, int64_t targetTripCount)
      : LoopInfo(idx, targetTripCount, hivm::TCoreType::CUBE) {}

  static bool classof(const LoopInfo *T) {
    return T->getLoopType() == hivm::TCoreType::CUBE;
  }

  void setTileCubeAttr(bool hasAttr) { hasTileCubeAttr = hasAttr; }
  bool getTileCubeAttr() { return hasTileCubeAttr; }

  void addLoopGroup(CubeLoopGroup group) { groups.push_back(std::move(group)); }
  ArrayRef<CubeLoopGroup> getLoopGroups() const { return groups; }

  transform_utils::SequenceBodyBuilderFn getBodyBuilder() override {
    return [this](OpBuilder &builder, Location loc, BlockArgument ba) {
      this->funcHandle = transform_utils::getFuncHandle(builder, loc, ba);

      for (const auto &group : groups) {
        SmallVector<Value> tiledLoopHandles;

        // Tiling all Fixpipes within the group
        for (const auto &branch : group.getBranches()) {
          auto opHandle = transform_utils::getMatchHandle(
              builder, loc, ba, branch.getTileTag(this->getIdx()));
          auto tiledHandle = transform_utils::createTileUsingForOp(
              builder, loc, opHandle, group.getGroupTilingParams().tileSizes);
          tiledLoopHandles.emplace_back(tiledHandle.getLoops().front());
        }

        Value fusedLoop;
        if (tiledLoopHandles.size() == 1) {
          fusedLoop = tiledLoopHandles.front();
        } else {
          // Fusing all base fixpipes in the group
          transform_utils::applyCleanUpPatterns(builder, loc, this->funcHandle);
          fusedLoop =
              transform_utils::fuseLoops(builder, loc, tiledLoopHandles);
        }

        // Fusing all producer in the group
        auto groupProducersToFuse = transform_utils::getMatchHandle(
            builder, loc, ba, group.getGroupProducerTag(this->getIdx()),
            /*reverse=*/true);

        this->createForEachFuseIntoBlock(builder, loc, groupProducersToFuse,
                                         fusedLoop);
      }

      transform_utils::applyCleanUpPatterns(builder, loc, this->funcHandle);
      builder.create<transform::YieldOp>(loc);
    };
  }

private:
  bool hasTileCubeAttr{false};
  SmallVector<CubeLoopGroup, 2> groups;
};

//===----------------------------------------------------------------------===//
// Vector Loop Information.
//===----------------------------------------------------------------------===//

/// Pattern to remove the dummy store.
class RemoveDummyStore : public OpRewritePattern<hivm::StoreOp> {
public:
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;

  RemoveDummyStore(MLIRContext *ctx, ArrayRef<OpToTile> maybeDummyStore)
      : OpRewritePattern<hivm::StoreOp>(ctx), maybeDummyStore(maybeDummyStore) {}

  LogicalResult matchAndRewrite(hivm::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (!storeOp->hasAttr(kDummyStore))
      return failure();

    // This is a module-level rewrite pattern, but we only want to erase the 
    // dummy store in the current vector loop.
    if (llvm::none_of(maybeDummyStore, [&storeOp](const OpToTile &opInfo) {
          return storeOp->hasAttr(opInfo.tag);
        }))
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

private:
  ArrayRef<OpToTile> maybeDummyStore;
};

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
    patterns.add<RemoveDummyStore>(ctx, this->getOpTileInfo());
    return applyPatternsGreedily(module, std::move(patterns));
  }

  transform_utils::SequenceBodyBuilderFn getBodyBuilder() override {
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
  template <typename OpType>
  WalkResult processCandidateLoop(OpType candidateLoop);
  template <typename OpType> LogicalResult collectCubeLoopInfo(OpType cubeLoop);
  template <typename OpType>
  LogicalResult collectVectorLoopInfo(OpType vectorLoop);

  // Obtains independent cube calculation branch.
  template <typename OpType>
  LogicalResult extractCubeBranches(hivm::detail::DimensionAnalyzer &analyzer,
                                    OpType cubeLoop,
                                    SmallVectorImpl<CubeBranch> &allBranches);

  // Extract the common axis of the branch for fusion.
  LogicalResult
  groupBranchesByCommonAxis(hivm::detail::DimensionAnalyzer &analyzer,
                            ArrayRef<CubeBranch> allBranches,
                            SmallVectorImpl<CubeLoopGroup> &loopGroups,
                            int64_t targetTripCount, bool hasTileCubeAttr);
  std::optional<TilingParams>
  calculateGroupTiling(hivm::detail::DimensionAnalyzer &analyzer,
                       ArrayRef<CubeBranch> branches, int64_t targetTripCount,
                       bool hasTileCubeAttr);
  bool canFitBranchInBuffer(ArrayRef<CubeBranch> branches,
                            int64_t targetTripCount, bool hasTileCubeAttr);

  /// Check if L0C-resident accumulation requires skipping tiling.
  bool hasL0CAccumulation(ArrayRef<CubeBranch> allBranches,
                          int64_t targetTripCount, bool hasTileCubeAttr);

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
template <typename TerminateType>
LogicalResult
tryCollectTilingInfoForTerminate(TerminateType terminateOp, Value yieldedVal,
                                 VectorLoopInfo &info,
                                 hivm::detail::DimensionAnalyzer &analyzer) {
  if (!isRankedTensor(yieldedVal.getType())) {
    LDBG("Yielding non-tensor values, skip");
    return failure();
  }

  // Try to find the immediate HIVM producer
  // Note: Currently we assume that all vector computations are done in
  // tensor.
  SmallVector<Operation *> trace = {terminateOp, yieldedVal.getDefiningOp()};
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
  Value storeResult = dummyStore.getResult(0);
  SetVector<Operation *> chainOfUsersToYield = {trace.begin(), trace.end()};
  singleResult.replaceUsesWithIf(
      storeResult,
      /*shouldReplace=*/[chainOfUsersToYield](OpOperand &operand) -> bool {
        return chainOfUsersToYield.contains(operand.getOwner());
      });

  // Record undo: erase the new init op.
  Operation *newInitOp = newInit.getDefiningOp();
  info.recordUndoAction([newInitOp]() { newInitOp->erase(); });
  // Record undo: erase the dummy store node.
  Operation *dummyStoreOp = dummyStore.getOperation();
  info.recordUndoAction([dummyStoreOp]() { dummyStoreOp->erase(); });
  // Record undo: restore the original init operand.
  Operation *hivmOpOp = hivmOp.getOperation();
  info.recordUndoAction([hivmOpOp, originalInit]() {
    cast<HIVMStructuredOp>(hivmOpOp).setDpsInitOperand(0, originalInit);
  });
  // Record undo: restore the replaced uses.
  info.recordUndoAction([storeResult, singleResult, chainOfUsersToYield]() {
    const_cast<Value &>(storeResult).replaceUsesWithIf(
        singleResult,
        /*shouldReplace=*/[chainOfUsersToYield](OpOperand &operand) -> bool {
          return chainOfUsersToYield.contains(operand.getOwner());
    });
  });

  ShapedType dstType = dummyStore.getDstOperandType();
  info.recordOpToTile(dummyStore, getVectorTiling(dstType, maybeTilingDim,
                                                  info.getTripCount()));
  // This store is a dummy one, mark it so that we can erase it later.
  info.lazySetAttr(dummyStore, kDummyStore.str(),
                   UnitAttr::get(dummyStore->getContext()));
  return success();
}

template <typename OpType>
WalkResult TileCubeVectorLoopPass::processCandidateLoop(OpType candidateLoop) {
  auto maybeLoopCoreType =
      candidateLoop->template getAttrOfType<hivm::TCoreTypeAttr>(
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
}

void TileCubeVectorLoopPass::collectLoopInfo(ModuleOp topLevelModule) {
  topLevelModule->walk([this](scf::ForOp candidateLoop) {
    return processCandidateLoop(candidateLoop);
  });
  topLevelModule->walk([this](scope::ScopeOp candidateLoop) {
    return processCandidateLoop(candidateLoop);
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
    bitUsed = bitUsed * static_cast<int>(resultType.getElementTypeBitWidth());

    // Determine alignment size based on element type because template supports
    // i1 type which does not require 32-byte alignment.
    int64_t actualAlignSize = alignSize;
    if (resultType.getElementType().isInteger(1)) {
      actualAlignSize = 8; // 8 bits
    }

    if (bitUsed % actualAlignSize != 0)
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
template <typename OpType>
LogicalResult TileCubeVectorLoopPass::collectVectorLoopInfo(OpType vectorLoop) {
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
  VectorLoopInfo tempInfo(info);
  if (!isa<scf::ForOp>(vectorLoop) && !isa<scope::ScopeOp>(vectorLoop)) {
    return vectorLoop.emitOpError(
        "Collect vector loop tiling info only support ForOp and ScopeOp.");
  } else {
    auto terminateOp = vectorLoop.getBody()->getTerminator();
    for (auto terminateOpVal : terminateOp->getOperands()) {
      if (failed(tryCollectTilingInfoForTerminate(terminateOp, terminateOpVal,
                                                  tempInfo, analyzer))) {
        tempInfo.rollback();
        return vectorLoop.emitOpError(
            "Failed to collect vector loop tiling info");
      }
    }
  }

  if (tempInfo.getOpTileInfo().empty()) {
    tempInfo.rollback();
    return failure();
  }

  // Finish collecting all info
  info = std::move(tempInfo);
  info.commitLazyActions();
  loopsToTile.emplace_back(std::make_unique<VectorLoopInfo>(info));
  return success();
}

template <typename OpType>
std::optional<int64_t> getMixCubeLoopNumber(OpType cubeLoop,
                                            CubeLoopInfo &info) {
  std::optional<int64_t> tileCubeLoopNum;
  cubeLoop->walk([&tileCubeLoopNum](hivm::MmadL1Op op) {
    if (op->hasAttr(hivm::TileMixCubeNumAttr::name)) {
      IntegerAttr attr =
          op->getAttrOfType<IntegerAttr>(hivm::TileMixCubeNumAttr::name);
      if (attr) {
        op.emitWarning("cube loop trip count will depend on attr instead of "
                       "compile option");
        tileCubeLoopNum = attr.getInt();
      }
    }
    return WalkResult::interrupt();
  });
  return tileCubeLoopNum;
}

// Extract independent cube computation branch.
template <typename OpType>
LogicalResult TileCubeVectorLoopPass::extractCubeBranches(
    hivm::detail::DimensionAnalyzer &analyzer, OpType cubeLoop,
    SmallVectorImpl<CubeBranch> &allBranches) {

  SmallVector<hivm::FixpipeOp, 4> fixpipeOps;
  cubeLoop->walk(
      [&](hivm::FixpipeOp fixpipeOp) { fixpipeOps.push_back(fixpipeOp); });

  if (fixpipeOps.empty()) {
    return success();
  }

  auto collectProducers = [&](Value operand, CubeBranch &branch) {
    SmallVector<Value> worklist = {operand};
    llvm::SmallPtrSet<Operation *, 4> visited;

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      Operation *defOp = current.getDefiningOp();

      if (!defOp || !cubeLoop->isProperAncestor(defOp))
        continue;
      if (!visited.insert(defOp).second)
        continue;
      branch.addProducer(defOp);

      if (auto loadOp = dyn_cast<hivm::LoadOp>(defOp)) {
        if (loadOp.hasPureBufferSemantics())
          continue;
        if (Operation *srcDef = loadOp.getSource().getDefiningOp()) {
          branch.addProducer(srcDef);
        }
        if (Operation *dstDef = loadOp.getTarget().getDefiningOp()) {
          branch.addProducer(dstDef);
        }
        continue;
      }

      for (Value opnd : defOp->getOperands()) {
        worklist.push_back(opnd);
      }
    }
  };

  size_t branchIdx = 0;
  for (auto fixpipeOp : fixpipeOps) {
    std::optional<Operation *> maybeMmad =
        traceDefOp<hivm::MmadL1Op>(fixpipeOp.getSource());
    auto mmadL1 = dyn_cast_or_null<hivm::MmadL1Op>(maybeMmad.value_or(nullptr));

    if (!mmadL1) {
      return fixpipeOp.emitOpError(
          "Cannot find matmul producer for fixpipe in this branch");
    }

    CubeBranch branch(branchIdx++, fixpipeOp, mmadL1);
    int64_t branchAxisId = analyzer.getGlobalTilingAxisId(fixpipeOp.getSrc());

    if (analyzer.getGlobalTilingAxisId(mmadL1.getA()) == branchAxisId) {
      collectProducers(mmadL1.getA(), branch);
    }

    if (analyzer.getGlobalTilingAxisId(mmadL1.getB()) == branchAxisId) {
      collectProducers(mmadL1.getB(), branch);
    }

    allBranches.push_back(std::move(branch));
  }

  return success();
}

std::optional<TilingParams> TileCubeVectorLoopPass::calculateGroupTiling(
    hivm::detail::DimensionAnalyzer &analyzer, ArrayRef<CubeBranch> branches,
    int64_t targetTripCount, bool hasTileCubeAttr) {

  if (branches.empty())
    return std::nullopt;

  Value firstSrc = branches.front().getFixpipe().getSrc();
  int64_t commonDimIdx = analyzer.getTilingDim(firstSrc);
  if (commonDimIdx == -1) {
    LDBG("No valid tiling dimension found for fixpipe src. Aborting fusion to "
         "prevent invalid memory access.");
    return std::nullopt;
  }

  auto firstDstType = branches.front().getFixpipe().getDstOperandType();
  if (!firstDstType.hasStaticShape()) {
    LDBG("Fixpipe dst doesn't have static shape.");
    return std::nullopt;
  }

  auto firstShape = firstDstType.getShape();
  int64_t commonDimSize = firstShape[commonDimIdx];
  for (const auto &branch : branches) {
    auto shape = branch.getFixpipe().getDstOperandType().getShape();
    if (shape[commonDimIdx] != commonDimSize) {
      LDBG("Branch shape mismatch on chosen common axis. Fusion impossible.");
      return std::nullopt;
    }
  }

  TilingParams params;
  params.tiledDims = {static_cast<int>(commonDimIdx)};
  params.tileSizes = SmallVector<int64_t>(firstShape.size(), 0);
  params.tileSizes[commonDimIdx] =
      llvm::divideCeilSigned(commonDimSize, targetTripCount);

  return params;
}

// Check whether the fixpipe destination data for all branches can fit
// entirely within the L0C buffer.  For a single branch the individual size
// is compared; for multiple branches the total size is summed.
bool TileCubeVectorLoopPass::canFitBranchInBuffer(ArrayRef<CubeBranch> branches,
                                                  int64_t targetTripCount,
                                                  bool hasTileCubeAttr) {
  if (branches.empty() || targetTripCount == 1 || hasTileCubeAttr)
    return false;

  const int64_t kL0CSizeInBits = hacc::utils::getIntegerSpecValue(
      this->spec.getSpecForIdentifierEnum(hacc::DeviceSpec::L0C_SIZE));

  auto getSizeBits = [&](const CubeBranch &branch) -> std::optional<int64_t> {
    auto dstType = branch.getFixpipe().getDstOperandType();
    if (!dstType.hasStaticShape())
      return std::nullopt;
    return mlir::utils::getStaticTotalSizeInBits(dstType.getShape(),
                                                 dstType.getElementType());
  };

  if (branches.size() == 1) {
    auto maybeSize = getSizeBits(branches.front());
    if (!maybeSize.has_value())
      return false;
    if (maybeSize.value() <= kL0CSizeInBits) {
      return true;
    }
    return false;
  }

  // Multi-branch: sum the sizes of all fixpipe destinations.
  int64_t totalSizeBits = 0;
  for (const auto &branch : branches) {
    auto maybeSize = getSizeBits(branch);
    if (!maybeSize.has_value())
      return false;
    totalSizeBits += maybeSize.value();
  }
  if (totalSizeBits <= kL0CSizeInBits) {
    return true;
  }
  return false;
}

// Check whether the cube loop has L0C-resident MmadL1 accumulation, which
// requires skipping tiling to preserve correct partial-result accumulation.
bool TileCubeVectorLoopPass::hasL0CAccumulation(
    ArrayRef<CubeBranch> allBranches, int64_t targetTripCount,
    bool hasTileCubeAttr) {

  for (const auto &branch : allBranches) {
    auto mmad = branch.getMmad();
    if (!mmad)
      continue;
    Value accFlag = mmad.getOperand(2);
    if (!accFlag)
      continue;
    if (!matchPattern(accFlag, m_Constant())) {
      LDBG("MmadL1 has non-constant accumulate flag, L0C accumulation"
           " detected");
      return true;
    }
  }

  return false;
}

// Branches that share same tiling axis are merged into same CubeLoopGroup, so
// that they can fused into the same sub loop.
LogicalResult TileCubeVectorLoopPass::groupBranchesByCommonAxis(
    hivm::detail::DimensionAnalyzer &analyzer, ArrayRef<CubeBranch> allBranches,
    SmallVectorImpl<CubeLoopGroup> &loopGroups, int64_t targetTripCount,
    bool hasTileCubeAttr) {

  DenseMap<int64_t, SmallVector<CubeBranch, 2>> axisIdToGroups;
  SmallVector<CubeBranch, 2> independentBranches;

  for (const auto &branch : allBranches) {
    Value fixpipeSrc = branch.getFixpipe().getSrc();
    int64_t globalAxisId = analyzer.getGlobalTilingAxisId(fixpipeSrc);

    if (globalAxisId == -1) {
      LDBG("Warning: Branch fixpipe lacks a valid global tiling axis.");
      branch.getFixpipe().emitWarning(
          "Failed to find a spatial global axis for tiling.");
      return failure();
    }

    axisIdToGroups[globalAxisId].push_back(branch);
  }

  auto createGroup = [&](ArrayRef<CubeBranch> branches) -> LogicalResult {
    if (branches.empty())
      return success();

    CubeLoopGroup group(loopGroups.size());
    for (auto &b : branches)
      group.addBranch(b);

    std::optional<TilingParams> params = calculateGroupTiling(
        analyzer, branches, targetTripCount, hasTileCubeAttr);
    if (!params) {
      LDBG("Failed to calculate tiling params for group.");
      return failure();
    }

    group.setGroupTilingParams(std::move(*params));
    loopGroups.push_back(std::move(group));
    return success();
  };

  for (auto &it : axisIdToGroups) {
    if (failed(createGroup(it.second)))
      return failure();
  }

  return success();
}

/// Collect tiling and fusion information for Cube loops.
///
/// Expected IR Structure (Multi-Branch Scenario):
/// ```mlir
///   // --- Branch 1 ---
///   %a1 = hivm.hir.load ins(%global_a) outs(%buf_a)
///   %b1 = hivm.hir.load ins(%global_b1) outs(%buf_b1)
///   %mmad1 = hivm.hir.mmadL1 ins(%a1, %b1)
///   hivm.hir.fixpipe ins(%mmad1) outs(%out1)
///
///   // --- Branch 2 ---
///   // May share inputs or just share the same tiling axis with Branch 1
///   %b2 = hivm.hir.load ins(%global_b2) outs(%buf_b2)
///   %mmad2 = hivm.hir.mmadL1 ins(%a1, %b2)
///   hivm.hir.fixpipe ins(%mmad2) outs(%out2)
/// ```
/// This function analyzes structured Cube operations within a loop, extracts
/// independent compute branches, groups them by their common tiling axis,
/// and tags them for downstream Transform Dialect tiling and fusion.
template <typename OpType>
LogicalResult TileCubeVectorLoopPass::collectCubeLoopInfo(OpType cubeLoop) {
  CubeLoopInfo info(loopsToTile.size(),
                    static_cast<int64_t>(this->tiledMixCubeLoopNumber));

  auto tileCubeLoopNum = getMixCubeLoopNumber(cubeLoop, info);
  if (tileCubeLoopNum.has_value()) {
    info.setTileCubeAttr(true);
    info.setTripCount(tileCubeLoopNum.value());
  }

  if (info.getTripCount() == 1)
    return success();

  hivm::detail::DimensionAnalyzer analyzer(cubeLoop.getOperation());
  if (failed(analyzer.initialize()))
    return failure();
  analyzer.computeTilingDim(false);

  SmallVector<CubeBranch, 4> allBranches;
  if (failed(extractCubeBranches(analyzer, cubeLoop, allBranches))) {
    return cubeLoop.emitOpError("Failed to extract compute branches");
  }
  if (allBranches.empty())
    return success();

  // If the cube loop uses L0C-resident MmadL1 accumulation, tiling would
  // break the partial-result accumulation pattern.
  if (hasL0CAccumulation(allBranches, info.getTripCount(),
                         info.getTileCubeAttr())) {
    LDBG("Skipping cube loop tiling: L0C accumulation detected");
    info.setTripCount(1);
    return success();
  }

  if (canFitBranchInBuffer(allBranches, info.getTripCount(),
                           info.getTileCubeAttr())) {
    LDBG("No need to tile because sum of branches can fit on L0C");
    info.setTripCount(1);
    return success();
  }

  SmallVector<CubeLoopGroup, 2> loopGroups;
  if (failed(groupBranchesByCommonAxis(analyzer, allBranches, loopGroups,
                                       info.getTripCount(),
                                       info.getTileCubeAttr()))) {
    return cubeLoop.emitOpError("Failed to resolve axis and grouping");
  }

  auto fuseTagAttr = UnitAttr::get(&getContext());

  auto isFromEntryBlock = [](hivm::LoadOp loadOp) -> bool {
    auto maybeReinterpretCastOp =
        traceDefOp<memref::ReinterpretCastOp>(loadOp.getSource());
    if (maybeReinterpretCastOp.has_value()) {
      if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(
              maybeReinterpretCastOp.value())) {
        if (auto blockArg =
                dyn_cast<mlir::BlockArgument>(castOp.getViewSource())) {
          return blockArg.getOwner()->isEntryBlock();
        }
      }
    }
    return false;
  };

  for (auto &group : loopGroups) {
    for (auto &branch : group.getBranchesMutable()) {

      info.recordOpToTile(branch.getFixpipe(), branch.getTileTag(info.getIdx()),
                          TilingParams(group.getGroupTilingParams()));

      std::string groupTag = group.getGroupProducerTag(info.getIdx());
      info.lazySetAttr(branch.getMmad(), groupTag, fuseTagAttr);

      // If Load does not have split axis, no need to pull it into tile loop
      int64_t groupAxisId =
          analyzer.getGlobalTilingAxisId(branch.getFixpipe().getSrc());
      for (Operation *op : branch.getProducers()) {
        if (auto loadOp = dyn_cast<hivm::LoadOp>(op)) {
          int64_t loadAxisId =
              analyzer.getGlobalTilingAxisId(loadOp.getResult(0));
          if (loadAxisId != groupAxisId) {
            if (info.getTileCubeAttr() && isFromEntryBlock(loadOp)) {
              LDBG("Skipping fuse tag for global weight LoadOp: " << loadOp);
              continue;
            }
          }
        }

        info.lazySetAttr(op, groupTag, fuseTagAttr);
      }
    }
    info.addLoopGroup(std::move(group));
  }

  info.commitLazyActions();
  loopsToTile.emplace_back(std::make_unique<CubeLoopInfo>(std::move(info)));
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

  // Return early if cv-pipelining is not enabled.
  bool hasPipelinedLoops = false;
  topLevelModule->walk([&hasPipelinedLoops](Operation *op) {
    if (isa<scf::ForOp, scope::ScopeOp>(op) &&
        op->hasAttr(hivm::kPipelinedLoopCoreTypeAttrName)) {
      hasPipelinedLoops = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!hasPipelinedLoops)
    return;

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