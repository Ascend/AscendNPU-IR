//===--------- AutoVectorizeV2.cpp - Auto vectorization pass
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/VFFusion/Utils.h"
#include "bishengir/Dialect/Analysis/VFFusion/VFStackInfo.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Attrs.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/PlanContext.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoVectorize/Verify.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <memory>

#define DEBUG_TYPE "hfusion-auto-vectorize-v2"

namespace mlir {
#define GEN_PASS_DEF_AUTOVECTORIZEV2
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::hfusion;
using mlir::hfusion::RetriedOptions;

namespace {

static bool isCubeScopeOp(Operation *op) {
  auto scopeOp = dyn_cast<scope::ScopeOp>(op);
  if (!scopeOp)
    return false;

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!attr)
    return false;

  return attr.getTcoretype() == mlir::hivm::TCoreType::CUBE;
}

static bufferization::ToTensorOp traceToTensorSource(Value value) {
  DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    if (auto toTensor = value.getDefiningOp<bufferization::ToTensorOp>())
      return toTensor;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return {};

    bool isTensorView =
        isa<tensor::ExtractSliceOp, tensor::CastOp, tensor::CollapseShapeOp,
            tensor::ExpandShapeOp>(defOp);
    bool isSingleOperandCast = isa<UnrealizedConversionCastOp>(defOp) &&
                               defOp->getNumOperands() == 1 &&
                               defOp->getNumResults() == 1;
    if (!isTensorView && !isSingleOperandCast)
      return {};

    value = defOp->getOperand(0);
  }
  return {};
}

static Attribute findWasBoolToInt8WriterAttr(Value memref, Operation *readOp,
                                             DominanceInfo &domInfo) {
  for (OpOperand &use : memref.getUses()) {
    Operation *user = use.getOwner();
    if (!domInfo.properlyDominates(user, readOp))
      continue;

    Attribute attr = user->getAttr(utils::wasBoolToInt8);
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user);
    if (attr && dpsOp && dpsOp.isDpsInit(&use))
      return attr;
  }

  return {};
}

static void propagateWasBoolToInt8ToTransferReads(func::FuncOp func) {
  DominanceInfo domInfo(func);
  func.walk([&](vector::TransferReadOp readOp) {
    if (readOp->hasAttr(utils::wasBoolToInt8) ||
        !readOp.getVectorType().getElementType().isInteger(8))
      return;

    auto toTensor = traceToTensorSource(readOp.getSource());
    if (!toTensor)
      return;

    Value memref = toTensor.getMemref();
    Attribute attr =
        findWasBoolToInt8WriterAttr(memref, readOp.getOperation(), domInfo);
    if (attr)
      readOp->setAttr(utils::wasBoolToInt8, attr);
  });
}

/// If two fusable ops are conflict with each other, they cannot be fused into
/// the same VF:
/// 1. The producer(upstream op) and consumer(downstream op) of
///    NonVectorizableOp are confilict with each other. For example:
///       A(FusableOp)
///       B(NonVectorizableOp, use A)
///       C(FusableOp, use B)
///    Then A and C are confilict with each other
/// 2. The previous and following of scf.for/hivm.hir.sync_block_wait/
///    hivm.hir.sync_block_set are confilict with each other. For example:
///       A(FusableOp)
///       hivm.hir.sync_block_wait
///       B(FusableOp)
///    Then A and B are confilict with each other

// A fusable op is output node when its all users is NonVectorizableOp or
// terminator op.
static bool isFusableOutputNode(Operation *op, Block *block) {
  if (!isFusableOp(op))
    return false;
  if (op->getUsers().empty())
    return true;
  for (Operation *user : op->getUsers()) {
    if (isa<hfusion::AssertOp>(user))
      continue;
    if (isa<annotation::MarkOp>(user))
      continue;
    if (!(isNonVectorizableOp(user) || !isOpInBlock(user, block) ||
          isa<scf::YieldOp, func::ReturnOp, scope::ReturnOp>(user) ||
          isa<bufferization::BufferizationDialect>(user->getDialect())))
      return false;
  }
  return true;
}

static void interchangeForLeafNodes(SmallVector<int64_t> commonAxis,
                                    FusableOpInfo &leafNodeInfo) {
  leafNodeInfo.tileInterchange = commonAxis;
  for (unsigned i = 0; i < leafNodeInfo.numLoops; i++) {
    if (!llvm::is_contained(commonAxis, i)) {
      leafNodeInfo.tileInterchange.push_back(i);
    }
  }
}

static bool isProducerConsumedImpl(Operation *target, Operation *source,
                                   DenseSet<Value> &visited) {
  if (!target || !source) {
    return false;
  }
  for (Value targetResult : target->getResults()) {
    if (visited.contains(targetResult)) {
      continue;
    }
    visited.insert(targetResult);
    // dfs check if any of the target result users is consumed by source op
    bool consumed =
        llvm::any_of(targetResult.getUsers(), [&](Operation *resultUser) {
          return source->isAncestor(resultUser) ||
                 isProducerConsumedImpl(resultUser, source, visited);
        });
    if (consumed) {
      return true;
    }
  }
  return false;
}

static bool isProducerConsumed(Operation *target, Operation *source) {
  DenseSet<Value> visited;
  if (target->isBeforeInBlock(source)) {
    return isProducerConsumedImpl(target, source, visited);
  } else {
    return isProducerConsumedImpl(source, target, visited);
  }
}

// fuse sibling will clone all users of those front siblings behind fused loop
// which will cause existing handle lost or IR order changed. So we move those
// front siblings and their users to their new positions after fuse. For
// example: we have 4 nodes in order A B C(use A) D, and A and D should fuse
// sibling, then we move A before D and move C hehind D, after moving the order
// will be B A D C(use A).

/// Returns true if an op's results are used by "many" distinct users.
/// We count distinct owning operations across all result values.
static bool hasManyUsers(Operation *op, unsigned threshold = 2) {
  if (!op)
    return false;

  DenseSet<Operation *> users;
  for (Value res : op->getResults()) {
    for (OpOperand &use : res.getUses())
      users.insert(use.getOwner());
  }
  return users.size() >= threshold;
}

/// Pre validation for fusion opportunity of Linalg's
/// tileAndFuseFirstExtractUse.
///
/// Returns true if all consumers of the producer will fuse into a single loop.
/// When consumers fuse into different loops (different fusedNode labels), the
/// producer has no valid fusion opportunity and should remain a standalone op.
static bool hasFusionOpportunity(Operation *producer, PlanContext &ctx) {
  if (!producer)
    return false;

  auto tileableProducer = dyn_cast<TilingInterface>(producer);
  if (!tileableProducer)
    return false;

  if (llvm::any_of(tileableProducer->getUsers(), [&](Operation *user) {
        return !ctx.hasOpInfo(user) || !ctx.getInfo(user).fusedNode;
      })) {
    // Sanity check for nullptr
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "======== FusionOpportunity ========\n");
  LLVM_DEBUG(llvm::dbgs() << "producer: " << *producer << "\n");
  // If all consumers share the same fusedNode (loop), producer can fuse there.
  std::set<std::string> consumerLabels;
  llvm::for_each(tileableProducer->getUsers(), [&](Operation *user) {
    const std::string &fusedNodeLabel = ctx.getInfo(user).fusedNode->getLabel();
    LLVM_DEBUG(llvm::dbgs() << "fusedNodeLabel: " << fusedNodeLabel << "\n");
    consumerLabels.insert(fusedNodeLabel);
  });
  return consumerLabels.size() == 1;
}

// Normally, we should fuse the producer into the closest fusedNode which
// contains its consumers. But in some context, we should give up fusing and
// return nullptr:
// 1. the producer is fill op or vsstb pattern transpose op
// 2. there is fusedOps within the closest fusedNode conflict with the producer
// 3. there is normal(not vsstb pattern) transpose user within the closest
//    fusedNode
static std::shared_ptr<FusedNode>
findBestFusedNodeForProducer(Block *block, Operation *producer,
                             PlanContext &ctx, int64_t vectorLength) {
  // Since the producers and consumers of a transpose op have opposite axes, we
  // cannot fuse them into the same fusedNode. For vsstb pattern transpose, we
  // fuse the producers into this transpose op; for other pattern transpose, we
  // fuse this transpose op into its consumer ops. For example: if we have user
  // chain: op1 -> vsstb transpose -> op2, we will fuse op1 into transpose op;
  // if we have user chain: op1 -> normal transpose -> op2, we will fuse
  // transpose op into op2
  if (isVsstbPatternTransposeOp(producer))
    return nullptr;

  // A standalone loop would not yield the tiled intermediate back to other
  // consumers; they would still read the original untiled tensor, producing
  // wrong slices. Keep it in-place so every consumer extract_slices from the
  // full tensor directly.
  if (isExpandShapeOpCanFuseIntoVsstbPatternTranspose(producer) &&
      hasManyUsers(producer) && !hasFusionOpportunity(producer, ctx)) {
    return nullptr;
  }

  if (!ctx.getEnableMultipleConsumerFusion() && hasManyUsers(producer) &&
      !hasFusionOpportunity(producer, ctx)) {
    return nullptr;
  }

  Operation *closestUser = nullptr;
  for (auto user : producer->getUsers()) {
    if (isOpInBlock(user, block)) {
      if (!closestUser || user->isBeforeInBlock(closestUser))
        closestUser = user;
    }
  }
  if (!closestUser || !isFusableOp(closestUser))
    return nullptr;

  auto bestFusedNode = ctx.getInfo(closestUser).fusedNode;
  assert(bestFusedNode);
  if (!bestFusedNode->canAccept(producer, AcceptContext::Producer))
    return nullptr;
  FusableOpInfo &producerInfo = ctx.getInfo(producer);
  if (!bestFusedNode->canFuseProducer(producer))
    return nullptr;
  int numUsersInBestFusedNode = 0;
  for (auto user : DenseSet<Operation *>(producer->getUsers().begin(),
                                         producer->getUsers().end())) {
    if (bestFusedNode->contains(user)) {
      numUsersInBestFusedNode++;
      if (isa<linalg::TransposeOp>(user) && !isVsstbPatternTransposeOp(user))
        return nullptr;
    }
  }

  if (numUsersInBestFusedNode > 1) {
    AffineMap map;
    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          if (!map) {
            map = consumerLinalgOp.getMatchingIndexingMap(&use);
          } else if (map != consumerLinalgOp.getMatchingIndexingMap(&use)) {
            return nullptr;
          }
        }
      }
    }
  }
  // If producer is reduction and user is broadcast, only fuse last-axis
  // reduction producer into last-axis broadcast op. Other fusion context will
  // generate memref.alloc inside VF and cannot be outlined, so give up fusing,
  // see issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengCompiler/AscendNPU-IR/issues/638
  if (producerInfo.numReductionLoops > 0) {
    bool producerIsNonLastAxisReduce = false;
    auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
    SmallVector<unsigned> reductionDims;
    producerLinalgOp.getReductionDims(reductionDims);
    if (reductionDims[0] != producerInfo.numLoops - 1)
      producerIsNonLastAxisReduce = true;

    for (OpOperand &use : producer->getUses()) {
      Operation *user = use.getOwner();
      if (bestFusedNode->contains(user)) {
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(user)) {
          AffineMap map = consumerLinalgOp.getMatchingIndexingMap(&use);
          if (map.getNumResults() < map.getNumDims()) {
            // last-axis broadcast, the affine map will be
            // `affine_map<(d0,d1)->(d0)>` non-last-axis broadcast, the affine
            // map will be `affine_map<(d0,d1)->(d1)>`
            for (auto indexAndResult : llvm::enumerate(map.getResults()))
              if (auto d = dyn_cast<AffineDimExpr>(indexAndResult.value()))
                if (d.getPosition() != indexAndResult.index())
                  return nullptr;
            if (producerIsNonLastAxisReduce)
              return nullptr;
          }
        }
      }
    }
  }

  return bestFusedNode;
}

/// For reduction op, tile_reduction_using_for has better performance than
/// tile_using_for. Firstly we should tile parallel axis by tile_using_for,
/// then tile reduction axis by tile_reduction_using_for.
static Value
tileReductionOp(OpBuilder &builder, transform::SequenceOp seqOp, Operation *op,
                Value &linalgOpHandle, SmallVector<int64_t> tileSize,
                std::string label,
                SmallVector<std::pair<std::string, SmallVector<int64_t>>>
                    &otherVectorizableOps) {
  assert(isa<linalg::LinalgOp>(op));
  auto reductionOp = cast<linalg::LinalgOp>(op);
  assert(reductionOp.getNumParallelLoops() > 0);
  assert(reductionOp.getNumReductionLoops() == 1);
  auto loc = seqOp->getLoc();
  // get parallel axis and tile.
  SmallVector<unsigned> parallelDims;
  reductionOp.getParallelDims(parallelDims);
  SmallVector<int64_t> parallelAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : parallelDims)
    parallelAxisTileSize[i] = tileSize[i];
  transform::TileUsingForOp parallelAxisTilingResult =
      builder.create<transform::TileUsingForOp>(loc, linalgOpHandle,
                                                parallelAxisTileSize);
  // get reduction axis and tile.
  SmallVector<unsigned> reductionDims;
  reductionOp.getReductionDims(reductionDims);
  SmallVector<int64_t> reductionAxisTileSize(reductionOp.getNumLoops(), 0);
  for (auto i : reductionDims)
    reductionAxisTileSize[i] = tileSize[i];
  transform::TileReductionUsingForOp reductionAxisTilingResult =
      builder.create<transform::TileReductionUsingForOp>(
          loc, parallelAxisTilingResult.getTiledLinalgOp(),
          reductionAxisTileSize);
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getForOp(), "reductionLoop", nullptr);
  // fillOp and combiningLinalgOp should also be vectorized, here we add them
  // into otherVectorizableOps which will be vectorized after tiling and
  // fusing.
  for (auto it : llvm::enumerate(reductionAxisTilingResult.getFillOp())) {
    Value fillOp = it.value();
    std::string fillOpLabel = label + "_fill_" + std::to_string(it.index());
    builder.create<transform::AnnotateOp>(loc, fillOp, fillOpLabel, nullptr);
    otherVectorizableOps.push_back(std::make_pair(fillOpLabel, tileSize));
  }
  std::string splitLinalgOpLabel = label + "_split";
  builder.create<transform::AnnotateOp>(
      loc, reductionAxisTilingResult.getSplitLinalgOp(), splitLinalgOpLabel,
      nullptr);
  otherVectorizableOps.push_back(std::make_pair(splitLinalgOpLabel, tileSize));
  // If combiningLinalgOp(is a linalg.reduce op) has dyncamic shape, it
  // cannot be vectorized, so we convert it to a linalg.generic op.
  Value generalizedCombiningLinalgOp = builder.create<transform::GeneralizeOp>(
      loc, builder.getType<transform::AnyOpType>(),
      reductionAxisTilingResult.getCombiningLinalgOp());
  std::string combiningLinalgOpLabel = label + "_combining";
  builder.create<transform::AnnotateOp>(loc, generalizedCombiningLinalgOp,
                                        combiningLinalgOpLabel, nullptr);
  otherVectorizableOps.push_back(
      std::make_pair(combiningLinalgOpLabel, tileSize));
  return parallelAxisTilingResult.getLoops().front();
}

static void
collectFusableFuncInModule(ModuleOp moduleOp,
                           SmallVector<func::FuncOp> &fusableFuncList) {
  moduleOp->walk([&](func::FuncOp func) {
    auto fusionKind = mlir::hfusion::tryGetFusionKind(func);
    if (hacc::utils::isDevice(func)) {
      if (fusionKind.has_value() &&
          (fusionKind.value() == mlir::hfusion::FusionKind::ShallowCV ||
           fusionKind.value() == mlir::hfusion::FusionKind::SingleCube)) {
        // Skip this for now
      } else
        fusableFuncList.push_back(func);
    }
  });
}

class AutoVectorizeV2 : public impl::AutoVectorizeV2Base<AutoVectorizeV2> {
public:
  explicit AutoVectorizeV2(const AutoVectorizeV2Options &options)
      : AutoVectorizeV2Base(options) {}
  void runOnOperation() override;

private:
  void buildTileAndFuseTransformSequenceForBlock(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps,
      PlanContext &result);
  void buildVectorizeTransformSequence(
      OpBuilder &builder, transform::SequenceOp seqOp,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps,
      PlanContext &result);
  Value getOpTransformHandle(std::string label, OpBuilder &builder,
                             transform::SequenceOp seqOp);
  void planFuseSiblingForLeafNodes(Block *block, PlanContext &result);
  void planFuseProducersIntoConsumers(Block *block, PlanContext &result);
  void planFuseProducerIntoFusedNode(Block *block, Operation *producer,
                                     PlanContext &result);
  void tileAndFuseSiblingForLeafNodes(
      OpBuilder &builder, transform::SequenceOp seqOp, PlanContext &result,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void fuseProducersIntoConsumers(
      OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
      PlanContext &result,
      SmallVector<std::pair<std::string, SmallVector<int64_t>>>
          &otherVectorizableOps);
  void applyCleanUp(OpBuilder &builder, transform::SequenceOp seqOp);
  void sortFunc(func::FuncOp func);
  transform::SequenceOp buildTransformSequence(func::FuncOp func,
                                               RetriedOptions &retryCtx,
                                               OpBuilder &builder);
  void emitTransformSequenceIR(func::FuncOp func, RetriedOptions &retryCtx,
                               OpBuilder &builder);
  LogicalResult runAttempt(func::FuncOp func, RetriedOptions &retryCtx,
                           OpBuilder &builder, IRRewriter &rewriter);
  LogicalResult vectorize(func::FuncOp func, RetriedOptions &retryCtx,
                          OpBuilder &builder);
};

void AutoVectorizeV2::planFuseSiblingForLeafNodes(Block *block,
                                                  PlanContext &ctx) {
  // Collect leafNodes in the block
  SmallVector<Operation *> leafNodes;
  block->walk([&](Operation *op) {
    if (isOpInBlock(op, block) && isFusableOutputNode(op, block)) {
      leafNodes.push_back(op);
    }
  });
  if (leafNodes.empty())
    return;
  // Group leafNodes, all leafNodes in the same group will be fused siblings
  for (auto leafNode : leafNodes) {
    if (ctx.size() == 0) {
      ctx.add(leafNode);
      continue;
    }
    if (isMemrefLinalgOp(leafNode)) {
      ctx.add(leafNode);
      continue;
    }

    bool isInserted = false;
    for (auto &node : ctx.nodes()) {
      if (!node->canAccept(leafNode))
        continue;
      node->addLeaf(leafNode);
      isInserted = true;
      break;
    }

    if (!isInserted) {
      ctx.add(leafNode);
    }
  }
}

// Determine how to fuse producers layer by layer.
void AutoVectorizeV2::planFuseProducersIntoConsumers(Block *block,
                                                     PlanContext &ctx) {
  std::queue<Operation *> queue;
  for (auto &node : ctx.nodes()) {
    for (Operation *leafNode : node->leafOps()) {
      queue.push(leafNode);
    }
  }
  while (!queue.empty()) {
    auto consumer = queue.front();
    queue.pop();
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (isFusableOp(producer)) {
        // It means that current producer has been handled if the value of
        // ctx.getInfo(producer).fusedNode is not nullptr.
        if (!isOpInBlock(producer, block) || ctx.getInfo(producer).fusedNode)
          continue;

        // If all users of the producer has been handled, we can fuse this
        // producer into proper fused node.
        if (llvm::all_of(producer->getUsers(), [&](Operation *user) {
              if (isFusableOp(user))
                return ctx.getInfo(user).fusedNode != nullptr;
              return true;
            })) {
          planFuseProducerIntoFusedNode(block, producer, ctx);
          queue.push(producer);
        }
      }
    }
  }
}

void AutoVectorizeV2::planFuseProducerIntoFusedNode(Block *block,
                                                    Operation *producer,
                                                    PlanContext &ctx) {
  FusableOpInfo &producerInfo = ctx.getInfo(producer);
  std::shared_ptr<FusedNode> bestFusedNode =
      findBestFusedNodeForProducer(block, producer, ctx, vectorLength);
  if (bestFusedNode) {
    bestFusedNode->addProducer(producer);

    // consumer leafNodes should interchange when tiling because of reduction
    // producer.
    if (producerInfo.numReductionLoops) {
      for (OpOperand &use : producer->getUses()) {
        Operation *user = use.getOwner();
        if (bestFusedNode->containsLeaf(user)) {
          // FIXME: here only find common axis for LinalgOp, also should find
          // for non LinalgOp(interleave, deinterleave...)
          if (auto consumer = dyn_cast<linalg::LinalgOp>(user)) {
            SmallVector<int64_t> commonAxis;
            AffineMap indexingMap = consumer.getMatchingIndexingMap(&use);
            for (AffineExpr e : indexingMap.getResults()) {
              if (auto d = dyn_cast<AffineDimExpr>(e)) {
                commonAxis.push_back(d.getPosition());
              }
            }
            for (Operation *leafNode : bestFusedNode->leafOps())
              interchangeForLeafNodes(commonAxis, ctx.getInfo(leafNode));
            break;
          }
        }
      }
    }
  } else {
    // expand_shape has no tile transform of its own — only vsstb-fused paths
    // can tile it.
    if (isExpandShapeOpCanFuseIntoVsstbPatternTranspose(producer))
      return;

    bool isInserted = false;
    for (auto &node : ctx.nodes()) {
      if (!node->canAccept(producer, AcceptContext::FallbackLeaf))
        continue;
      if (llvm::any_of(node->leafOps(), [&](Operation *otherLeafNode) {
            return isProducerConsumed(producer, otherLeafNode);
          }))
        continue;
      node->addLeaf(producer);
      isInserted = true;
      break;
    }
    if (!isInserted) {
      ctx.add(producer);
    }
  }
}

Value AutoVectorizeV2::getOpTransformHandle(std::string label,
                                            OpBuilder &builder,
                                            transform::SequenceOp seqOp) {
  DictionaryAttr opAttr = builder.getDictionaryAttr(
      builder.getNamedAttr(label, builder.getUnitAttr()));
  Value linalgOpHandle =
      builder
          .create<transform::MatchOp>(
              seqOp.getLoc(), builder.getType<transform::AnyOpType>(),
              seqOp.getBodyBlock()->getArguments().front(), ArrayAttr(),
              transform::MatchInterfaceEnumAttr{}, opAttr, DictionaryAttr{},
              TypeAttr{}, ArrayAttr{})
          .getResults();
  return linalgOpHandle;
}

void AutoVectorizeV2::tileAndFuseSiblingForLeafNodes(
    OpBuilder &builder, transform::SequenceOp seqOp, PlanContext &ctx,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp->getLoc();
  for (auto &node : ctx.nodes()) {
    SmallVector<Value> tiledLoopHandles;
    bool hasFillOp = false;
    for (Operation *leafNode : node->leafOps()) {
      if (mlir::hfusion::isFillOp(leafNode))
        hasFillOp = true;
      FusableOpInfo &leafNodeInfo = ctx.getInfo(leafNode);
      Value leafNodeHandle =
          getOpTransformHandle(leafNodeInfo.label, builder, seqOp);
      if (hfusion::shouldUseTileReductionUsingForV2(leafNode)) {
        tiledLoopHandles.push_back(tileReductionOp(
            builder, seqOp, leafNode, leafNodeHandle, leafNodeInfo.tileSize,
            leafNodeInfo.label, otherVectorizableOps));
      } else {
        transform::TileUsingForOp tilingResult =
            builder.create<transform::TileUsingForOp>(
                loc, leafNodeHandle, leafNodeInfo.tileSize,
                leafNodeInfo.tileInterchange);
        tiledLoopHandles.push_back(tilingResult.getLoops().front());
        if (isVsstbPatternTransposeOp(leafNode)) {
          assert(tilingResult.getLoops().size() >= 2);
          builder.create<transform::AnnotateOp>(
              loc, tilingResult.getLoops()[1],
              builder.getStringAttr("unroll_for_vsstb"), nullptr);
        }
      }
    }
    assert(!std::empty(tiledLoopHandles) && "Should fuse more than one loops");
    applyCleanUp(builder, seqOp);
    Value fusedLoopHandle = tiledLoopHandles.front();
    for (Value nextLoopHandle : llvm::drop_begin(tiledLoopHandles)) {
      fusedLoopHandle =
          builder
              .create<transform::LoopFuseSiblingOp>(
                  loc, builder.getType<transform::AnyOpType>(),
                  /*target=*/fusedLoopHandle, /*source=*/nextLoopHandle)
              .getFusedLoop();
      fusedLoopHandle = builder
                            .create<transform::LoopFuseNestedSiblingsOp>(
                                loc, builder.getType<transform::AnyOpType>(),
                                fusedLoopHandle, /*recursive=*/true)
                            .getTransformed();
    }
    if (hasFillOp)
      builder.create<transform::AnnotateOp>(loc, fusedLoopHandle,
                                            "outlinedLoopWithFill", nullptr);
    builder.create<transform::AnnotateOp>(loc, fusedLoopHandle,
                                          node->getLabel(), nullptr);
    if (tiledLoopHandles.size() > 1)
      applyCleanUp(builder, seqOp);
  }
}

void AutoVectorizeV2::fuseProducersIntoConsumers(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    PlanContext &ctx,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps) {
  auto loc = seqOp.getLoc();
  for (auto &node : ctx.nodes())
    for (Operation *producer : node->producerOps()) {
      FusableOpInfo &producerInfo = ctx.getInfo(producer);
      Value producerHandle =
          getOpTransformHandle(producerInfo.label, builder, seqOp);
      std::shared_ptr<FusedNode> fusedNode = producerInfo.fusedNode;
      Value containingLoopHandle =
          getOpTransformHandle(fusedNode->getLabel(), builder, seqOp);
      builder.create<transform::ApplyPatternsOp>(
          loc, containingLoopHandle, [](OpBuilder &innerBuilder, Location loc) {
            innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(
                loc);
          });
      builder.create<transform::MergeProducerExtractUsesOp>(
          loc, producerHandle, containingLoopHandle);
      transform::FuseIntoContainingOp fuseIntoOp =
          builder.create<transform::FuseIntoContainingOp>(
              loc, builder.getType<transform::AnyOpType>(),
              builder.getType<transform::AnyOpType>(), producerHandle,
              containingLoopHandle);
      Value fusedOp = fuseIntoOp.getFusedOp();
      Value newContainingLoopHandle = fuseIntoOp.getNewContainingOp();
      builder.create<transform::ApplyPatternsOp>(
          loc, newContainingLoopHandle,
          [](OpBuilder &innerBuilder, Location loc) {
            innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(
                loc);
          });
      if (hfusion::shouldUseTileReductionUsingForV2(producer)) {
        tileReductionOp(builder, seqOp, producer, fusedOp,
                        producerInfo.tileSize, producerInfo.label,
                        otherVectorizableOps);
      } else if (!isa<tensor::ExpandShapeOp>(producer)) {
        builder.create<transform::TileUsingForOp>(loc, fusedOp,
                                                  producerInfo.tileSize);
      }
      builder.create<transform::ApplyPatternsOp>(
          loc, newContainingLoopHandle,
          [](OpBuilder &innerBuilder, Location loc) {
            innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(
                loc);
          });
      Value funcHandle = builder.create<transform::MatchOp>(
          loc, seqOp.getBodyBlock()->getArguments().front(),
          ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
      builder.create<transform::ApplyRegisteredPassOp>(
          loc, builder.getType<transform::AnyOpType>(), funcHandle,
          builder.getStringAttr("eliminate-single-iteration-scf-for"));
      applyCleanUp(builder, seqOp);
    }
}

void AutoVectorizeV2::buildTileAndFuseTransformSequenceForBlock(
    OpBuilder &builder, transform::SequenceOp seqOp, Block *block,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps,
    PlanContext &ctx) {
  ctx.resetForBlock();
  planFuseSiblingForLeafNodes(block, ctx);
  planFuseProducersIntoConsumers(block, ctx);
  ctx.computeTileSize();
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "========Dumping LeafNodeGroups begin========\n");
  for (auto &node : ctx.nodes()) {
    LLVM_DEBUG(llvm::dbgs() << "========Dumping group========\n");
    for (auto op : node->leafOps()) {
      LLVM_DEBUG(llvm::dbgs() << *op);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "----shape:[");
      for (auto i : ctx.getInfo(op).shape)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
      LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
      for (auto i : ctx.getInfo(op).tileSize)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "====Dumping ProducersToBeFusedInto begin====\n");
  for (auto &node : ctx.nodes()) {
    for (auto op : node->producerOps()) {
      LLVM_DEBUG(llvm::dbgs() << *op);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "----shape:[");
      for (auto i : ctx.getInfo(op).shape)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
      LLVM_DEBUG(llvm::dbgs() << "----tilesize:[");
      for (auto i : ctx.getInfo(op).tileSize)
        LLVM_DEBUG(llvm::dbgs() << i << ",");
      LLVM_DEBUG(llvm::dbgs() << "]----\n");
    }
  }
#endif
  tileAndFuseSiblingForLeafNodes(builder, seqOp, ctx, otherVectorizableOps);
  fuseProducersIntoConsumers(builder, seqOp, block, ctx, otherVectorizableOps);
}

void AutoVectorizeV2::buildVectorizeTransformSequence(
    OpBuilder &builder, transform::SequenceOp seqOp,
    SmallVector<std::pair<std::string, SmallVector<int64_t>>>
        &otherVectorizableOps,
    PlanContext &result) {
  auto loc = seqOp.getLoc();
  for (auto info : result.opInfos()) {
    // Here only vectorize LinalgOp, interleave/deinterleave will be vectorized
    // in convert-hfusion-to-hivmave pass.
    if (isa<linalg::LinalgOp>(info.first)) {
      FusableOpInfo &opInfo = info.second;
      builder.create<transform::VectorizeOp>(
          loc, getOpTransformHandle(opInfo.label, builder, seqOp),
          SmallVector<Value>(), opInfo.tileSize, nullptr,
          SmallVector<bool>(opInfo.tileSize.size(), false));
    }
  }
  for (auto vectorizableOp : otherVectorizableOps) {
    builder.create<transform::VectorizeOp>(
        loc, getOpTransformHandle(vectorizableOp.first, builder, seqOp),
        SmallVector<Value>(), vectorizableOp.second, nullptr,
        SmallVector<bool>(vectorizableOp.second.size(), false));
  }
}

void AutoVectorizeV2::applyCleanUp(OpBuilder &builder,
                                   transform::SequenceOp seqOp) {
  auto loopLikeAttr = transform::MatchInterfaceEnumAttr::get(
      builder.getContext(), transform::MatchInterfaceEnum::LoopLikeInterface);
  Value loopLikeHandle = builder
                             .create<transform::MatchOp>(
                                 builder.getInsertionPoint()->getLoc(),
                                 builder.getType<transform::AnyOpType>(),
                                 seqOp.getBodyBlock()->getArguments().front(),
                                 ArrayAttr(), loopLikeAttr, DictionaryAttr(),
                                 DictionaryAttr{}, TypeAttr{}, ArrayAttr{})
                             .getResults();
  builder.create<transform::ApplyLoopInvariantCodeMotionOp>(
      loopLikeHandle.getLoc(), loopLikeHandle);

  Value funcHandle = builder.create<transform::MatchOp>(
      builder.getInsertionPoint()->getLoc(),
      seqOp.getBodyBlock()->getArguments().front(),
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
  auto bodyBuilder = [](OpBuilder &innerBuilder, Location loc) {
    innerBuilder.create<transform::ApplyCanonicalizationPatternsOp>(loc);
    innerBuilder
        .create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            loc);
  };
  transform::ApplyPatternsOp applyPatternsOp =
      builder.create<transform::ApplyPatternsOp>(funcHandle.getLoc(),
                                                 /*target=*/funcHandle,
                                                 /*bodyBuilder=*/bodyBuilder);
  applyPatternsOp.setApplyCse(true);
  applyPatternsOp.setDisablePatternsAttr(builder.getArrayAttr(
      SmallVector<Attribute>{builder.getStringAttr("SimplifyTrivialLoops")}));
}

transform::SequenceOp AutoVectorizeV2::buildTransformSequence(
    func::FuncOp func, RetriedOptions &retryCtx, OpBuilder &builder) {
  analysis::VFStackInfoBuilder vfStack{retryCtx.enableVFStackLimit};
  PlanContext ctx(retryCtx, vfStack);
  ctx.initFusableOpInfoFrom(func);
  SmallVector<std::pair<std::string, SmallVector<int64_t>>>
      otherVectorizableOps;

  transform::SequenceOp seqOp = buildTransformSequenceOp(builder, func);
  func.walk([&](Block *block) {
    if (scope::utils::isInCubeScope(block->getParentOp()) ||
        isCubeScopeOp(block->getParentOp()))
      return;

    if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp, scope::ScopeOp>(
            block->getParentOp()))
      buildTileAndFuseTransformSequenceForBlock(builder, seqOp, block,
                                                otherVectorizableOps, ctx);
  });
  buildVectorizeTransformSequence(builder, seqOp, otherVectorizableOps, ctx);
  return seqOp;
}

void AutoVectorizeV2::emitTransformSequenceIR(func::FuncOp func,
                                              RetriedOptions &retryCtx,
                                              OpBuilder &builder) {
  // Emit mode materializes the payload tags and transform sequence for
  // debugging/replay only. It intentionally does not call the transform
  // interpreter, so the payload IR is left unvectorized.
  transform::SequenceOp seqOp = buildTransformSequence(func, retryCtx, builder);
  StringRef funcName = func.getSymName();
  func->setAttr(transform::TransformDialect::kTargetTagAttrName,
                builder.getStringAttr(
                    hfusion::auto_vectorize::getPayloadRootTag(funcName)));
  seqOp->setAttr(transform::TransformDialect::kTargetTagAttrName,
                 builder.getStringAttr(
                     hfusion::auto_vectorize::getTransformRootTag(funcName)));
}

LogicalResult AutoVectorizeV2::vectorize(func::FuncOp func,
                                         RetriedOptions &retryCtx,
                                         OpBuilder &builder) {
  transform::SequenceOp seqOp = buildTransformSequence(func, retryCtx, builder);

  transform::TransformOptions transformOptions;
  transformOptions.enableExpensiveChecks(false);
  LogicalResult result = transform::applyTransformNamedSequence(
      func, seqOp, func->getParentOfType<ModuleOp>(), transformOptions);
  seqOp->erase();
  if (succeeded(result))
    propagateWasBoolToInt8ToTransferReads(func);

  hfusion::AutoVectorizeVerifier verifier;
  return failure(
      failed(result) ||
      failed(verifier.verifyFreeVectorRegion(true).emitDiagnostics(false).check(
          func)));
}

void AutoVectorizeV2::sortFunc(func::FuncOp func) {
  func.walk([&](Block *block) {
    if (isa<func::FuncOp, scf::ForOp, scf::WhileOp, scf::IfOp, scope::ScopeOp>(
            block->getParentOp()))
      sortTopologically(block);
  });
}

LogicalResult AutoVectorizeV2::runAttempt(func::FuncOp func,
                                          RetriedOptions &retryCtx,
                                          OpBuilder &builder,
                                          IRRewriter &rewriter) {
  std::string funcName = func.getSymName().str();

  builder.setInsertionPointAfter(func);
  func::FuncOp cloned = cast<func::FuncOp>(builder.clone(*func));
  SymbolTable::setSymbolName(cloned, "cloned_" + funcName);

  if (succeeded(vectorize(cloned, retryCtx, builder))) {
    rewriter.eraseOp(func);
    SymbolTable::setSymbolName(cloned, funcName);
    sortFunc(cloned);
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "========Failed========\n");
  rewriter.eraseOp(cloned);
  return failure();
}

static bool isSIMDFunc(func::FuncOp func) {
  if (auto pmAttr = func->getAttrOfType<StringAttr>("parallel_mode"))
    return pmAttr.getValue() == "simd";
  // Fallback: scan for SIMT ops via standard codebase APIs.
  bool hasSIMT = false;
  func.walk([&](Operation *inner) {
    if (hfusion::isSimtOps(inner) || hivm::util::isSIMTVF(inner)) {
      hasSIMT = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasSIMT;
}

// Minimum tensor-element count above which an intermediate result between
// VFs is considered large enough to risk UB overflow when multiplied by
// tt.num_stages.
static constexpr int64_t kLargeTensorMinElements = 4096;

// Inline private _fused_ calls whose return tensors are large enough to
// overflow UB if left as cross-VF intermediate buffers.
static DenseSet<func::FuncOp>
inlineFusedCalls(func::FuncOp func, ModuleOp moduleOp, IRRewriter &rewriter) {
  SmallVector<func::CallOp> callsToInline;
  func.walk([&](func::CallOp callOp) {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee || !callee.isPrivate() ||
        !callee.getSymName().contains("_fused_")) {
      return;
    }
    bool hasLargeResult = false;
    for (Type t : callee.getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(t)) {
        if (tensorTy.hasStaticShape() &&
            ShapedType::getNumElements(tensorTy.getShape()) >
                kLargeTensorMinElements) {
          hasLargeResult = true;
          break;
        }
      }
    }
    if (hasLargeResult) {
      callsToInline.push_back(callOp);
    }
  });

  DenseSet<func::FuncOp> inlinedFuncs;
  for (auto callOp : callsToInline) {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee)
      continue;

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(callOp);
    IRMapping mapping;
    for (auto [arg, operand] :
         llvm::zip(callee.getArguments(), callOp.getOperands()))
      mapping.map(arg, operand);

    for (auto &op : callee.getBody().front()) {
      if (isa<func::ReturnOp>(op)) {
        for (auto [result, retVal] :
             llvm::zip(callOp->getResults(), op.getOperands()))
          result.replaceAllUsesWith(mapping.lookup(retVal));
      } else {
        rewriter.clone(op, mapping);
      }
    }
    rewriter.eraseOp(callOp);
    inlinedFuncs.insert(callee);
  }
  return inlinedFuncs;
}

void AutoVectorizeV2::runOnOperation() {
  ModuleOp op = getOperation();
  MLIRContext *context = op->getContext();
  IRRewriter rewriter(context);
  OpBuilder builder(context);
  bool fallback = false;

  SmallVector<func::FuncOp> fusableFuncList;
  collectFusableFuncInModule(op, fusableFuncList);

  // Inline VFFusionPass _fused_ sub-functions with large return tensors.
  // TODO: remove it after better solution.
  DenseSet<func::FuncOp> allInlined;
  for (func::FuncOp func : fusableFuncList) {
    if (hacc::utils::isDevice(func) && isSIMDFunc(func)) {
      auto inlined = inlineFusedCalls(func, op, rewriter);
      allInlined.insert(inlined.begin(), inlined.end());
    }
  }
  llvm::erase_if(fusableFuncList,
                 [&](func::FuncOp f) { return allInlined.contains(f); });
  for (auto callee : allInlined)
    rewriter.eraseOp(callee);

  for (func::FuncOp func : fusableFuncList) {
    RetriedOptions retryCtx{maxFusedOps, enableMultipleConsumerFusion,
                            enableCrossIfFusion, enableVFStackLimit,
                            vectorLength};
    if (emitTransformSequence) {
      // Emit payload IR with transform sequence.
      emitTransformSequenceIR(func, retryCtx, builder);
      // Return early to avoid applying the transform sequence.
      continue;
    }

    LogicalResult result = runAttempt(func, retryCtx, builder, rewriter);
    bool retried = false;
    if (failed(result) && retryCtx.enableMultipleConsumerFusion) {
      func.emitWarning() << "AutoVectorizeV2 failed; "
                            "retrying with enableMultipleConsumerFusion=false";
      retryCtx.enableMultipleConsumerFusion = false;
      result = runAttempt(func, retryCtx, builder, rewriter);
      retried = true;
    }

    if (failed(result)) {
      func.emitWarning()
          << (retried ? "AutoVectorizeV2 retry failed"
                      : "AutoVectorizeV2 failed")
          << "; falling back to legacy HFusionAutoVectorize pass";
      fallback = true;
      break;
    }
  }

  if (fallback) {
    PassManager pm(context);
    AutoVectorizeOptions vecOptions;
    // Set maxVectorizeAxes to be 1 for compatiblity.
    // TODO: Remove this constraint after e2e support for multi axes
    // vectorization.
    vecOptions.maxVectorizeAxes = 2;
    vecOptions.treeReduce = treeReduce;
    pm.addPass(hfusion::createHFusionAutoVectorizePass());
    std::ignore = pm.run(op);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createHFusionAutoVectorizeV2Pass(
    const AutoVectorizeV2Options &options) {
  return std::make_unique<AutoVectorizeV2>(options);
}
