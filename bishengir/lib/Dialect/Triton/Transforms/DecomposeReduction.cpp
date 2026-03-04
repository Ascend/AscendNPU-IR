//===- DecomposeReduction.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace bishengir::triton {

#define GEN_PASS_DEF_DECOMPOSEREDUCTION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "decompose-reduction"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

class DecomposeReductionPass
    : public impl::DecomposeReductionBase<DecomposeReductionPass> {

public:
  template <typename T>
  SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, int value) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + index, static_cast<T>(value));
    return res;
  }

  void printBlockedLayout(SmallVector<unsigned> &sizePerThread,
                          SmallVector<unsigned> &threadsPerWarp,
                          SmallVector<unsigned> &warpsPerCTA) {
    LLVM_DEBUG({
      DBGS() << "sizePerThread: [ ";
      for (size_t i = 0; i < sizePerThread.size(); i++)
        llvm::dbgs() << sizePerThread[i] << " ";
      llvm::dbgs() << "]\n";
      DBGS() << "threadsPerWarp: [ ";
      for (size_t i = 0; i < threadsPerWarp.size(); i++)
        llvm::dbgs() << threadsPerWarp[i] << " ";
      llvm::dbgs() << "]\n";
      DBGS() << "warpsPerCTA: [ ";
      for (size_t i = 0; i < warpsPerCTA.size(); i++)
        llvm::dbgs() << warpsPerCTA[i] << " ";
      llvm::dbgs() << "]\n";
    });
  }

  void initBlockedLayout(SmallVector<unsigned int> &sizePerThread,
                         SmallVector<unsigned int> &threadsPerWarp,
                         SmallVector<unsigned int> &warpsPerCTA, int dim) {
    sizePerThread.append(dim, 1);
    threadsPerWarp.append(dim, 1);
    warpsPerCTA.append(dim, 1);
  }

  void completeLayout(SmallVector<unsigned int> &vec, int dimToDump, int ub) {
    // It is possible layout is greater than shape
    int lproduct = 1;
    for (int x : vec) {
      lproduct *= x;
    }
    if (lproduct < ub) {
      // Dump them all missing lproduct at the last dimension
      lproduct = ub / lproduct;
      vec[dimToDump] *= static_cast<unsigned int>(lproduct);
    }
  }

  void calcVectHighDimLayout(SmallVector<unsigned int> &sizePerThread,
                             SmallVector<unsigned int> &threadsPerWarp,
                             SmallVector<unsigned int> &warpsPerCTA, int dim,
                             ArrayRef<int64_t> srcShape) {
    // A simple algorithm to favor gReductionAxis vectorization and high
    // dimension warping and threading.
    SmallVector<int64_t> shape(srcShape.begin(), srcShape.end());
    initBlockedLayout(sizePerThread, threadsPerWarp, warpsPerCTA, dim);
    int nw = gNumWarps;
    for (int i = dim - 1; i >= 0; i--) {
      // Update sizePerThread: 4-way vectorization at gReductionAxis
      if (i == gReductionAxis) {
        sizePerThread[i] = shape[i] >= gVectWidth ? gVectWidth :  static_cast<unsigned int>(shape[i]);
        shape[i] = shape[i] >= gVectWidth ? shape[i] / gVectWidth : 1;
      }
      // Update warpsPerCTA: favor all warps towards higher dims
      if (i != gReductionAxis) {
        if (shape[i] >= nw) {
          warpsPerCTA[i] = static_cast<unsigned int>(nw);
          shape[i] /= nw;
          nw = 1;
        } else {
          warpsPerCTA[i] = static_cast<unsigned int>(shape[i]);
          nw = nw / shape[i];
          shape[i] = 1;
        }
      }
    }
    // Update threadsPerWarp: favor threads towards higher dims
    int nt = gNumThreads;
    for (int i = dim - 1; i >= 0; i--) {
      if (shape[i] >= nt) {
        threadsPerWarp[i] = static_cast<unsigned int>(nt);
        shape[i] /= nt;
        nt = 1;
      } else {
        threadsPerWarp[i] = static_cast<unsigned int>(shape[i]);
        nt = nt / shape[i];
        shape[i] = 1;
      }
    }
    // It is possible layout is greater than shape
    // Dump all missing lproduct at the last dimension
    completeLayout(threadsPerWarp, dim - 1, gNumThreads);
    int nonReductionAxis = -1; // Search for nonReductionAxis
    for (int i = dim - 1; i >= 0; i--) {
      if (i != gReductionAxis) {
        nonReductionAxis = i;
        break;
      }
    }
    // Dump all missing lproduct at the last nonReduction dimension
    completeLayout(warpsPerCTA, nonReductionAxis, gNumWarps);
    int sproduct = 1;
    for (int x : shape) {
      sproduct *= x;
    }
    LDBG("Layout will be replicated " << sproduct << " times.");
    printBlockedLayout(sizePerThread, threadsPerWarp, warpsPerCTA);
  }

  BlockedEncodingAttr create3DLayout(MLIRContext *ctx,
                                     ArrayRef<int64_t> srcShape) {
    SmallVector<unsigned int> sizePerThread, threadsPerWarp, warpsPerCTA;
    calcVectHighDimLayout(sizePerThread, threadsPerWarp, warpsPerCTA, 3,
                          srcShape);

    // Default template values
    SmallVector<unsigned int> order = {2, 0, 1};
    SmallVector<unsigned int> ctasPerCGA = {1, 1, 1};
    SmallVector<unsigned int> ctasOrder = {2, 1, 0};
    SmallVector<unsigned int> ctasSplitNum = {1, 1, 1};
    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        ctx, ctasPerCGA, ctasSplitNum, ctasOrder);
    auto blocked2d = mlir::triton::gpu::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
    return blocked2d;
  }

  BlockedEncodingAttr create2DLayout(MLIRContext *ctx,
                                     ArrayRef<int64_t> srcShape) {
    SmallVector<unsigned int> sizePerThread, threadsPerWarp, warpsPerCTA;
    calcVectHighDimLayout(sizePerThread, threadsPerWarp, warpsPerCTA, 2,
                          srcShape);

    // Default template values
    SmallVector<unsigned int> order = {1, 0};
    SmallVector<unsigned int> ctasPerCGA = {1, 1};
    SmallVector<unsigned int> ctasOrder = {1, 0};
    SmallVector<unsigned int> ctasSplitNum = {1, 1};

    auto ctaLayout = mlir::triton::gpu::CTALayoutAttr::get(
        ctx, ctasPerCGA, ctasSplitNum, ctasOrder);
    auto blocked2d = mlir::triton::gpu::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
    return blocked2d;
  }

  Operation *reductionFeedingLayoutConvert(triton::ReduceOp reduceOp) {
    Value cDst = ((Operation *)reduceOp)->getResult(0);
    for (auto *iOp : cDst.getUsers()) {
      if (dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(iOp)) {
        return iOp;
      }
    }
    return nullptr;
  }

  Operation *reductionFeedingReshape(triton::ReduceOp reduceOp) {
    Value cDst = ((Operation *)reduceOp)->getResult(0);
    for (auto *iOp : cDst.getUsers()) {
      if (dyn_cast<triton::ReshapeOp>(iOp)) {
        return iOp;
      }
    }
    return nullptr;
  }

  Operation *reshapeFeedingLayoutConvert(triton::ReshapeOp reshapeOp) {
    Value cDst = ((Operation *)reshapeOp)->getResult(0);
    for (auto *iOp : cDst.getUsers()) {
      if (dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(iOp)) {
        return iOp;
      }
    }
    return nullptr;
  }

  bool hasWarpSynchronousInput(RankedTensorType srcType, int axis) {
    auto srcEncoding = srcType.getEncoding();
    auto blocked =
        dyn_cast<mlir::triton::gpu::BlockedEncodingAttr>(srcEncoding);
    assert(blocked && "Expecting blocked encoding on srcType.");
    auto WarpsPerCTA = blocked.getWarpsPerCTA();
    if (WarpsPerCTA[axis] == 1)
      return true;
    return false;
  }

  triton::ReduceOp createReduceOp(Value input, triton::ReduceOp &reduceOp,
                                  OpBuilder &builder) {
    SmallVector<Value> src;
    src.push_back(input);
    auto reduceOp2D = builder.create<triton::ReduceOp>(
        reduceOp.getLoc(), src, builder.getI32IntegerAttr(gReductionAxis));
    auto &newCombineOp2D = reduceOp2D.getCombineOp();
    builder.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp2D,
                              newCombineOp2D.end());
    LLVM_DEBUG({
      DBGS() << "Created 2DTo1D ReduceOp:\n";
      reduceOp2D.dump();
    });
    return reduceOp2D;
  }

  mlir::triton::gpu::ConvertLayoutOp
  createConvertLayoutOp(MLIRContext *ctx, Value reduceOperand,
                        RankedTensorType &newDstType, Location loc,
                        OpBuilder &builder) {
    auto srcType = cast<RankedTensorType>(reduceOperand.getType());
    ArrayRef<int64_t> srcShape = srcType.getShape();
    auto rank = srcShape.size();
    // Build an explicit warp-synchronous 2D or 3D layout against axis=1
    RankedTensorType rDst = cast<RankedTensorType>(srcType);
    switch (rank) {
    case 2: {
      auto blocked2d = create2DLayout(ctx, srcShape);
      newDstType = RankedTensorType::get(rDst.getShape(), rDst.getElementType(),
                                         blocked2d);
      break;
    }
    case 3: {
      auto blocked3d = create3DLayout(ctx, srcShape);
      newDstType = RankedTensorType::get(rDst.getShape(), rDst.getElementType(),
                                         blocked3d);
      break;
    }
    default:
      llvm_unreachable("Not implemented!");
    }

    auto cvtOp = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
        loc, newDstType, reduceOperand);
    LLVM_DEBUG({
      DBGS() << "Created ConvertLayoutOp to Warp-synchronous Layout:\n";
      cvtOp.dump();
    });
    return cvtOp;
  }

  LogicalResult inferWarpSyncEncoding(ArrayRef<int64_t> dstTensorShape,
                                      RankedTensorType srcType,
                                      BlockedEncodingAttr &inferredDstEnc,
                                      int requireWarpSyncLayout, Location loc) {
    LLVM_DEBUG({
      DBGS() << "InferWarpSyncEncoding for dstTensorShape: [ ";
      for (size_t i = 0; i < dstTensorShape.size(); i++)
        llvm::dbgs() << dstTensorShape[i] << " ";
      llvm::dbgs() << "]\n";
    });
    // Try to infer layout for the output of ReshapeOp.
    Attribute srcEnc = srcType.getEncoding();
    auto layoutInterface =
        cast<DialectInferLayoutInterface>(&srcEnc.getDialect());
    auto result = layoutInterface->inferReshapeOpEncoding(
        srcType.getShape(), srcEnc, dstTensorShape, inferredDstEnc, loc);
    if (failed(result)) {
      LDBG("Failed to infer encoding.\n");
      return failure();
    }
    auto warps = inferredDstEnc.getWarpsPerCTA();
    LLVM_DEBUG({
      DBGS() << "Succeeded to infer encoding:\n";
      inferredDstEnc.dump();
      if (requireWarpSyncLayout >= 0) {
        DBGS() << "RequireWarpSyncLayout at axis: " << requireWarpSyncLayout
               << " WarpsPerCTA at axis: " << warps[requireWarpSyncLayout]
               << "\n";
      } else {
        DBGS() << "Does not require inferred encoding to be warpSync.\n";
      }
    });
    if (requireWarpSyncLayout != -1 && warps[requireWarpSyncLayout] != 1) {
      LDBG("Failed because encoding is not warp sync at axis: "
           << gReductionAxis << "\n");
      return failure();
    }
    return success();
  }

  // RequireWarpSyncLayout is -1 when there is no requirement for warp
  // synchronous layout. When it is greater than -1, e.g. 0, or 1, it means the
  // layout needs to be warp synchronous at that axis
  FailureOr<triton::ReshapeOp>
  createReshapeOp(ArrayRef<int64_t> dstTensorShape, Value reduceOperand,
                  RankedTensorType srcType, int requireWarpSyncLayout,
                  Location loc, OpBuilder &builder) {
    BlockedEncodingAttr inferredDstEnc;
    LogicalResult result = inferWarpSyncEncoding(dstTensorShape, srcType, inferredDstEnc,
                                        requireWarpSyncLayout, loc);
    if (failed(result))
      return failure();
    auto dstTensorType = RankedTensorType::get(
        dstTensorShape, srcType.getElementType(), inferredDstEnc);
    auto newReshapeOp =
        builder.create<triton::ReshapeOp>(loc, dstTensorType, reduceOperand);
    LLVM_DEBUG({
      DBGS() << "Created ReshapeOp:\n";
      newReshapeOp.dump();
    });
    return newReshapeOp;
  }

  void initThreadsAndWarpsInfo(ModuleOp moduleOp) {
    if (auto numWarpAttr = moduleOp->getAttr("ttg.num-warps")) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(numWarpAttr)) {
        gNumWarps = intAttr.getInt();
      }
    }
    if (auto numThreadAttr = moduleOp->getAttr("ttg.threads-per-warp")) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(numThreadAttr)) {
        gNumThreads = intAttr.getInt();
      }
    }
    if (gNumWarps <= 0 || gNumThreads <= 0) {
      llvm::report_fatal_error(
          "Cannot determine num-warps or threads-per-warp! num-warps: " +
          Twine(gNumWarps) + ", threads-per-warp: " + Twine(gNumThreads));
    }
  }

  Value createConvertReduceSequence(MLIRContext *ctx,
                                    triton::ReduceOp &reduceOp,
                                    OpBuilder &builder) {
    Value reduceOperand = reduceOp->getOperand(0);
    auto srcType = dyn_cast<RankedTensorType>(reduceOperand.getType());
    assert(srcType && "Expecting tensor type for reduce operand.");
    Location loc = reduceOp.getLoc();
    mlir::triton::gpu::ConvertLayoutOp cvtOp =
        createConvertLayoutOp(ctx, reduceOperand, srcType, loc, builder);
    triton::ReduceOp newReduceOp = createReduceOp(cvtOp, reduceOp, builder);
    Value finalResult = newReduceOp->getResult(0);
    return finalResult;
  }

  bool shouldPerformDecompose(RankedTensorType srcType) {
    auto srcShape = srcType.getShape();
    auto blocked =
        dyn_cast<mlir::triton::gpu::BlockedEncodingAttr>(srcType.getEncoding());
    assert(blocked && "Expecting blocked encoding on srcType.");
    auto threads = blocked.getThreadsPerWarp();
    // Simple but extremely tight cost model.
    if (srcShape.size() == 2 && gReductionAxis == 0 &&
        threads[gReductionAxis] >= 4 && srcShape[1] <= gNumThreads &&
        srcShape[gReductionAxis] >= gNumThreads) {
      LDBG("Profitable to decompose axis: " << gReductionAxis << "\n");
      return true;
    }
    return false;
  }

  Value createDecomposeConvertReduceSequence(MLIRContext *ctx,
                                             triton::ReduceOp &reduceOp,
                                             OpBuilder &builder) {
    // Reshape + WarpSyncReduce + ConvertReduceSequence
    Location loc = reduceOp.getLoc();
    Value reduceOperand = reduceOp->getOperand(0);
    auto srcType = dyn_cast<RankedTensorType>(reduceOperand.getType());
    assert(srcType && "Expecting tensor type for reduce operand.");
    auto srcShape = srcType.getShape();

    // Not profitable to decompose.
    if (!shouldPerformDecompose(srcType)) {
      return createConvertReduceSequence(ctx, reduceOp, builder);
    }

    // First attempt to decompose with localRAxis = 0
    int localRAxis = 0;
    auto outputShape = insertValue(srcShape, 0, 4);
    outputShape[1] /= 4;
    FailureOr<triton::ReshapeOp> newReshapeOp = createReshapeOp(
        outputShape, reduceOperand, srcType, localRAxis, loc, builder);

    // Second attempt to decompose with localRAxis = 1
    if (failed(newReshapeOp)) {
      localRAxis = 1;
      auto outputShape = insertValue(srcShape, 0, 8);
      outputShape[1] /= 8;
      newReshapeOp = createReshapeOp(outputShape, reduceOperand, srcType,
                                     localRAxis, loc, builder);
    }
    if (failed(newReshapeOp)) {
      return createConvertReduceSequence(ctx, reduceOp, builder);
    }

    // Build 3DTo2D reduction on the warp-synchronous input.
    SmallVector<Value> srcs;
    srcs.push_back(*newReshapeOp);
    auto reduceOp3DTo2D = builder.create<triton::ReduceOp>(
        // loc, srcs, builder.getI32IntegerAttr(0));
        loc, srcs, builder.getI32IntegerAttr(localRAxis));
    auto &newCombineOp3D = reduceOp3DTo2D.getCombineOp();
    builder.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp3D,
                              newCombineOp3D.end());

    reduceOperand = reduceOp3DTo2D->getResult(0);
    srcType = cast<RankedTensorType>(reduceOperand.getType());

    mlir::triton::gpu::ConvertLayoutOp cvtOp =
        createConvertLayoutOp(ctx, reduceOperand, srcType, loc, builder);
    triton::ReduceOp newReduceOp = createReduceOp(cvtOp, reduceOp, builder);
    Value finalResult = newReduceOp->getResult(0);
    return finalResult;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp->getContext();
    // Ops scheduled for deferred erasure.
    // ensure users are inserted before their defs.
    llvm::SmallVector<Operation *, 4> opsToErase;
    OpBuilder builder(ctx);
    initThreadsAndWarpsInfo(moduleOp);
    if (gNumThreads != 32) { // TODO: broaden this.
      LDBG("Custom warp synchronous layouts performance not tested.\n");
      return;
    }

    moduleOp.walk<WalkOrder::PreOrder>([&](triton::ReduceOp reduceOp) {
      Operation *convertLayoutOp = reductionFeedingLayoutConvert(reduceOp);
      Value reduceOperand = reduceOp.getOperands()[0];
      auto srcType = cast<RankedTensorType>(reduceOperand.getType());
      if (!isa_and_nonnull<BlockedEncodingAttr>(srcType.getEncoding()))
        return WalkResult::advance();

      auto srcShape = srcType.getShape();
      auto rank = srcShape.size();
      gReductionAxis = reduceOp.getAxis();
      Location loc = reduceOp.getLoc();
      builder.setInsertionPointAfter((Operation *)reduceOp);

      if (gReductionAxis == 1 || gReductionAxis == 0) {
        if (hasWarpSynchronousInput(srcType, gReductionAxis))
          return WalkResult::advance();
      }

      // Enable for limited configurations as per required.
      if ((gReductionAxis == 0 && rank == 2) ||
          (gReductionAxis == 1 && rank == 3)) {
        convertLayoutOp = reductionFeedingLayoutConvert(reduceOp);
        if (convertLayoutOp) {
          Value finalResult =
              createDecomposeConvertReduceSequence(ctx, reduceOp, builder);
          convertLayoutOp->setOperands(finalResult);
          opsToErase.push_back(reduceOp);
          return WalkResult::advance();
        }
      }

      // First attempt to generalize to limited rank == 3,
      // any axis. Special handling for the extra reshape.
      Operation *rOp = reductionFeedingReshape(reduceOp);
      if (rOp != nullptr) {
        if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(rOp)) {
          convertLayoutOp = reshapeFeedingLayoutConvert(reshapeOp);
          if (convertLayoutOp == nullptr || rank != 3)
            return WalkResult::advance();
          Value rOut = createConvertReduceSequence(ctx, reduceOp, builder);
          auto srcType = cast<RankedTensorType>(rOut.getType());
          auto outputShape = reshapeOp.getType().getShape();
          FailureOr<triton::ReshapeOp> newReshapeOp =
              createReshapeOp(outputShape, rOut, srcType, -1, loc, builder);
          SmallVector<Value> newLayoutOperands;
          newLayoutOperands.push_back(*newReshapeOp);
          convertLayoutOp->setOperands(newLayoutOperands);
          // Record erase order:
          //   reshapeOp uses reduceOp
          //   -> push reshapeOp first, reductOp second
          opsToErase.push_back(reshapeOp);
          opsToErase.push_back(reduceOp);
          return WalkResult::advance();
        }
      }

      convertLayoutOp = reductionFeedingLayoutConvert(reduceOp);
      if (convertLayoutOp == nullptr)
        return WalkResult::advance();

      if (rank == 2 && gReductionAxis == 1) {
        // TODO: Generalize this for wider cases of decomposition.
      } else
        return WalkResult::advance();

      // Below only handles rank 2 with axis 1 but allows decomposition.
      Value finalResult;
      mlir::triton::gpu::ConvertLayoutOp cvtOp;
      while (true) {
        // Check if needDecompose
        if (srcShape[gReductionAxis] / gNumThreads > 1) {

          // Build ReshapeOp.
          auto outputShape = insertValue(srcShape, rank, gNumThreads);
          outputShape[gReductionAxis] /= gNumThreads;
          FailureOr<triton::ReshapeOp> newReshapeOp = createReshapeOp(
              outputShape, reduceOperand, srcType, -1, loc, builder);

          // Build 3DTo2D reduction on the warp-synchronous input.
          SmallVector<Value> srcs;
          srcs.push_back(*newReshapeOp);
          auto reduceOp3DTo2D = builder.create<triton::ReduceOp>(
              loc, srcs, builder.getI32IntegerAttr(2));
          auto &newCombineOp3D = reduceOp3DTo2D.getCombineOp();
          builder.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp3D,
                                    newCombineOp3D.end());

          LLVM_DEBUG({
            DBGS() << "Created 3DTo2D ReduceOp:\n";
            reduceOp3DTo2D.dump();
          });

          reduceOperand = reduceOp3DTo2D->getResult(0);
          srcType = cast<RankedTensorType>(reduceOperand.getType());
          srcShape = srcType.getShape();
          rank = srcShape.size();
        }
        // Build warp synchronous layout.
        cvtOp =
            createConvertLayoutOp(ctx, reduceOperand, srcType, loc, builder);
        reduceOperand = cvtOp->getResult(0);

        if (srcShape[gReductionAxis] / gNumThreads <= 1) {
          // Does not need more decompose
          break;
        }
      } // end while(true)
      // Get the convert_layout on the original reduction's result.
      triton::ReduceOp reduceOp2D = createReduceOp(cvtOp, reduceOp, builder);
      finalResult = reduceOp2D->getResult(0);

      SmallVector<Value> newLayoutOperands;
      newLayoutOperands.push_back(finalResult);
      convertLayoutOp->setOperands(newLayoutOperands);
      opsToErase.push_back(reduceOp);
      return WalkResult::advance();
    });

    // Erase in insertion order so that users are removed before defs.
    for (Operation *op : opsToErase) {
      assert(op->use_empty() && "uses remaining for ops!");
      op->erase();
    }
  }

private:
  // Global parameters
  int gNumWarps = 0;
  int gNumThreads = 0;
  int gReductionAxis = -1;
  const int gVectWidth = 4;
};

std::unique_ptr<mlir::Pass> createDecomposeReductionPass() {
  return std::make_unique<DecomposeReductionPass>();
}

} // namespace bishengir::triton
