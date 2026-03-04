//===- PureElemwiseSchedule.cpp -- Auto-schedule fused kernels --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for pure elementwise kernels.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/PureElemwiseSchedule.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/BufferUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "AutoScheduleAttrDefs.h"

#include <algorithm>

#define DEBUG_TYPE "hfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Pure Elemwise] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hfusion;

namespace {
/// Tiling Data is organized as:
///   1. Tiling Key
///   2. UB Tile Size
///   3. UB Buffer Size
constexpr size_t kUBTileSizePos = 1;
constexpr size_t kUBBufferSizePos = 2;

/// Tiling Key
constexpr int64_t kTilingCaseKey100 = 100;

} // namespace

//===----------------------------------------------------------------------===//
// PureElemwiseScheduler
//===----------------------------------------------------------------------===//

TilingComputeFn PureElemwiseScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            StmtExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);

    int64_t maxBufferCnt = kernelInfo->maxBufferCnt;
    assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();
    // The number of element that can be store on unified buffer.

    Expr smallestTypeBits =
        opBuilder->createConstExpr(kernelInfo->getSmallestElementTypeBits());
    Expr ubMaxSizeInBits = opBuilder->createConstExpr(kUBMaxSizeInBits);
    Expr ubAvailableNumInSmallestTypeBits =
        ubMaxSizeInBits.floorDiv(smallestTypeBits).floorDiv(maxBufferCnt);
    // ub available number align down to block size
    Expr alignedBufferSizeInBits =
        (ubAvailableNumInSmallestTypeBits * smallestTypeBits)
            .alignDown(kUBAlignSizeInBytes * kNumBitsInByte);
    ubAvailableNumInSmallestTypeBits =
        alignedBufferSizeInBits.floorDiv(smallestTypeBits);

    Expr const100 = opBuilder->createConstExpr(kTilingCaseKey100);
    Expr tilingKeyExpr = (ubAvailableNumInSmallestTypeBits > 0) * const100;

    auto tilingDataType = IntegerType::get(ctx, 64);
    TilingData tilingData0 =
        TilingData(std::move(tilingKeyExpr), tilingDataType);
    TilingData tilingData1 =
        TilingData(std::move(ubAvailableNumInSmallestTypeBits), tilingDataType);
    TilingData tilingData2 = TilingData(
        alignedBufferSizeInBits.floorDiv(kNumBitsInByte), tilingDataType);

    // Build tiling struct.
    TilingStruct s;
    s.push_back(std::move(tilingData0));
    s.push_back(std::move(tilingData1));
    s.push_back(std::move(tilingData2));

    // Set tiling keys.
    TilingCases c;
    if (failed(c.addKey(kTilingCaseKey100)))
      return {};

    return TilingFnResultTy(std::make_pair(std::move(c), std::move(s)));
  };
}

LogicalResult PureElemwiseScheduler::createScheduleImpl(TilingKey key,
                                                        OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  // Pure Elemwise only have one tiling case
  if (key != kTilingCaseKey100)
    return failure();

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Step 1: Cache read input arguments.
  getOpsWithName(hfusion::LoadOp::getOperationName(), opBuilder);

  // Step 2: Cache write kernel results.
  CacheIOResult cacheWriteResult = {
      getOpsWithName(hfusion::StoreOp::getOperationName(), opBuilder)};

  // Step 3: Tile cache writes using `scf.forall` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);
  ForallTilingResult tileUsingForAllResult =
      tileUsingForAll(splitCachedOps, getKernelInfo()->blockDim, opBuilder);

  // Step 4: Fuse independent `scf.forall` ops.
  ValueHandle *fusedLoop = fuseLoops(tileUsingForAllResult.loops, opBuilder);
  // Handle to cached ops is invalidated after loop fuse, needs rematching.
  cacheWriteResult.cachedOps->setStatus(HandleStatus::kNeedsRematch);

  // Step 5: Fuse producers into `scf.forall` op.
  // We wish to fuse producers ops by reverse topological ordering.
  ValueHandle *producerOps = getIntermediateProducers(opBuilder);
  ValueHandles targetsToFuseInto = {producerOps};
  ValueHandles fusedLoopList = {fusedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder,
                     /*duplicateProducers=*/true,
                     /*applyCanonicalizeAfterEachFusion=*/true);

  // Step 6: Tile cache writes again using `scf.for` op.
  splitCachedOps = splitHandle(cacheWriteResult.cachedOps,
                               getKernelInfo()->numOutputs, opBuilder);
  // For Pure Elemwise schedule, the tile size should be one dimensional
  auto ubTilingDataHandle =
      ValueHandleFoldResults{tilingDataHandles[kUBTileSizePos]};
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, ubTilingDataHandle, opBuilder);

  // Step 7: Apply canonicalize patterns.
  //         Disabled `kSimplifyTrivialLoops` because loop handles might be
  //         invalidate if the tiled loop is trivial during compile-time
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{
          TransformPatternKind::CSE, TransformPatternKind::CANONICALIZATION,
          TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE},
      opBuilder,
      /*disablePatterns=*/
      SmallVector<CanonicalizationPatternKind>{
          CanonicalizationPatternKind::kSimplifyTrivialLoops});

  // Step 8: Fuse independent `scf.for` ops.
  auto loops = llvm::map_to_vector(tileUsingForResult.loops,
                                   [](ValueHandles hs) { return hs.front(); });
  fusedLoop = fuseLoops(loops, opBuilder);
  // Handle are invalidated after loop fuse, needs rematching.
  fusedLoop->setStatus(HandleStatus::kNeedsRematch);
  cacheWriteResult.cachedOps->setStatus(HandleStatus::kNeedsRematch);

  // Step 9: Fuse producers into `scf.for` op.
  fusedLoopList = {fusedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder,
                     /*duplicateProducers=*/true,
                     /*applyCanonicalizeAfterEachFusion=*/true);

  // Step 10: Set buffer size.
  ValueHandles targetsToSetBufferSize = {producerOps};
  TilingData *bufferSize = tilingInfo->getTilingData(kUBBufferSizePos);
  assert(bufferSize->isConst() && "buffer size should be const");
  uint64_t bufferSizeConst = static_cast<uint64_t>(bufferSize->getConst());
  if (bufferSizeConst == 0u)
    return getToBeScheduledKernel().emitError(
        "Buffer size is less than or equal to zero. Possibly because there is "
        "not enough space on local memory!");

  // Rematch handles to make sure they are valid.
  setStatusTo(targetsToSetBufferSize, HandleStatus::kNeedsRematch);
  SetBufferSizeOptions bufferSizeOptions{transform::SetBufferSizeMode::kPerByte,
                                         getKernelInfo()->smallestElementType};
  setBufferSize(targetsToSetBufferSize, bufferSizeConst, opBuilder,
                bufferSizeOptions);
  return success();
}
