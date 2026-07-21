//===- PopulateSharedMemoryOffsetToDPX.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs after AllocateAscendSharedMemory, which attaches an
// "allocation.offset" integer attribute to every ttg.local_alloc op.
//
// For each tt.call_scalar whose shm_desc is produced by a ttg.local_alloc,
// this pass:
//
//   1. Creates an ascend_dpx.call_scalar with the same callee and arguments,
//      carrying the offset as the "use_shmem_offset" formal attribute.
//
//   2. Replaces all uses of the tt.call_scalar results with the new op's
//      results.
//
//   3. Removes every ttg.local_load that reads from the same memdesc.
//      - Scalar case (load result unused): the load is simply erased.
//      - Tensor case (load feeds tt.broadcast): each broadcast is replaced
//        by a tt.splat of the ascend_dpx.call_scalar's i32 result, then
//        both the broadcast and the local_load are erased.
//
//   4. Erases the tt.call_scalar (which held the only remaining use of the
//      local_alloc result), then erases the local_alloc itself.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/Triton/IR/TritonExtension.h"
#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_POPULATESHAREDMEMORYOFFSETTODPX
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

class PopulateSharedMemoryOffsetToDPXPass
    : public impl::PopulateSharedMemoryOffsetToDPXBase<
          PopulateSharedMemoryOffsetToDPXPass> {
public:
  void runOnOperation() override {
    mlir::triton::FuncOp func = getOperation();

    // Collect (tt.call_scalar, local_alloc) pairs in walk order so we process
    // them after the walk completes (avoids mutation during traversal).
    struct Entry {
      mlir::triton::CallScalarOp callOp;
      LocalAllocOp allocOp;
    };
    SmallVector<Entry> entries;

    func.walk([&](mlir::triton::CallScalarOp callOp) {
      auto allocOp = callOp.getShmDesc().getDefiningOp<LocalAllocOp>();
      if (!allocOp)
        return;
      entries.push_back({callOp, allocOp});
    });

    for (auto &[callOp, allocOp] : entries) {
      OpBuilder b(callOp);

      // ------------------------------------------------------------------
      // 1. Create ascend_dpx.call_scalar with the same callee/args and
      //    store the allocation offset in the "use_shmem_offset" attribute.
      // ------------------------------------------------------------------
      auto offsetAttr =
          allocOp->getAttrOfType<IntegerAttr>("allocation.offset");
      if (!offsetAttr) {
        allocOp->emitOpError(
            "tt.call_scalar's local_alloc has no 'allocation.offset' — "
            "populate-shared-memory-offset-to-dpx must run after "
            "allocate-ascend-shared-memory");
        return signalPassFailure();
      }
      auto newCall = b.create<mlir::ascend_dpx::CallScalarOp>(
          callOp.getLoc(), TypeRange(callOp.getResultTypes()),
          callOp.getCalleeAttr(), ValueRange(callOp.getCallArgs()),
          /*arg_attrs=*/ArrayAttr{}, /*res_attrs=*/ArrayAttr{},
          /*use_shmem_offset=*/offsetAttr);

      // ------------------------------------------------------------------
      // 2. Replace all uses of the tt.call_scalar results.
      // ------------------------------------------------------------------
      callOp->replaceAllUsesWith(newCall->getResults());
      Value callResult =
          newCall.getNumResults() > 0 ? newCall.getResult(0) : Value{};

      // ------------------------------------------------------------------
      // 3. Process every ttg.local_load that reads from this memdesc.
      // ------------------------------------------------------------------
      SmallVector<LocalLoadOp> loadsToErase;
      for (Operation *user :
           llvm::make_early_inc_range(allocOp.getResult().getUsers())) {
        auto loadOp = dyn_cast<LocalLoadOp>(user);
        if (!loadOp)
          continue;

        Value loadResult = loadOp.getResult();

        if (loadResult.use_empty()) {
          // Scalar ordering load — no uses, just erase.
          loadsToErase.push_back(loadOp);
          continue;
        }

        if (!callResult)
          continue;

        // Tensor case: loadResult feeds one or more tt.broadcast ops.
        // Replace each broadcast with tt.splat(callResult).
        SmallVector<BroadcastOp> broadcastsToErase;
        bool allUsesBroadcast =
            llvm::all_of(loadResult.getUsers(),
                         [](Operation *u) { return isa<BroadcastOp>(u); });
        if (!allUsesBroadcast)
          continue;

        b.setInsertionPoint(loadOp);
        for (Operation *u : llvm::make_early_inc_range(loadResult.getUsers())) {
          auto bcastOp = cast<BroadcastOp>(u);
          Value splat = b.create<SplatOp>(
              bcastOp.getLoc(), bcastOp.getResult().getType(), callResult);
          bcastOp.getResult().replaceAllUsesWith(splat);
          broadcastsToErase.push_back(bcastOp);
        }
        for (BroadcastOp bcast : broadcastsToErase)
          bcast->erase();
        loadsToErase.push_back(loadOp);
      }

      for (LocalLoadOp load : loadsToErase)
        load->erase();

      // ------------------------------------------------------------------
      // 4. Erase the tt.call_scalar (removes its shm_desc use from allocOp),
      //    then erase the now-unused local_alloc.
      // ------------------------------------------------------------------
      callOp->erase();
      allocOp->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createPopulateSharedMemoryOffsetToDPXPass() {
  return std::make_unique<PopulateSharedMemoryOffsetToDPXPass>();
}

} // namespace triton
} // namespace bishengir
