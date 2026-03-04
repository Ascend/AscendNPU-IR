#include "bishengir/Analysis/AscendAllocation.h"
#include "bishengir/Analysis/AscendUtility.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "allocation-shared-memory"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::ascend {

// Check if allocated shared memory size overflow.
void AscendAllocationSharedMemCheckFn(Operation* op, int allocatedSharedMemorySize) {
  LDBG("Check if shared memory allocation size beyond max of memory capacity ---");

  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    LDBG("module op not found. Skip checking.");
    return;
  }

  // Shared memory capacity allowed for simt-vf.
  int sharedMemoryCapacity = 0;

  // The module attribute specified for shared memory capacity size.
  constexpr static char AttrShared[] = "ttg.shared";

  if (auto sharedMemAttr = moduleOp->getAttr(AttrShared)) {
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(sharedMemAttr)) {
      sharedMemoryCapacity = intAttr.getInt();
    } else {
      moduleOp->emitError() << AttrShared << " must be an IntegerAttr, but got: "
        << sharedMemAttr;
      return;
    }
  } else {
    LDBG(Twine(AttrShared) + " attribute missing. Skip checking.");
    return;
  }

  if (sharedMemoryCapacity < 0) {
    moduleOp->emitError() << AttrShared << " must be non-negative, got "
      << sharedMemoryCapacity;
    return;
  } else if (sharedMemoryCapacity == 0) {
    LDBG("shared memory capacity is unknown. Skip checking.");
    return;
  }

  if (allocatedSharedMemorySize > sharedMemoryCapacity)
    llvm::report_fatal_error("UB overflow, requires at least "
        + Twine(allocatedSharedMemorySize) + " bytes while only "
        + Twine(sharedMemoryCapacity) + " bytes available!");

  LDBG("Check PASS. Allocated: " + Twine(allocatedSharedMemorySize)
       + ", available: " + Twine(sharedMemoryCapacity));
}

unsigned AscendAllocationAnalysisScratchSizeFn(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    AscendReduceOpHelper helper(reduceOp);
    return helper.getScratchSizeInBytes();
  }

  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::ascend
