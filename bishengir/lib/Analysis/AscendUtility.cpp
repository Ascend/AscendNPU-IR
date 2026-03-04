#include "bishengir/Analysis/AscendUtility.h"

#include <deque>

#include "bishengir/Analysis/AscendAllocation.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {

using namespace triton::gpu;

namespace triton::ascend {

bool AscendReduceOpHelper::isSharedMemoryReductionPreferred() {
  // This method determines if for Ascend device shared-memory
  // reduction, specifically in reduceWithinWarps phase is profitable
  // over butterfly implementation. There should be a way to access
  // target device information for this kind of detail, so when possible
  // this method should be moved accordingly.
  unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
  unsigned interleave = getThreadOffsetOnReductionAxis();
  bool Result = false;
  if ((sizeIntraWarps >= 16) && (interleave == 1) &&
      isWarpSynchronous()) {
    ReduceOp rop = getOperation();
    Operation *BaseOp = rop.getOperation();
    if (util::getPassColumnDigit(BaseOp, "reduce-op") != 0) {
      Result = true;
    }
  }
  return Result;
}

SmallVector<unsigned> AscendReduceOpHelper::getScratchRepShape() {
  SmallVector<unsigned> smemShape;
  // This case doesn't need inter-warp communication
  if (isWarpSynchronous()) {
    if (!isSharedMemoryReductionPreferred())
      return {0, 0};
    // Otherwise we need some memory to store data from
    // each thread. For the moment, get the shape size
    // without any compression, so this is conservative.
    smemShape = convertType<unsigned>(srcShape);
    return smemShape;
  }

  smemShape = convertType<unsigned>(srcShape);
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

} // namespace triton::ascend
} // namespace mlir
