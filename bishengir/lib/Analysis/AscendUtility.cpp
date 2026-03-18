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

#define str_attr(str) ::mlir::StringAttr::get(ctx, (str))

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type) {
  MLIRContext *ctx = layout.getContext();
  auto shape = type.getShape();
  unsigned rank = shape.size();

  auto ll = triton::gpu::toLinearLayout(type);

  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  SmallVector<SmallVector<unsigned>> offsets;
  for (int i = 0; i < ll.getInDimSize(str_attr("register")); i++) {
    auto idxs = ll.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    assert(idxs.size() == rank);
    for (unsigned k = 0; k < rank; ++k) {
      assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
    }
    offsets.push_back(
        llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
  }
  return offsets;
}

unsigned AscendReduceOpHelper::getAccumulatorCount() {
  // From here we determine how many accumulators a thread will use
  // AFTER it passes through reduceWithinThreads. This is how we will know
  // how many data elements each thread will need in shared memory for storage.
  ReduceOp op = getOperation();
  RankedTensorType operandType = op.getInputTypes()[0];
  // Assumes offsets don't actually depend on type
  SmallVector<SmallVector<unsigned>> offsets =
      emitOffsetForLayout(getSrcLayout(), operandType);
  // Thread X might hold the same input value in two registers.  Get the
  // indices in `offsets` that hold unique values, and only accumulate over
  // those.
  llvm::MapVector<ArrayRef<unsigned>, int> uniqueOffsets;
  std::map<SmallVector<unsigned>, bool> accs;
  for (size_t i = 0; i < offsets.size(); ++i) {
    uniqueOffsets.insert({offsets[i], i});
  }
  // reduce within threads
  for (const auto &[_, i] : uniqueOffsets) {
    SmallVector<unsigned> key = offsets[i];
    key[axis] = 0;
    accs[key] = true;
  }

  return accs.size();
}

bool AscendReduceOpHelper::isSharedMemoryReductionPreferred() {
  // This method determines if for Ascend device shared-memory
  // reduction, specifically in reduceWithinWarps phase is profitable
  // over butterfly implementation. There should be a way to access
  // target device information for this kind of detail, so when possible
  // this method should be moved accordingly.
  ReduceOp op = getOperation();
  unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
  auto mod = op->getParentOfType<ModuleOp>();
  int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  int numWarps = triton::gpu::lookupNumWarps(op);
  bool Result = false;
  Operation *BaseOp = op.getOperation();
  // Variadic outputs are not supported for this transformation.
  if (op.getNumResults() > 1)
    return false;
  // If this optimization is turned off then return false.
  if (util::getPassColumnDigit(BaseOp, "reduce-op") < 2)
    return false;

  unsigned numAccs = getAccumulatorCount();
   
  // Determine if size is multiple of 128, which would result in shared
  // memory reduction without bank conflicts on data access. Also, check if
  // the number of working threads is a multiple of 128.
  unsigned workingThreads = numLanes * numWarps / sizeIntraWarps;
  unsigned size = numLanes * numWarps * numAccs;
  unsigned numReductions = numAccs * sizeIntraWarps;
  bool isMultipleOf128 = ((size % 128) == 0) && ((workingThreads % 128) == 0);
  if (isMultipleOf128 && (srcShape.size() >= 2) && (numReductions > 8) &&
      isWarpSynchronous()) {
    Result = true;
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
    // each thread. The exact space needed is the number of threads in
    // total times the number of accumulators per thread.
    ReduceOp op = getOperation();
    auto mod = op->getParentOfType<ModuleOp>();
    int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    int numWarps = triton::gpu::lookupNumWarps(op);
    unsigned numAccs = getAccumulatorCount();
    smemShape.push_back(numLanes);
    smemShape.push_back(numWarps * numAccs);
    return smemShape;
  }

  smemShape = convertType<unsigned>(srcShape);
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

} // namespace triton::ascend
} // namespace mlir
