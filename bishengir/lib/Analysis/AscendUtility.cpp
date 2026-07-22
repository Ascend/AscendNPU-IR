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

#include <algorithm>

namespace mlir {

using namespace triton::gpu;

namespace triton::ascend {

#define str_attr(str) ::mlir::StringAttr::get(ctx, (str))

// "Given a layout mapping onto dim0..dimn, remove a dimension `dim` and rename
// the rest as dim0..dimn-1".  Newer SliceEncodingAttr::toLinearLayout uses this
// helper instead of building a transform and composing it with the parent
// layout.
static LinearLayout removeStandardDim(const LinearLayout &layout, int dim) {
  auto rank = layout.getNumOutDims();
  assert(rank > 0);
  assert(dim < static_cast<int>(rank));
  auto *ctx = layout.getOutDimNames().begin()->getContext();
  auto dims = llvm::to_vector(layout.getOutDimNames());
  auto standardDims = standardOutDimNames(ctx, rank);
  assert(dims == standardDims);
  dims.erase(dims.begin() + dim);
  auto newLayout = layout.sublayout(llvm::to_vector(layout.getInDimNames()), dims);
  auto dimSizes = newLayout.getOutDims();
  auto newDims = standardOutDimNames(ctx, rank - 1);
  for (auto [i, newDim] : llvm::enumerate(newDims)) {
    dimSizes[i].first = newDim;
  }
  return LinearLayout(newLayout.getBases(), dimSizes,
                      /*isSurjective=*/false);
}

unsigned AscendReduceOpHelper::getCorrectedThreadOffsetOnReductionAxis() {
  auto srcLayout = getSrcLayout();
  auto srcShape = getSrcShape();
  unsigned reduceAxis = getOperation().getAxis();
  auto *ctx = srcLayout.getContext();
  auto srcTy = RankedTensorType::get(srcShape,
                                      getOperation().getElementTypes()[0],
                                      srcLayout);
  LinearLayout linearLayout;

  if (auto sliceEnc = dyn_cast<SliceEncodingAttr>(srcLayout)) {
    // Align with newer community SliceEncodingAttr::toLinearLayout: build the
    // sliced LinearLayout through parent-layout + removeStandardDim, then use
    // that layout to derive the reduce shuffle offset from lane bases.

    // First compute the linear layout for this layout's parent.
    SmallVector<int64_t> parentShape(srcShape);
    parentShape.insert(parentShape.begin() + sliceEnc.getDim(), 1);
    LinearLayout parentLL =
        triton::gpu::toLinearLayout(parentShape, sliceEnc.getParent());
    auto sliceLL = removeStandardDim(parentLL, sliceEnc.getDim());

    // Step 3: Along the "register" dim, remove any all-zero bases.
    auto bases = sliceLL.getBases();
    std::vector<std::vector<int>> newRegBases;
    for (const auto &basis : bases[str_attr("register")]) {
      if (llvm::any_of(basis, [](int b) { return b != 0; })) {
        newRegBases.push_back(basis);
      }
    }
    bases[str_attr("register")] = newRegBases;
    linearLayout =
        LinearLayout(std::move(bases), llvm::to_vector(sliceLL.getOutDimNames()));
  } else {
    linearLayout = triton::gpu::toLinearLayout(srcTy);
  }

  auto kLane = StringAttr::get(ctx, "lane");
  const auto &bases = linearLayout.getBases();
  const auto &lanes = bases.find(kLane)->second;
  auto offset = 1;
  for (const auto &lane : lanes) {
    if (lane[reduceAxis] != 0)
      break;
    offset *= 2;
  }
  return offset;
}

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
  unsigned numLanes = (unsigned)triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  unsigned numWarps = (unsigned)triton::gpu::lookupNumWarps(op);
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
