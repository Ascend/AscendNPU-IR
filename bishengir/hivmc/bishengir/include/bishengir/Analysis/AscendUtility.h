#ifndef ASCEND_ANALYSIS_UTILITY_H
#define ASCEND_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Analysis/Utility.h"

namespace mlir {
namespace triton {
namespace ascend {
class AscendReduceOpHelper : public ReduceOpHelper {
public:
  AscendReduceOpHelper(triton::ReduceOp op) : ReduceOpHelper(op) {}

  // The shape of the shared memory space needed for the reduction.
  SmallVector<unsigned> getScratchRepShape() override;

  // Check if prefer shared memory (UB) for reduction.
  bool isSharedMemoryReductionPreferred();
};
} // namespace ascend
} // namespace triton
} // namespace mlir

#endif // ASCEND_ANALYSIS_UTILITY_H
