#ifndef ASCEND_ANALYSIS_ALLOCATION_H
#define ASCEND_ANALYSIS_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::ascend {

void AscendAllocationSharedMemCheckFn(Operation *op, int allocatedSharedMemorySize);

unsigned AscendAllocationAnalysisScratchSizeFn(Operation *op);

} // namespace mlir::triton::ascend

#endif // ASCEND_ANALYSIS_ALLOCATION_H
