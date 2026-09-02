#ifndef TRITON_ASCEND_GPU_TO_LLVM_PASSES_H_
#define TRITON_ASCEND_GPU_TO_LLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/Function.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton::ascend {
#define GEN_PASS_DECL
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "bishengir/Conversion/TritonAscendGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton::ascend

#endif // TRITON_ASCEND_GPU_TO_LLVM_PASSES_H_
