#ifndef BISHENGIR_CONVERSION_TRITONASCENDGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define BISHENGIR_CONVERSION_TRITONASCENDGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::ascend {

void populateAscendReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          const TargetInfoBase &targetInfo,
                                          PatternBenefit benefit);

void populateAscendElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit);
} // namespace mlir::triton::ascend

#endif // BISHENGIR_CONVERSION_TRITONASCENDGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
