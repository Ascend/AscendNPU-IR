#pragma once
#include "mlir/Config/mlir-config.h"

#if !BSPUB_DAVINCI_BISHENGIR
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#endif

#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"

#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#if !BSPUB_DAVINCI_BISHENGIR
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#endif
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"

#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#if !BSPUB_DAVINCI_BISHENGIR
// Below headers will allow registration to ROCm passes
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#endif

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
#if !BSPUB_DAVINCI_BISHENGIR
void registerAMDTestAlignmentPass();
#endif
void registerTestAllocationPass();
void registerTestMembarPass();
#if !BSPUB_DAVINCI_BISHENGIR
void registerTestAMDGPUMembarPass();
void registerTestTritonAMDGPURangeAnalysis();
#endif
void registerTestLoopPeelingPass();
namespace proton {
void registerTestScopeIdAllocationPass();
} // namespace proton
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
#if !BSPUB_DAVINCI_BISHENGIR
  // This is also called in bishengir, so we have to disable it here.
  mlir::registerAllPasses();
#endif
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::gluon::registerGluonPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
#if !BSPUB_DAVINCI_BISHENGIR
  mlir::test::registerAMDTestAlignmentPass();
#endif
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::test::registerTestLoopPeelingPass();
#if !BSPUB_DAVINCI_BISHENGIR
  mlir::test::registerTestAMDGPUMembarPass();
  mlir::test::registerTestTritonAMDGPURangeAnalysis();
#endif
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerRelayoutTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::triton::registerAllocateSharedMemoryNvPass();
  mlir::registerLLVMDIScope();

#if !BSPUB_DAVINCI_BISHENGIR
  // Currently, we don't build AMD backed.
  // TritonAMDGPUToLLVM passes
  mlir::triton::registerAllocateAMDGPUSharedMemory();
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::registerTritonAMDGPUInThreadTranspose();
  mlir::registerTritonAMDGPUCoalesceAsyncCopy();
  mlir::registerTritonAMDGPUUpdateAsyncWaitCount();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();
  mlir::registerTritonAMDFoldTrueCmpI();
#endif

  // NVWS passes
  mlir::triton::registerNVWSTransformsPasses();

  // NVGPU transform passes
  mlir::registerNVHopperTransformsPasses();

  // Proton passes
#if !BSPUB_DAVINCI_BISHENGIR
  mlir::test::proton::registerTestScopeIdAllocationPass();
#endif
  mlir::triton::proton::registerConvertProtonToProtonGPU();
#if !BSPUB_DAVINCI_BISHENGIR
  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
  mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
#endif
  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
#if !BSPUB_DAVINCI_BISHENGIR
  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
  mlir::triton::proton::gpu::registerAddSchedBarriersPass();
#endif

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
#if !BSPUB_DAVINCI_BISHENGIR
      mlir::triton::amdgpu::TritonAMDGPUDialect,
#endif
      mlir::triton::proton::ProtonDialect,
      mlir::triton::proton::gpu::ProtonGPUDialect,
      mlir::ROCDL::ROCDLDialect, mlir::triton::gluon::GluonDialect>();
}
