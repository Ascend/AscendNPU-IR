add_bishengir_conversion_library(TritonAscendGPUToLLVM
  PARTIAL_SOURCES_INTENDED
  TritonOpToDPX.cpp
  LoadStoreOpToLLVM.cpp
  ReduceOpToLLVM.cpp
  TargetInfo.cpp
  TritonAscendGPUToLLVMPass.cpp
  ElementwiseOpToLLVM.cpp
  FractalMemoryOpToLLVM.cpp
  SinkDPXLoadPass.cpp
  ${LLVM_EXTERNAL_TRITON_SOURCE_DIR}/lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp

  DEPENDS
  BiShengIRConversionPassIncGen
  TritonAscendGPUToLLVMPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRAscendDPXDialect
  BiShengIRTritonExtIR
  TritonAnalysis
  TritonGPUToLLVM
)
