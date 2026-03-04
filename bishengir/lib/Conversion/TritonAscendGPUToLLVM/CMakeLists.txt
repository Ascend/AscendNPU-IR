add_bishengir_conversion_library(TritonAscendGPUToLLVM
  PARTIAL_SOURCES_INTENDED
  TritonOpToDPX.cpp
  LoadStoreOpToLLVM.cpp
  ReduceOpToLLVM.cpp
  TargetInfo.cpp
  TritonAscendGPUToLLVMPass.cpp
  ElementwiseOpToLLVM.cpp

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
  TritonAnalysis
  TritonGPUToLLVM
)
