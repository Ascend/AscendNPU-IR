add_bishengir_conversion_library(TritonAscendGPUToDPX
  PARTIAL_SOURCES_INTENDED
  GPUOpToDPX.cpp

  DEPENDS
  BiShengIRConversionPassIncGen
  TritonAscendGPUToLLVMPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRLLVMDialect
  MLIRAscendDPXDialect
  TritonAnalysis
  TritonGPUToLLVM
)
