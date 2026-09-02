add_bishengir_dialect_library(BiShengIRTritonTransforms
  PARTIAL_SOURCES_INTENDED

  DecomposeFRem.cpp

  DEPENDS
  BiShengIRTritonTransformsIncGen
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRHIVMRegbaseIntrinsDialect
  MLIRNVVMDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
