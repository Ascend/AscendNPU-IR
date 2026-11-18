add_bishengir_dialect_library(BiShengIRTensorViewTransforms
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  BiShengIRTensorViewDialect
  BiShengIRHIVMDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRMemRefDialect
)
