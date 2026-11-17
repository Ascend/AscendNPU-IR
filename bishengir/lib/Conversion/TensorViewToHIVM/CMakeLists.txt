add_bishengir_conversion_library(BiShengIRTensorViewToHIVM
  TensorViewToHIVM.cpp

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRTensorViewDialect
  BiShengIRHIVMDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
)
