add_bishengir_conversion_library(BiShengIRTensorToHIVM

  TensorToHIVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TensorToHIVM

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRHIVMDialect
  MLIRRewrite
  MLIRPass
)
