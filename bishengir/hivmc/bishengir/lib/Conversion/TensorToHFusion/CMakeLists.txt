add_bishengir_conversion_library(BiShengIRTensorToHFusion

  TensorToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TensorToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRHFusionDialect
  MLIRRewrite
  MLIRPass
)
