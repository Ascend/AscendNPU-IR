add_bishengir_conversion_library(BiShengIRArithToHFusion
  ArithToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/ArithToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRMathExtDialect
  BiShengIRTensorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  BiShengIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
)
