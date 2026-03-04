set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  MLIRMathExtDialect
  BiShengIRTensorDialect
  )

add_bishengir_conversion_library(BiShengIRLinalgToHFusion
  IS_OPEN_SOURCE

  LinalgToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/LinalgToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
)
