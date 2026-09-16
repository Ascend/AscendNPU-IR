set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  )

add_bishengir_conversion_library(BiShengIRGPUToHFusion
  IS_OPEN_SOURCE

  GPUToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/GPUToHFusion

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
  MLIRGPUDialect
  MLIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
)
