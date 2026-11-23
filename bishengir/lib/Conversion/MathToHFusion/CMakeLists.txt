add_bishengir_conversion_library(BiShengIRMathToHFusion
  MathToHFusion.cpp
  ScalarMathPromotion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/MathToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRHACCUtils
  BiShengIRMathExtDialect
  BiShengIRTensorDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  BiShengIRHFusionDialect
  MLIRTransformUtils
  MLIRRewrite
  MLIRPass
)

target_include_directories(BiShengIRMathToHFusion PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Template/include)
