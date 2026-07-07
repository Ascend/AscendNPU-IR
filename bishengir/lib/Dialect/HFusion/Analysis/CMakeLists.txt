set(BISHENGIR_PRIVATE_LIBS
  BiShengIRAnalysis
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRTensorTransforms
  BishengIRTransform
  BiShengIRHACCDialect
  BiShengIRHACCUtils
)

add_bishengir_dialect_library(BiShengIRHFusionAnalysis
  IS_OPEN_SOURCE
  ReshapeAnalyzer/ReshapeAnalyzer.cpp
  DimensionAnalyzer/DimensionAnalyzer.cpp
  DimensionAnalyzer/DimensionMarker.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  BiShengIRAnnotationDialect
  BiShengIRHFusionDialect
  BiShengIRHFusionUtils
  MLIRAffineDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorUtils
)
