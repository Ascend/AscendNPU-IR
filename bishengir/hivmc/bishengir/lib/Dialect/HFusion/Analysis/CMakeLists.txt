set(BISHENGIR_PRIVATE_LIBS
  BiShengIRAnalysis
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRTensorTransforms
  BishengIRTransform
  MLIRHACCDialect
  MLIRHACCUtils
)

add_bishengir_dialect_library(MLIRHFusionAnalysis
  IS_OPEN_SOURCE
  ReshapeAnalyzer/ReshapeAnalyzer.cpp
  DimensionAnalyzer/DimensionAnalyzer.cpp
  DimensionAnalyzer/DimensionMarker.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  MLIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRAnnotationDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRHFusionDialect
  MLIRHFusionUtils
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorUtils
)
