add_bishengir_dialect_library(BiShengIRHFusionAnalysis
  ReshapeAnalyzer/ReshapeAnalyzer.cpp
  DimensionAnalyzer/DimensionAnalyzer.cpp
  DimensionAnalyzer/DimensionMarker.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnalysis
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRTensorTransforms
  BiShengIRTransform
  BiShengIRHACCDialect
  BiShengIRHACCUtils
  MLIRAffineDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  BiShengIRHFusionDialect
  BiShengIRHFusionUtils
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRPass
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorUtils
)
