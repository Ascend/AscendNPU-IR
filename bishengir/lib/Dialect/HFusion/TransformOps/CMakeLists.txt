add_bishengir_dialect_library(BiShengIRHFusionTransformOps
  DialectExtension.cpp
  HFusionTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion/TransformOps

  DEPENDS
  BiShengIRHFusionTransformOpsIncGen
  BiShengIRHFusionTransformEnumsIncGen
  MLIRLinalgTransformEnumsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnnotationDialect
  BiShengIRMathExtDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  BiShengIRHFusionDialect
  BiShengIRHFusionTransforms
  BiShengIRHFusionUtils
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  )
