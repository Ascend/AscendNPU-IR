set(BISHENGIR_PRIVATE_LIBS
  MLIRAnnotationDialect
  MLIRMathExtDialect
  )

add_bishengir_dialect_library(MLIRHFusionTransformOps
  IS_OPEN_SOURCE

  DialectExtension.cpp
  HFusionTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion/TransformOps

  DEPENDS
  MLIRHFusionTransformOpsIncGen
  MLIRHFusionTransformEnumsIncGen
  MLIRLinalgTransformEnumsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRHFusionDialect
  MLIRHFusionTransforms
  MLIRHFusionUtils
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  )