add_bishengir_dialect_library(BiShengIRAnalysisTransforms
  TransformOps.cpp
  DialectExtension.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Analysis/Transforms

  DEPENDS
  BiShengIRAnalysisTransformsIncGen
  MLIRLinalgTransformEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRSCFUtils
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformDialect
  MLIRTransformDialectUtils
  )
