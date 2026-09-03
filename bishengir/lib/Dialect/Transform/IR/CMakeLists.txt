add_bishengir_dialect_library(BiShengIRTransformOps
  TransformOps.cpp
  DialectExtension.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Transform/IR

  DEPENDS
  BiShengIRTransformOpsIncGen
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
