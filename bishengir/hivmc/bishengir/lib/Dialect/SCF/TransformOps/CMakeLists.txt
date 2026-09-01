add_bishengir_dialect_library(BiShengIRSCFTransformOps
  SCFTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/SCF/TransformOps

  DEPENDS
  BiShengIRSCFTransformOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRSCFUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformDialect
  MLIRTensorDialect
)
