add_bishengir_dialect_library(BiShengIRVectorTransforms
  MaterializeVectorWriteToDestination.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Vector/Transforms

  DEPENDS
  BishengIRVectorTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRScopeUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
)
