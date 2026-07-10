add_bishengir_dialect_library(BiShengIRVectorTransforms
  MaterializeVectorWriteToDestination.cpp
  VectorTransferLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Vector/Transforms

  DEPENDS
  BishengIRVectorTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRScopeUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRVectorTransforms
)
