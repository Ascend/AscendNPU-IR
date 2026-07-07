add_bishengir_dialect_library(BishengIRVectorTransforms
  VectorTransferLowering.cpp
  FlattenVectorOps.cpp
  MaterializeVectorWriteToDestination.cpp
  NormalizeVector.cpp
  PeelLoopsContainingTranspose.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Vector/Transforms
  DEPENDS
  MLIRVectorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRVectorDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)