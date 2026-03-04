add_bishengir_dialect_library(BiShengIRMeshTransforms
  LowerMeshHost.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Mesh/Transforms

  DEPENDS
  MLIRMeshPassIncGen

  LINK_LIBS PUBLIC
  MLIRMeshDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
