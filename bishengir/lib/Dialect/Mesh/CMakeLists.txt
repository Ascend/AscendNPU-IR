add_bishengir_dialect_library(BiShengIRMeshDialect
  Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Mesh

  LINK_LIBS PUBLIC
  MLIRMeshDialect
  MLIRIR
)

add_subdirectory(Transforms)
