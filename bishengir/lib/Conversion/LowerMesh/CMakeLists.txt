add_bishengir_conversion_library(BiShengIRLowerMesh
  LowerMesh.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMeshDialect
  BiShengIRMeshDialect
)
