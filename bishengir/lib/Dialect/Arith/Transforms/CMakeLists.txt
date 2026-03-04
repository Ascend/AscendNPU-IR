add_mlir_library(BishengIRArithTransforms
  MeshShardingInterfaceImpl.cpp
  NormalizeArith.cpp
  LiftArithIndexCast.cpp
  MoveUpArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Arith/Transforms

  DEPENDS
  MLIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRShardingInterface
)
