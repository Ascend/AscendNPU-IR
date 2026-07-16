add_bishengir_dialect_library(BiShengIRArithTransforms
  NormalizeArith.cpp
  LiftArithIndexCast.cpp
  MoveUpArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Arith/Transforms

  DEPENDS
  BiShengIRArithTransformsIncGen
  MLIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  )
