add_bishengir_dialect_library(BiShengIRArithTransforms
  NormalizeArith.cpp
  LiftArithIndexCast.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Arith/Transforms

  DEPENDS
  BiShengIRArithTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  )
