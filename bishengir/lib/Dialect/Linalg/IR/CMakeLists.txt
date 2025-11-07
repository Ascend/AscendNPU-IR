add_bishengir_dialect_library(BiShengIRLinalgCanonicalizations
  LinalgCanonicalizations.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Linalg

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
)