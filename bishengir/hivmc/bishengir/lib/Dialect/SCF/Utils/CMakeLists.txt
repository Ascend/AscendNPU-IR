add_bishengir_dialect_library(BiShengIRSCFUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/SCF

  LINK_LIBS PUBLIC
  MLIRSCFDialect
)
