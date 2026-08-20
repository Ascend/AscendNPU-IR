add_bishengir_dialect_library(MLIRAVEUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMAVE

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAVEDialect
)
