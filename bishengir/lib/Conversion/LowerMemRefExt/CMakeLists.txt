add_bishengir_conversion_library(BiShengIRMemRefExtLowering
  LowerMemRefExt.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRMemRefExtDialect
)
