add_bishengir_dialect_library(MLIRHIVMRegbaseIntrinsUtils
  RegbaseUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMRegbaseIntrins

  LINK_LIBS PUBLIC
  MLIRHIVMRegbaseIntrinsDialect
)
