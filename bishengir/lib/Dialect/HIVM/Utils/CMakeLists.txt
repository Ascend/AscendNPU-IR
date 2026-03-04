add_bishengir_dialect_library(MLIRHIVMUtils
  Utils.cpp
  RegbaseUtils.cpp
  MemrefAliasAnalysisState.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRHACCDialect
  MLIRHIVMDialect
  MLIRMemRefDialect
  MLIRMemRefExtDialect
)
