add_bishengir_dialect_library(MLIRHIVMUtils
  Utils.cpp
  RegbaseUtils.cpp
  MemrefAliasAnalysisState.cpp
  MultiBufferLoopAdapter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFunctionInterfaces
  MLIRHACCDialect
  MLIRHIVMDialect
  MLIRMemRefDialect
  MLIRMemRefExtDialect
  MLIRSCFDialect
)
