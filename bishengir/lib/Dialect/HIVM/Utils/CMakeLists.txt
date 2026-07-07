add_bishengir_dialect_library(BiShengIRHIVMUtils
  Utils.cpp
  RegbaseUtils.cpp
  MemrefAliasAnalysisState.cpp
  MultiBufferLoopAdapter.cpp
  WorklistBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils

  LINK_LIBS PUBLIC
  BiShengIRScopeDialect
  MLIRAnnotationDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  BiShengIRHACCDialect
  BiShengIRHIVMDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
  MLIRSCFDialect
  MLIRTensorDialect
)
