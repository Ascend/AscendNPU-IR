add_bishengir_dialect_library(BiShengIRBufferizationTransforms
  BufferizationTilingInterfaceImpl.cpp
  FuncBufferizableOpInterfaceImpl.cpp
 
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Bufferization/Transforms
 
  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTilingInterface
)
