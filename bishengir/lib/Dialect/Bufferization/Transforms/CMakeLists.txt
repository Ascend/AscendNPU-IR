add_bishengir_dialect_library(BiShengIRBufferizationTransforms
  BufferizationTilingInterfaceImpl.cpp
  BufferizeModuleOp.cpp
  FuncBufferizableOpInterfaceImpl.cpp
  NormalizeToTensorOp.cpp
  TensorCopyInsertion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Bufferization/Transforms

  DEPENDS
  BiShengIRBufferizationTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRScopeDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTilingInterface
)
