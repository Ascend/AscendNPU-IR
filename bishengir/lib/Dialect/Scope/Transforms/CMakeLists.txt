add_bishengir_dialect_library(BiShengIRScopeTransforms
  BufferizableOpInterfaceImpl.cpp
  OutlineScope.cpp
  InlineScope.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Scope/Transforms

  DEPENDS
  BiShengIRScopeTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRScopeDialect
  MLIRBufferizationDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
)
