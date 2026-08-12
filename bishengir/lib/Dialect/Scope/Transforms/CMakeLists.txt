add_bishengir_dialect_library(BiShengIRScopeTransforms
  BufferizableOpInterfaceImpl.cpp
  InlineScope.cpp
  OutlineScope.cpp
  TransformOpForSIMT.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Scope/Transforms

  DEPENDS
  BiShengIRScopeTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRScopeDialect
  BiShengIRHACCUtils
  BiShengIRHIVMDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
)
