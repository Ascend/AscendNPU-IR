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
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRHIVMDialect
  MLIRHIVMUtils
  MLIRIR
  MLIRPass
)
