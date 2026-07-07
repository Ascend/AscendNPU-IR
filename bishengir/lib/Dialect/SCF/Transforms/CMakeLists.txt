add_bishengir_dialect_library(BiShengIRSCFTransforms
  BufferizableOpInterfaceImpl.cpp
  CanonicalizeIterArg.cpp
  LegalizeLoopIterArg.cpp
  MapForToForall.cpp
  RemoveRedundantLoopInit.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/SCF/Transforms

  DEPENDS
  BiShengIRSCFTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  BiShengIRSCFUtils
  BiShengIRScopeDialect
  MLIRSCFDialect
  MLIRIR
  MLIRDialect
  MLIRPass
  MLIRBufferizationTransforms
)
