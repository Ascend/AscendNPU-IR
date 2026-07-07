set(BISHENGIR_PRIVATE_LIBS
  BiShengIRLinalgDialectExt
  BiShengIRSymbolDialect
  BiShengIRHACCDialect
  BiShengIRHMAPDialect
  BiShengIRMathExtDialect
)

add_bishengir_dialect_library(BiShengIRHFusionDialect
  IS_OPEN_SOURCE

  HFusionDialect.cpp
  HFusionImpl.cpp
  HFusionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRSymbolIncGen
  BiShengIRHFusionAttributesIncGen
  BiShengIRHFusionEnumsIncGen
  BiShengIRHFusionOpsIncGen
  BiShengIRHFusionStructuredOpsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
)
