set(BISHENGIR_PRIVATE_LIBS
  BiShengIRLinalgDialectExt
  BiShengIRSymbolDialect
  MLIRHACCDialect
  MLIRHMAPDialect
  MLIRMathExtDialect
  )

add_bishengir_dialect_library(MLIRHFusionDialect
  IS_OPEN_SOURCE

  HFusionDialect.cpp
  HFusionImpl.cpp
  HFusionOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRSymbolIncGen
  MLIRHFusionAttributesIncGen
  MLIRHFusionEnumsIncGen
  MLIRHFusionOpsIncGen
  MLIRHFusionStructuredOpsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  )
