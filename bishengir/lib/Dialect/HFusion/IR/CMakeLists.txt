if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  list(APPEND BS_FEAT_LINK_LIBS BiShengIRLinalgDialectExt)
  list(APPEND BS_FEAT_LINK_LIBS BiShengIRHACCDialect)
endif()

add_bishengir_dialect_library(BiShengIRHFusionDialect
  HFusionDialect.cpp
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
  ${BS_FEAT_LINK_LIBS}
  BiShengIRSymbolDialect
  BiShengIRMathExtDialect
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  )
