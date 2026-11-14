if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  list(APPEND BS_FEAT_LINK_LIBS BiShengIRLinalgDialectExt)
endif()

# MatmulMx decomposition queries this catalogue from the dialect. Standalone
# IR builds never configure lib/Target, so that build compiles the query here.
set(MMAD_CAPABILITY_SOURCES)
set(MMAD_CAPABILITY_LIBS)
set(MMAD_CAPABILITY_DEPENDS
  BiShengIRMmadCapabilityIncGen
  BiShengIRMmadCapabilityBindingIncGen
)
if(BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  # The generated query includes NPUArchitecture.h. This build compiles the
  # query here, so it has to wait for that header.
  list(APPEND MMAD_CAPABILITY_SOURCES
       ../../../Target/NPU/Cube/Mmad/MmadCapability.cpp)
  list(APPEND MMAD_CAPABILITY_DEPENDS BiShengIRNPUArchitectureIncGen)
else()
  list(APPEND MMAD_CAPABILITY_LIBS BiShengIRNPUMmadCapability)
endif()

add_bishengir_dialect_library(BiShengIRHFusionDialect
  HFusionDialect.cpp
  HFusionOps.cpp
  HFusionImpl.cpp
  ${MMAD_CAPABILITY_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRSymbolIncGen
  BiShengIRHFusionAttributesIncGen
  BiShengIRHFusionEnumsIncGen
  BiShengIRHFusionOpsIncGen
  BiShengIRHFusionStructuredOpsIncGen
  ${MMAD_CAPABILITY_DEPENDS}

  LINK_LIBS PUBLIC
  ${BS_FEAT_LINK_LIBS}
  BiShengIRHACCDialect
  BiShengIRHACCUtils
  BiShengIRSymbolDialect
  BiShengIRMathExtDialect
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRSideEffectInterfaces
  ${MMAD_CAPABILITY_LIBS}
  )
