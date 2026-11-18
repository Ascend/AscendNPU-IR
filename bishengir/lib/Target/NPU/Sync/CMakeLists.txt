add_bishengir_library(BiShengIRNPUSyncCapability
  SyncCapability.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Target/NPU

  DEPENDS
  BiShengIRSyncCapabilityIncGen
  BiShengIRNPUArchitectureIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
