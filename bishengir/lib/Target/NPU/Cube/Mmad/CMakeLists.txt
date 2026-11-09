add_bishengir_library(BiShengIRNPUMmadCapability
  MmadCapability.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Target/NPU

  DEPENDS
  BiShengIRMmadCapabilityIncGen
  BiShengIRNPUArchitectureIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
