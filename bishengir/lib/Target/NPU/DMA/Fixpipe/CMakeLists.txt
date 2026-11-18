add_bishengir_library(BiShengIRNPUFixpipeCapability
  FixpipeCapability.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Target/NPU

  DEPENDS
  BiShengIRFixpipeCapabilityIncGen
  BiShengIRNPUArchitectureIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
