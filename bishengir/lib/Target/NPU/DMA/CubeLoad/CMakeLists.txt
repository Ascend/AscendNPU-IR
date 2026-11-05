add_bishengir_library(BiShengIRNPUCubeLoadCapability
  CubeLoadCapability.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Target/NPU

  DEPENDS
  BiShengIRCubeLoadCapabilityIncGen
  BiShengIRNPUArchitectureIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
