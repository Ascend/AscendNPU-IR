add_bishengir_library(BiShengIRHIVMCLibA3
  HIVMCMainA3.cpp
  PassPipelineA3.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/hivmc


)

# Vendor header (bisheng-llc.h) for the Native bridge.
if(BISHENG_LLC_INCLUDE_DIR)
  target_include_directories(BiShengIRHIVMCLibA3 PRIVATE ${BISHENG_LLC_INCLUDE_DIR})
endif()
