set(BS_FEAT_LIBS)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND BS_FEAT_LIBS
    TritonToTritonGPU
    TritonGPUTransforms
    TritonNVIDIAGPUToLLVM
    TritonAscendGPUToLLVM
    NVGPUToLLVM
    ProtonToProtonGPU
    ProtonGPUToLLVM
    ProtonAscendGPUToLLVM
  )
endif()

add_bishengir_dialect_library(BiShengIRTritonPipelines
  TritonPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Triton
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TritonAscendGPUToLLVM

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  ${BS_FEAT_LIBS}
)
