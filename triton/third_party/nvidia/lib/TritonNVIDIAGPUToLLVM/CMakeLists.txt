set(BS_SOURCE_FILES)
if(NOT BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND BS_SOURCE_FILES
    TensorMemoryToLLVM.cpp
  )
endif()

add_triton_library(TritonNVIDIAGPUToLLVM
    PARTIAL_SOURCES_INTENDED

    ConvertLayoutOpToLLVM.cpp
    ConvertWarpSpecializeToLLVM.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/MMAv5.cpp
    DotOpToLLVM/WGMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    TMAToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    Fp4ToFpOpToLLVM.cpp
    TargetInfo.cpp
    Allocation.cpp
    ${BS_SOURCE_FILES}

    DEPENDS
    TritonNVIDIAGPUConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    TritonInstrumentToLLVM
    MLIRReconcileUnrealizedCasts
    NVGPUIR
    MLIRUBToLLVM
)
