add_bishengir_conversion_library(ProtonAscendGPUToLLVM
    AllocateProtonAscendGlobalScratchBuffer.cpp
    TargetInfo.cpp
    ../TritonAscendGPUToLLVM/TargetInfo.cpp
    ProtonGPUOpToLLVM.cpp
    ProtonAscendGPUToLLVM.cpp

    DEPENDS
    BiShengIRConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
)
