set(BS_FEAT_SOURCES)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND BS_FEAT_SOURCES
    AdaptTritonIRKernel.cpp
    DecomposeReduction.cpp
    DumpFractalLayout.cpp
    FixFusedCatPass.cpp
    GetTritonMetadata.cpp
    LegalizeF16ForTriton.cpp
    OptimizeLayouts.cpp
    OptimizeLayoutsAnalysis.cpp
    OptimizeLoads.cpp
    SetBishengirSimtOptAttr.cpp
    PopulateSharedMemoryOffsetToDPX.cpp
    SIMTFastDiv.cpp
    LoopRestructureArangeOptimization.cpp
    EnableAscendDPXMMA.cpp
  )
endif()

add_bishengir_dialect_library(BiShengIRTritonTransforms
  PARTIAL_SOURCES_INTENDED

  ${BS_FEAT_SOURCES}

  DEPENDS
  BiShengIRTritonTransformsIncGen
  BiShengIRTritonOpsIncGen
  MLIRLLVMPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRTritonExtension
  TritonGPUIR
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  MLIRAscendDPXDialect
  MLIRHIVMTransformOps
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRNVVMDialect
  MLIRHIVMDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
