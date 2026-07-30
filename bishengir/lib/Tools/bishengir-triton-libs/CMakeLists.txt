list(APPEND triton_libs
  GluonIR
  GluonTransforms
  TritonInstrumentTransforms
  TritonLLVMIR
  TritonTestAnalysis
  TritonTestDialect
  NVGPUIR
  NVHopperTransforms
)

# Link proton libraries so that the proton TableGen IncGen targets are
# transitively ordered before any tool (e.g. bishengir-opt, bishengir-compile)
# that includes RegisterTritonDialects.h and therefore depends on the
# generated proton .h.inc files. Without this, ninja may schedule the tool's
# .cpp.o compilation before the .h.inc files are generated, causing
# "fatal error: '....h.inc' file not found" under RelWithDebInfo builds.
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND triton_libs
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
    ProtonToProtonGPU
    ProtonGPUToLLVM
    ProtonGPUTransforms
  )
endif()

# This library is for linking all triton related libraries
# We cannot directly use Triton's `TRITON_LIBS` property because
# Triton is a seperate project so the GLOBAL PROPERTY is not shared.
add_bishengir_dialect_library(TritonAggregateLib
  TritonAggregateLib.cpp

  LINK_LIBS PUBLIC
  ${triton_libs}
)
