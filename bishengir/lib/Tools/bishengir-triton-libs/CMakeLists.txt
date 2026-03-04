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

# This library is for linking all triton related libraries
# We cannot directly use Triton's `TRITON_LIBS` property because
# Triton is a seperate project so the GLOBAL PROPERTY is not shared.
add_bishengir_dialect_library(TritonAggregateLib
  TritonAggregateLib.cpp

  LINK_LIBS PUBLIC
  ${triton_libs}
)
