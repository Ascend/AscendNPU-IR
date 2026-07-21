add_bishengir_dialect_library(BiShengIRTritonExtIR
  Dialect.cpp
  FractalLinearLayoutConversions.cpp

  DEPENDS
  BiShengIRTritonExtDialectIncGen
  BiShengIRTritonExtAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  TritonTools
  MLIRIR
  MLIRSupport
)
