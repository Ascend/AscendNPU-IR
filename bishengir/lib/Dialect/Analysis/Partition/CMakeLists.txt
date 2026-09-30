add_bishengir_dialect_library(BiShengIRPartition
  VFFusionHIVM/VFFusionHIVMPass.cpp
  VFFusionHIVM/LinearScanFusion.cpp

  DEPENDS
  MLIRAnalysisIncGen

  LINK_LIBS PUBLIC
  BiShengIRHACCUtils
  BiShengIRHIVMDialect
  BiShengIRVFFusion
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
