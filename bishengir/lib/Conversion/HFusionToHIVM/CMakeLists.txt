add_bishengir_conversion_library(BiShengIRHFusionToHIVM
  HFusionToHIVM.cpp
  Matmul.cpp
  Reduction.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  MLIRIR
  MLIRBufferizationTransforms
  MLIRLinalgDialect
  BiShengIRHFusionDialect
)
