set(BISHENGIR_PRIVATE_LIBS
  BiShengIRHIVMDialect
  )

add_bishengir_conversion_library(BiShengIRHFusionToHIVM
  IS_OPEN_SOURCE

  HFusionToHIVM.cpp
  Matmul.cpp
  Reduction.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRIR
  MLIRBufferizationTransforms
  MLIRLinalgDialect
  BiShengIRHFusionDialect
)
