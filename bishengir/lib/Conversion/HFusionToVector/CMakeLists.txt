add_bishengir_conversion_library(MLIRHFusionToVector
  HFusionToVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HFusionToVector

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRHIVMDialect
  MLIRHIVMUtils
)
