add_bishengir_conversion_library(MLIRArithToHIVMAVE
  ArithToHIVMAVE.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToHIVMAVE

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRHIVMDialect
  MLIRHIVMUtils
)
