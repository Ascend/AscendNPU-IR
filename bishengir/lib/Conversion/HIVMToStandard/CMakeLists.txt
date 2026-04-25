add_bishengir_conversion_library(MLIRHIVMToStandard
  HIVMToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HIVMToStandard

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRFuncDialect
  MLIRHIVMDialect
  MLIRMemRefDialect
)
