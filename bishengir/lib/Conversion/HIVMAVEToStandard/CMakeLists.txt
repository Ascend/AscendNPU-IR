add_bishengir_conversion_library(MLIRHIVMAVEToStandard
  HIVMAVEToStandard.cpp

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRFuncDialect
  MLIRAVEDialect
)
