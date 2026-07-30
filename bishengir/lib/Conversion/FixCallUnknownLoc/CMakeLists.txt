add_bishengir_conversion_library(MLIRFixCallUnknownLoc
  FixCallUnknownLoc.cpp

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLLVMDialect
)
