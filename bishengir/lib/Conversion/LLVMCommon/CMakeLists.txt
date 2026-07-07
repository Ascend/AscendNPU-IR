add_bishengir_conversion_library(BiShengIRLLVMCommonConversion
  TypeConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTransforms
  MLIRLLVMCommonConversion
  )
