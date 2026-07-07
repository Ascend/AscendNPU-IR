add_bishengir_conversion_library(MLIRArithToHIVMLLVM
  ArithToHIVMLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bsihengir/Conversion/ArithToHIVMLLVM

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRArithDialect
  MLIRHIVMDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
