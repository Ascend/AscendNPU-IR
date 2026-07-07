  add_bishengir_conversion_library(MLIRHACCToLLVM
  HACCToLLVM.cpp
  ConversionUtils.cpp
  GetOrCreateUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HACCToLLVM

  DEPENDS
  BiShengIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRHFusionDialect
  MLIRHIVMDialect
  MLIRHIVMTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMathExtDialect
  MLIRMemRefToLLVM
  MLIRSCFDialect
  )
