add_bishengir_conversion_library(MLIRHIVMToLLVM
  ConvertMemRefToBarePtr.cpp
  HIVMToLLVM.cpp
  TritonGlobalKernelArgsToLLVM.cpp
  AppendUsePrintDebugData.cpp
  AppendSanitizerData.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HIVMToLLVM

  DEPENDS
  BiShengIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRLLVMCommonConversion
  BiShengIRFuncToLLVM
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRArithToHIVMLLVM
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRHIVMDialect
  MLIRHIVMTransforms
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRSCFDialect
  )
