add_bishengir_conversion_library(BiShengIRFuncToLLVM
  FuncToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/FuncToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformUtils
  )
