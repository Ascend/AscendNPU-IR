add_bishengir_translation_library(MLIRHACCToLLVMIRTranslation
  HACCToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRHACCDialect
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  )
