add_bishengir_translation_library(MLIRHIVMToLLVMIRTranslation
  HIVMToLLVMIRTranslation.cpp

  DEPENDS
  MLIRHIVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRHIVMDialect
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  )
