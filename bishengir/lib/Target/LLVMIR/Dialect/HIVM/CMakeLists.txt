add_bishengir_translation_library(MLIRHIVMToLLVMIRTranslation
  HIVMToLLVMIRTranslation.cpp

  DEPENDS
  BiShengIRHIVMConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  )
