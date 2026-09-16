add_bishengir_translation_library(MLIRHIVMRegbaseIntrinsToLLVMIRTranslation
  HIVMRegbaseIntrinsToLLVMIRTranslation.cpp

  DEPENDS
  MLIRHIVMRegbaseConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRHIVMRegbaseIntrinsDialect
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
  )
