add_bishengir_translation_library(BiShengIRBMUToLLVMIRTranslation
  BMUToLLVMIRTranslation.cpp

  DEPENDS
  BiShengIRBMUConversionsIncGen
  BiShengIRBMUIntrinOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRBMUDialect
  MLIRLLVMDialect
  MLIRTargetLLVMIRExport
)
