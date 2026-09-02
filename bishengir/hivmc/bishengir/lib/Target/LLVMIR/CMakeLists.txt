add_subdirectory(Dialect)

set(LLVM_OPTIONAL_SOURCES
  ConvertToLLVMIR.cpp
)

add_bishengir_translation_library(BiShengIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  MLIRHACCToLLVMIRTranslation
  MLIRHIVMToLLVMIRTranslation
  MLIRHIVMRegbaseIntrinsToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
)
