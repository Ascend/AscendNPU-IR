add_bishengir_unittest(BiShengIRHIVMCUtilityTests
  HIVMCUtilityTest.cpp
)

target_link_libraries(BiShengIRHIVMCUtilityTests PRIVATE
  BiShengIRHIVMCLib
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  LLVMAsmParser
  LLVMCore
  LLVMSupport
)
