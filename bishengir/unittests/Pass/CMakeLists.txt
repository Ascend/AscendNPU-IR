add_bishengir_unittest(BiShengIRPassTests
  FilterPassesTest.cpp
  PassExecutionPolicyTest.cpp
)

target_link_libraries(BiShengIRPassTests PRIVATE
  BiShengIRAnnotationDialect
  BiShengIRDialectUtils
  BiShengIRExecutionEngineTransforms
  BiShengIRPass
  BiShengIRHIVMDialect
  BiShengIRHIVMRegbasePipelines
  BiShengIRHIVMTransforms
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRFuncDialect
)
