add_bishengir_unittest(BiShengIRPassTests
  FilterPassesTest.cpp
  PassExecutionPolicyTest.cpp
)

target_link_libraries(BiShengIRPassTests PRIVATE
  BiShengIRAnnotationDialect
  BiShengIRDialectUtils
  BiShengIRPass
  BiShengIRHIVMRegbasePipelines
  MLIRIR
  MLIRPass
  MLIRFuncDialect
)
