add_bishengir_unittest(BiShengIRNPUArchitectureTests
  NPUArchitectureFeatureTest.cpp
  NPUArchitectureTest.cpp
)

add_dependencies(BiShengIRNPUArchitectureTests
  BiShengIRNPUArchitectureIncGen
  BiShengIRNPUArchitectureFeaturesIncGen
  BiShengIRNPUTargetSpecIncGen
)

target_link_libraries(BiShengIRNPUArchitectureTests PRIVATE
  BiShengIRHACCDialect
  BiShengIRHACCTransforms
  BiShengIRHACCUtils
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
)
