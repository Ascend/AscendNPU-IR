# Exclude tests from libMLIR.so
add_bishengir_library(BiShengIRTestDialectHFusion
  TestAssignFusionKindAttrsPass.cpp
  TestBufferUtils.cpp
  TestCanFuse.cpp
  TestFunctionCall.cpp
  ValidPropagatedReshape.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRAnalysis
  MLIRDialectUtils
  MLIRFuncDialect
  BiShengIRMathExtDialect
  MLIRTransforms
  BiShengIRHACCDialect
  BiShengIRHACCUtils
  BiShengIRHFusionDialect
  BiShengIRHFusionUtils
  )

target_include_directories(BiShengIRTestDialectHFusion PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)
