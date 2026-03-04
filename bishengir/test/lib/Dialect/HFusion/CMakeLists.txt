# Exclude tests from libMLIR.so
add_bishengir_library(BishengIRTestDialectHFusion
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
  MLIRMathExtDialect
  MLIRTransforms
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRHFusionDialect
  MLIRHFusionUtils
  )

target_include_directories(BishengIRTestDialectHFusion PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)