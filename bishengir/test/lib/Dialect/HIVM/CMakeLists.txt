# Exclude tests from libMLIR.so
add_bishengir_library(BiShengIRTestDialectHIVM
  TestFlattenInterface.cpp
  TestHIVMDimensionAnalyzer.cpp
  TestHIVMTransforms.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRHIVMDialect
  BiShengIRHIVMUtils
  )

target_include_directories(BiShengIRTestDialectHIVM PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)