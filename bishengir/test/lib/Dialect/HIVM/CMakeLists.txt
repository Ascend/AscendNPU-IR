# Exclude tests from libMLIR.so
add_bishengir_library(BishengIRTestDialectHIVM
  TestFlattenInterface.cpp
  TestHIVMDimensionAnalyzer.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRHIVMDialect
  MLIRHIVMUtils
  )

target_include_directories(BishengIRTestDialectHIVM PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)