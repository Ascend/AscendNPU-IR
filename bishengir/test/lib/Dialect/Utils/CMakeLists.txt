# Exclude tests from libMLIR.so
add_bishengir_library(BishengIRTestDialectUtils
  BiShengSegmenter.cpp
  InstructionMarker.cpp
  TestDimensionAnalyzer.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRAnalysis
  MLIRAnnotationDialect
  MLIRFuncDialect
  MLIRHIVMAnalysis
  MLIRHIVMDialect
  MLIRTensorUtils
  MLIRTransforms
)

target_include_directories(BishengIRTestDialectUtils PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)