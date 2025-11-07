# Exclude tests from libMLIR.so
add_bishengir_library(BiShengIRTestDialectUtils
  BiShengSegmenter.cpp
  InstructionMarker.cpp
  TestDimensionAnalyzer.cpp

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  MLIRAnalysis
  BiShengIRAnnotationDialect
  MLIRFuncDialect
  BiShengIRHIVMDialect
  MLIRTensorUtils
  BiShengIRTransform
)

target_include_directories(BiShengIRTestDialectUtils PRIVATE ${BISHENGIR_SRC_DIR}/test/lib/Dialect)