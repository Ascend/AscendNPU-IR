add_bishengir_unittest(BiShengIRDialectUtilsTests
  ReportFatalErrorDeathTest.cpp
  SelectRoundModeTest.cpp
  L12L0BuilderTest.cpp
  IntermediateValuesAnalyzerTest.cpp
  UnionFindTest.cpp
)

target_link_libraries(BiShengIRDialectUtilsTests PRIVATE
  MLIRIR
  MLIRArithDialect
  MLIRLLVMDialect
  BiShengIRDialectUtils
  BiShengIRHFusionAnalysis
  BiShengIRHFusionDialect
  BiShengIRHFusionTransforms
  BiShengIRHIVMDialect
  BiShengIRHIVMTransforms
  BiShengIRVFFusion
  MLIRAVEUtils
  MLIRHIVMRegbaseIntrinsUtils
)
