add_bishengir_unittest(BiShengIRDialectUtilsTests
  ReportFatalErrorDeathTest.cpp
  SelectRoundModeTest.cpp
  UnionFindTest.cpp
)

target_link_libraries(BiShengIRDialectUtilsTests PRIVATE
  MLIRIR
  MLIRArithDialect
  MLIRLLVMDialect
  BiShengIRDialectUtils
  BiShengIRHFusionDialect
  BiShengIRHFusionTransforms
  BiShengIRHIVMDialect
  BiShengIRHIVMTransforms
  BiShengIRVFFusion
  MLIRAVEUtils
  MLIRHIVMRegbaseIntrinsUtils
)
