add_bishengir_unittest(BiShengIRDialectUtilsTests
  SelectRoundModeTest.cpp
)

target_link_libraries(BiShengIRDialectUtilsTests PRIVATE
  MLIRIR
  MLIRHFusionDialect
  MLIRHIVMDialect
)
