add_bishengir_unittest(BiShengIRPassTests
  FilterPassesTest.cpp
)

target_link_libraries(BiShengIRPassTests PRIVATE
  BiShengIRDialectUtils
  MLIRAnnotationDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
)
