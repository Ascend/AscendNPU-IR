add_bishengir_unittest(BiShengIRAgglomerationBlockTests
  PARTIAL_SOURCES_INTENDED
  AgglomerationBlockTest.cpp
)

target_link_libraries(BiShengIRAgglomerationBlockTests PRIVATE
  BiShengIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRAgglomerationBlockTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRAgglomerationBlockTests)
endif()
