add_bishengir_unittest(BiShengIRHIVMTransactionTests
  PARTIAL_SOURCES_INTENDED
  TransactionTest.cpp
)

target_link_libraries(BiShengIRHIVMTransactionTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMTransformOps
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRParser
  MLIRSCFDialect
  MLIRTransformDialect
  MLIRTransformDialectTransforms
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMTransactionTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMTransactionTests)
endif()

add_bishengir_unittest(BiShengIRHIVMAgglomerationTests
  PARTIAL_SOURCES_INTENDED
  AgglomerationTest.cpp
)

target_link_libraries(BiShengIRHIVMAgglomerationTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRAsmParser
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRTransformDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMAgglomerationTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMAgglomerationTests)
endif()

add_bishengir_unittest(BiShengIRHIVMOutlineTests
  PARTIAL_SOURCES_INTENDED
  OutlineTest.cpp
)

target_link_libraries(BiShengIRHIVMOutlineTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRAsmParser
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRTransformDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMOutlineTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMOutlineTests)
endif()

add_bishengir_unittest(BiShengIRHIVMLiftToLinalgTests
  PARTIAL_SOURCES_INTENDED
  LiftToLinalgTest.cpp
)

target_link_libraries(BiShengIRHIVMLiftToLinalgTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRParser
  MLIRTensorDialect
  MLIRTransformDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMLiftToLinalgTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMLiftToLinalgTests)
endif()

add_bishengir_unittest(BiShengIRHIVMVectorizeTests
  PARTIAL_SOURCES_INTENDED
  VectorizeLinalgTest.cpp
)

target_link_libraries(BiShengIRHIVMVectorizeTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRVectorDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMVectorizeTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMVectorizeTests)
endif()

add_bishengir_unittest(BiShengIRHIVMPartitionTests
  PARTIAL_SOURCES_INTENDED
  PartitionTest.cpp
)

target_link_libraries(BiShengIRHIVMPartitionTests PRIVATE
  BiShengIRHIVMDialect
  BiShengIRHIVMTransformOps
  BiShengIRHIVMVectorizeTransformOps
  MLIRArithDialect
  MLIRAsmParser
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMPartitionTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMPartitionTests)
endif()

add_bishengir_unittest(BiShengIRHIVMCustomOpMemoryEffectsTests
  PARTIAL_SOURCES_INTENDED
  CustomOpMemoryEffectsTest.cpp
)

target_link_libraries(BiShengIRHIVMCustomOpMemoryEffectsTests PRIVATE
  BiShengIRHIVMDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
)

# The default bishengir link flags use lld-only options; keep this target
# linkable when the test binary is built directly from CMake.
target_link_options(BiShengIRHIVMCustomOpMemoryEffectsTests PRIVATE -fuse-ld=lld)

if(TARGET check-bishengir)
  add_dependencies(check-bishengir run-BiShengIRHIVMCustomOpMemoryEffectsTests)
endif()
