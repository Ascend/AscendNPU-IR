# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  BiShengIRRunnerUtils.cpp
  CreateHostMain.cpp
  ExecutionEnginePipelines.cpp
)

if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_bishengir_extension_library(BiShengIRExecutionEngineTransforms
  CreateHostMain.cpp
  ExecutionEnginePipelines.cpp

  DEPENDS
  BiShengIRExecutionEngineTransformsIncGen

  LINK_LIBS PUBLIC
  ${conversion_libs}
  ${dialect_libs}
  BiShengIRHACCUtils
  MLIRPass
  BiShengIRTransform
)

# When -fPIC is not provided, shared libraries cannot be built if it links against
# non-PIC code.
if(LLVM_ENABLE_PIC)
  add_mlir_library(bishengir_runner_utils
    SHARED
    BiShengIRRunnerUtils.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PUBLIC
    MLIRSupport
    mlir_float16_utils
  )
  set_property(TARGET bishengir_runner_utils PROPERTY CXX_STANDARD 17)
  target_compile_definitions(bishengir_runner_utils PRIVATE bishengir_runner_utils_EXPORTS)
endif()
