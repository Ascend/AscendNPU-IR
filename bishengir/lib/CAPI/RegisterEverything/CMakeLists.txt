# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(triton_libs)
set(triton_deps)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND triton_libs TritonAggregateLib)
  list(APPEND triton_deps ProtonToProtonGPUIncGen)
endif()

add_mlir_public_c_api_library(BiShengIRCAPIRegisterEverything
  RegisterEverything.cpp

  DEPENDS
  ${triton_deps}

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}
  ${triton_libs}

  MLIRBiShengIRCompileLib
  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRCAPITransforms
)
