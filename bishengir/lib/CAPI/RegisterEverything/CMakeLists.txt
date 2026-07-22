# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(triton_dialect_registration_deps GLOBAL PROPERTY TRITON_DIALECT_REGISTRATION_DEPS)

add_mlir_public_c_api_library(BiShengIRCAPIRegisterEverything
  RegisterEverything.cpp

  DEPENDS
  ${triton_dialect_registration_deps}

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}

  BiShengIRCompileLib
  BiShengIRRegBaseCompileLib
  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRCAPITransforms
)
