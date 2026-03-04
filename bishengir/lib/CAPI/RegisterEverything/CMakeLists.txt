# Dialect registration.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_public_c_api_library(BiShengIRCAPIRegisterEverything
  RegisterEverything.cpp

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRBiShengIRCompileLib
  MLIRBiShengIRHIVMCompileLib
  MLIRBuiltinToLLVMIRTranslation
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRCAPITransforms
)
