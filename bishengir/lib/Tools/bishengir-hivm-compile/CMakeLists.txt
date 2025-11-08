get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_bishengir_library(BiShengIRHIVMCompileLib
  BiShengIRHIVMCompileConfig.cpp
  BiShengIRHIVMCompileMain.cpp
  PassPipeline.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Tools/bishengir-hivm-compile

  LINK_LIBS PUBLIC
  ${translation_libs}
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  BiShengIRPass
  BiShengIRTransform
  BiShengIRCompileLib
  BiShengIRToolUtils
)