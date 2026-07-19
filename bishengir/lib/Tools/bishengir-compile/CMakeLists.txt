get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_bishengir_library(BiShengIRCompileConfigLib
  BiShengIRCompileConfig.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/bishengir-compile

  LINK_LIBS PUBLIC
  BiShengIRHACCUtils
  BiShengIRPass
  BiShengIRToolUtils
)

add_bishengir_library(BiShengIRCompileLib
  BiShengIRCompileMain.cpp
  PassPipeline.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Tools/bisheng-compile

  LINK_LIBS PUBLIC
  ${translation_libs}
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  BiShengIRCompileConfigLib
  BiShengIRPass
  BiShengIRTransform
  BiShengIRToolUtils
)
