get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(triton_libs)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND triton_libs TritonAggregateLib)
endif()

add_bishengir_library(MLIRBiShengIRCompileLib
  BiShengIRCompileConfig.cpp
  BiShengIRCompileMain.cpp
  PassPipeline.cpp
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Tools/bisheng-compile

  LINK_LIBS PUBLIC
  ${translation_libs}
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  BiShengIRPass
  BishengIRTransform
  ${triton_libs}

  LINK_COMPONENTS
  Object
  Support
)