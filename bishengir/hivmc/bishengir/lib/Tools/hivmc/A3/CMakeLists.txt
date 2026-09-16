get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_bishengir_library(BiShengIRHIVMCLibA3
  HIVMCMainA3.cpp
  PassPipelineA3.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/hivmc


)
