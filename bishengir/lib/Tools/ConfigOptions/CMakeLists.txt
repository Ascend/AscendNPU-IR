llvm_add_library(BiShengIRConfigOptions STATIC
  Options.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/ConfigOptions

  LINK_LIBS PUBLIC
  MLIRTableGen
)