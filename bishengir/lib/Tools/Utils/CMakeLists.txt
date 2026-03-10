add_bishengir_library(BiShengIRToolUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/Utils

  LINK_LIBS PUBLIC
  BiShengIRPass
)