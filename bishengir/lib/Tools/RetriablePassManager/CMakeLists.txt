add_bishengir_library(BiShengIRRetriablePassManager
  TuningRetryPolicy.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/RetriablePassManager

  LINK_COMPONENTS
  Support
)
