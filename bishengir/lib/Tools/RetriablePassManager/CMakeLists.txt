add_bishengir_library(BiShengIRRetriablePassManager
  DiagnosticUtils.cpp
  OverFlowPolicyBase.cpp
  RetriablePassManager.cpp
  TuningRetryPolicy.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Tools/RetriablePassManager

  DEPENDS
  BiShengIROptionsIncGen

  LINK_LIBS PUBLIC
  BiShengIRPass
)
