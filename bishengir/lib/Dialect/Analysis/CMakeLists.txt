add_subdirectory(Schedule)
add_subdirectory(VFFusion)

add_bishengir_dialect_library(BiShengIRAnalysis
  AgglomerationBlock.cpp
  DimensionAnalyzer.cpp
  DimensionMarker.cpp
  Initializer.cpp
  ProcessOperations.cpp

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRTensorUtils
)
