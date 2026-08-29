add_subdirectory(Transforms)
add_subdirectory(Schedule)
add_subdirectory(VFFusion)

add_bishengir_dialect_library(BiShengIRAnalysis
  DimensionAnalyzer.cpp
  DimensionMarker.cpp
  Initializer.cpp
  ProcessOperations.cpp

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRTensorUtils
)
