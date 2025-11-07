add_bishengir_dialect_library(BiShengIRTorchPipelines
  TorchPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TorchToHFusion
  ${TORCH_MLIR_INCLUDE_DIRS}

  LINK_LIBS PUBLIC
  BiShengIRTransform
)