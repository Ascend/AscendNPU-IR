add_bishengir_dialect_library(BiShengIRTorchTransforms
  LiteralDataTypeCast.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TorchToHFusion
  ${TORCH_MLIR_INCLUDE_DIRS}

)