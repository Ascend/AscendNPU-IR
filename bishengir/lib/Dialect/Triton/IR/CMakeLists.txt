add_bishengir_dialect_library(BiShengIRTritonExtension
  TritonExtension.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Triton

  DEPENDS
  BiShengIRTritonOpsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRCallInterfaces
  MLIRFunctionInterfaces
)
