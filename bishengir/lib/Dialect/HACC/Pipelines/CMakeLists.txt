add_bishengir_dialect_library(BiShengIRHACCPipelines
  HACCPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC

  LINK_LIBS PUBLIC
  BiShengIRLLVMTransforms
  BiShengIRHACCDialect
  BiShengIRHACCTransforms
  MLIRPass
  MLIRTransforms
)
