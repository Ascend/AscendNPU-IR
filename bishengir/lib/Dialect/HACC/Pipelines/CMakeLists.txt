add_bishengir_dialect_library(MLIRHACCPipelines
  HACCPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HACC

  LINK_LIBS PUBLIC
  BishengIRLLVMTransforms
  MLIRHACCDialect
  MLIRHACCTransforms
  MLIRPass
  MLIRTransforms
)