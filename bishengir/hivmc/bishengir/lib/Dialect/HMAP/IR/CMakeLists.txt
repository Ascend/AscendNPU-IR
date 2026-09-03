add_bishengir_dialect_library(MLIRHMAPDialect
  HMAPMeshOps.cpp
  HMAPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HMAP

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
)
