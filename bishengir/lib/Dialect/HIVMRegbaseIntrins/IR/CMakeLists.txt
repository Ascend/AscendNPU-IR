add_bishengir_dialect_library(MLIRHIVMRegbaseIntrinsDialect
  HIVMRegbaseIntrins.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMRegbaseIntrins

  DEPENDS
  MLIRHIVMRegbaseIntrinOpsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
