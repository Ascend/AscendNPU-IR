add_bishengir_library(BiShengIRTensorDialect
  TensorImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Tensor/IR

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIRIR
  MLIRDialect
)
