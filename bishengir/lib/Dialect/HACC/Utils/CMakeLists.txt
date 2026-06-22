add_bishengir_dialect_library(BiShengIRHACCUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/HACC

  LINK_LIBS PUBLIC
  BiShengIRHACCDialect
  BiShengIRHACCTransforms
  MLIRLLVMCommonConversion
)
