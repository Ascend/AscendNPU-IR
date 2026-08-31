add_bishengir_dialect_library(MLIRAVECostModelInfo
  AVECostModel.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMAVE

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRVFFusion
  MLIRArithDialect
  MLIRAVEDialect
  BiShengIRHACCUtils
  MLIRMathDialect
  MLIRSCFDialect
)
