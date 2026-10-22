add_bishengir_dialect_library(BiShengIRHFusionUtils
  BufferUtils.cpp
  ExtraBuffer.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  LINK_LIBS PUBLIC
  BiShengIRHACCDialect
  BiShengIRMathExtDialect
  MLIRArithDialect
  BiShengIRHFusionDialect
  MLIRMemRefDialect
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRPass
)
