set(BISHENGIR_PRIVATE_LIBS
  BiShengIRHACCDialect
  BiShengIRMathExtDialect
)

add_bishengir_dialect_library(BiShengIRHFusionUtils
  IS_OPEN_SOURCE
  BufferUtils.cpp
  ExtraBuffer.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  BiShengIRScopeDialect
  MLIRArithDialect
  BiShengIRHFusionDialect
  MLIRMemRefDialect
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRPass
)
