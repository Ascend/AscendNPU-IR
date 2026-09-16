set(BISHENGIR_PRIVATE_LIBS
  MLIRHACCDialect
  MLIRMathExtDialect
  )

add_bishengir_dialect_library(MLIRHFusionUtils
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
  MLIRHFusionDialect
  MLIRMemRefDialect
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRPass
)
