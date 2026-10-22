add_bishengir_dialect_library(BiShengIRHIVMV2Pipelines
  HIVMV2Pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Pipelines/v2

  LINK_LIBS PUBLIC
  BiShengIRBufferizationTransforms
  BiShengIRHIVMPipelines
  BiShengIRHIVMTransforms
  BiShengIRMemRefExtLowering
  BiShengIRMemRefTransforms
  BiShengIRSCFTransforms
  BiShengIRTransform
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
