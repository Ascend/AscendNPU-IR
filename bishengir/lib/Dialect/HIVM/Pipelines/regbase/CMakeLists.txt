add_bishengir_dialect_library(BiShengIRHIVMRegbasePipelines
  HIVMRegbasePipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Pipelines/regbase

  LINK_LIBS PUBLIC
  BiShengIRArithToAffine
  BiShengIRArithTransforms
  BiShengIRHFusionTransforms
  BiShengIRHIVMTransforms
  BiShengIRHIVMToStandard
  BiShengIRMemRefExtLowering
  BiShengIRMemRefTransforms
  BiShengIRSCFTransforms
  BiShengIRScopeTransforms
  BiShengIRTensorToHIVM
  BiShengIRTensorTransforms
  BiShengIRTransform
  BiShengIRVectorTransforms
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRSCFTransforms
  MLIRTransforms
)
