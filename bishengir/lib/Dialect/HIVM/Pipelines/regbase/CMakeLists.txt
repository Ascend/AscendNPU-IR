add_bishengir_dialect_library(BiShengIRHIVMRegbasePipelines
  KernelKindPasses.cpp
  HIVMRegbasePipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Pipelines/regbase

  DEPENDS
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRArithToAffine
  BiShengIRArithTransforms
  BiShengIRBufferizationTransforms
  BiShengIRHFusionTransforms
  BiShengIRHIVMTransforms
  BiShengIRHIVMToStandard
  BiShengIRMemRefExtLowering
  BiShengIRMemRefTransforms
  BiShengIRPass
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
