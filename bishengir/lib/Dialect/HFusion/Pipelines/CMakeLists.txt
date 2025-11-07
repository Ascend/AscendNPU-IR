add_bishengir_dialect_library(BiShengIRHFusionPipelines
  HFusionPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  LINK_LIBS PUBLIC
  BiShengIRTensorTransforms
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  BiShengIRHFusionTransforms
  MLIRPass
  BiShengIRHFusionToHIVM
  BiShengIRArithToHFusion
  BiShengIRMathToHFusion
  BiShengIRLinalgToHFusion
  BiShengIRGPUToHFusion
  BiShengIRTransform
)
