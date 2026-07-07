set(BISHENGIR_PRIVATE_LIBS
  BiShengIRTensorTransforms
)

add_bishengir_dialect_library(BiShengIRHFusionPipelines
  IS_OPEN_SOURCE
  HFusionPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  BiShengIRTransform
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  BiShengIRHFusionTransforms
  MLIRMemRefTransforms
  MLIRPass
  BiShengIRHFusionToHIVM
  BiShengIRArithToHFusion
  BiShengIRMathToHFusion
  BiShengIRLinalgToHFusion
  BiShengIRGPUToHFusion
)
