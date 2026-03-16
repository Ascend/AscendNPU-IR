set(BISHENGIR_PRIVATE_LIBS
  BiShengIRTensorTransforms
)

add_bishengir_dialect_library(MLIRHFusionPipelines
  IS_OPEN_SOURCE

  HFusionPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  BishengIRTransform
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRHFusionTransforms
  MLIRMemRefTransforms
  MLIRPass
  MLIRHFusionToHIVM
  BiShengIRArithToHFusion
  BiShengIRMathToHFusion
  BiShengIRLinalgToHFusion
  BiShengIRGPUToHFusion
)
