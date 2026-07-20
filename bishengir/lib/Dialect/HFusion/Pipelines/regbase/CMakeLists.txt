add_bishengir_dialect_library(BiShengIRHFusionRegbasePipelines
  HFusionRegbasePipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion/Pipelines/regbase

  LINK_LIBS PUBLIC
  BiShengIRHFusionTransforms
  BiShengIRHFusionToHIVM
  BiShengIRTensorTransforms
  BiShengIRVectorTransforms
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRPass
  BiShengIRTransform
)
