add_bishengir_dialect_library(BiShengIRHIVMPipelines
  ConvertToHIVMPipeline.cpp
  HIVMBufferizationPipeline.cpp
  HIVMPipelineBase.cpp
  HIVMPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  LINK_LIBS PUBLIC
  BiShengIRBufferizationTransforms
  BiShengIRHFusionToHIVM
  BiShengIRHFusionTransforms
  BiShengIRHIVMTransforms
  BiShengIRMathExtDialect
  BiShengIRMemRefTransforms
  BiShengIRSCFTransforms
  BiShengIRTensorToHIVM
  BiShengIRTensorTransforms
  BiShengIRTransform
  MLIRArithDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRPass
  MLIRTransforms
)

add_subdirectory(regbase)
add_subdirectory(v2)
