add_bishengir_dialect_library(MLIRHIVMAVEPipelines
  HIVMAVEPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMAVE

  LINK_LIBS PUBLIC
  BiShengIRTensorTransforms
  BishengIRMemRefTransforms
  BishengIRLLVMTransforms
  BishengIRSCFTransforms
  BishengIRArithTransforms
  BishengIRVectorTransforms
  BishengIRTransform
  MLIRArithDialect
  MLIRHIVMTransforms
  MLIRMathDialect
  MLIRMathExtDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRHFusionTransforms
  MLIRReconcileUnrealizedCasts
)