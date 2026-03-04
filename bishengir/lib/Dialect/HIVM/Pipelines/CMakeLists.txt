add_bishengir_dialect_library(MLIRHIVMPipelines
  HIVMPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  LINK_LIBS PUBLIC
  BiShengIRTensorTransforms
  BishengIRMemRefTransforms
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