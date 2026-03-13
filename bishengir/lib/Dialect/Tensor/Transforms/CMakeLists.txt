add_bishengir_library(BiShengIRTensorTransforms
  BubblePadUp.cpp
  BubbleUpExtractSlice.cpp
  CanonicalizeTensorReshape.cpp
  DecomposeTensorConcat.cpp
  FoldTensorEmpty.cpp
  MergeConsecutiveInsertExtractSlice.cpp
  NarrowTensorOp.cpp
  NormalizeLastDimUnalignedTensorOp.cpp
  NormalizeTensorOps.cpp
  OptimizeDpsOpWithYieldedInsertSlice.cpp
  PropagateReshape.cpp
  PropagateReshape/PropagateCollapseDown.cpp
  PropagateReshape/PropagateExpandUp.cpp
  PropagateReshape/PropagateNearEndExpandDown.cpp
  PropagateReshape/PropagateOp/PropagatableOp.cpp
  PropagateReshape/PropagateOp/ScfForOp.cpp
  PropagateReshape/SwapCollapseExpand.cpp
  PropagateReshape/Utils.cpp
  PropagateReshape/Utils/ConcatUtils.cpp
  ReplicateOutEmptyTensor.cpp
  TilingInterfaceImpl.cpp
  TrickleConcatDown.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Tensor/Transforms

  DEPENDS
  MLIRTensorTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRMemRefTransforms
  BiShengIRTensorUtils
  MLIRControlFlowDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRDialect
  MLIRPass
  BiShengIRAnnotationDialect
  BiShengIRHFusionDialect
  BiShengIRHFusionUtils
  BiShengIRHIVMDialect
  MLIRLinalgTransforms
)
