file(GLOB ALIGN_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/AlignBuffer/*.cpp)
file(GLOB BUBBLE_UP_EXTRACT_SLICE ${CMAKE_CURRENT_SOURCE_DIR}/BubbleUpExtractSlice/*.cpp)
file(GLOB INJECT_SYNC ${CMAKE_CURRENT_SOURCE_DIR}/InjectSync/*.cpp)
file(GLOB TILING ${CMAKE_CURRENT_SOURCE_DIR}/Tiling/*.cpp)
file(GLOB TILE_AND_BIND_SUB_BLOCK ${CMAKE_CURRENT_SOURCE_DIR}/TileAndBindSubBlock/*.cpp)

add_bishengir_dialect_library(BiShengIRHIVMTransforms
  ${ALIGN_BUFFER}
  ${BUBBLE_UP_EXTRACT_SLICE}
  ${INJECT_SYNC}
  ${TILING}
  ${TILE_AND_BIND_SUB_BLOCK}
  AddFFTSToSyncBlockSetOp.cpp
  AllocExtraBuffer.cpp
  AllocToAlloca.cpp
  AllocToPointerCast.cpp
  AutoBlockifyParallelLoop.cpp
  AutoInferBufferSize.cpp
  BindSyncBlockLockArg.cpp
  BindWorkSpaceArg.cpp
  BufferizableOpInterfaceImpl.cpp
  CloneTensorEmpty.cpp
  ComposeCollapseExpand.cpp
  ConstantizeBufferSize.cpp
  ConvertNonContiguousReshapeToCopy.cpp
  ConvertToHIVMOp.cpp
  CVPipelining.cpp
  EnableMultiBuffer.cpp
  FlattenOps.cpp
  HIVMAggregatedDecomposeOp.cpp
  HIVMBubbleUpExtractSlice.cpp
  HIVMDecomposeOp.cpp
  HIVMInlineOTFLoadStore.cpp
  HIVMLowerToLoops.cpp
  HIVMMapForallToBlocks.cpp
  HIVMOptFuncOutput.cpp
  HIVMOptSinglePoint.cpp
  InferFuncCoreType.cpp
  InferHIVMDataLayout.cpp
  InferHIVMMemScope.cpp
  InitEntryKernel.cpp
  InjectBlockSync.cpp
  InlineFixpipe.cpp
  InlineOTFBroadcast.cpp
  InsertNZ2NDForDebug.cpp
  InsertInferSyncBlockLockNumAndInitFunc.cpp
  InsertInferWorkSpaceSizeFunc.cpp
  InsertInitAndFinishForDebug.cpp
  InsertLoadStoreForMixCV.cpp
  InsertWorkSpaceForMixCV.cpp
  LiftLowestStride.cpp
  LiftZeroRank.cpp
  LowerCreateSyncBlockLock.cpp
  MarkMultiBuffer.cpp
  NormalizeLoopIterator.cpp
  NormalizeMatmul.cpp
  OptMemPlanForPipeline.cpp
  PlanMemory.cpp
  RecognizeDeinterleaveOp.cpp
  ReduceRankSubview.cpp
  SetBufferSize.cpp
  SplitMixKernel.cpp
  TileAndBindSubBlock.cpp
  TileBatchMMIntoLoop.cpp
  TileCubeVectorLoop.cpp
  TritonGlobalKernelArgsToHIVMOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/Utils
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Symbol

  DEPENDS
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithUtils
  BiShengIRAnnotationDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationTransformOps
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  BiShengIRHACCUtils
  BiShengIRHACCDialect
  BiShengIRHIVMAnalysis
  BiShengIRHIVMDialect
  BiShengIRHIVMUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
  BiShengIRDialectUtils
  BiShengIRTransform
  )
