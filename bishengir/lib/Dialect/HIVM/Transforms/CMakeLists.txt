file(GLOB ALIGN_BUFFER ${CMAKE_CURRENT_SOURCE_DIR}/AlignBuffer/*.cpp)
file(GLOB BUBBLE_UP_EXTRACT_SLICE CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/BubbleUpExtractSlice/*.cpp)
file(GLOB GRAPH_SYNC_CONFLICT_SOLVER CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/GraphSyncSolver/*.cpp)
file(GLOB INJECT_SYNC ${CMAKE_CURRENT_SOURCE_DIR}/InjectSync/*.cpp)
file(GLOB INSERT_LOAD_STORE_FOR_MIX_CV ${CMAKE_CURRENT_SOURCE_DIR}/InsertLoadStoreForMixCV/*.cpp)
file(GLOB NDDMA CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/NDDMA/*.cpp)
file(GLOB TILING ${CMAKE_CURRENT_SOURCE_DIR}/Tiling/*.cpp)
file(GLOB NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/regbase/Normalize/*.cpp)
set(TILE_AND_BIND_SUB_BLOCK
    ${CMAKE_CURRENT_SOURCE_DIR}/TileAndBindSubBlock/Helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileAndBindSubBlock/TileUtils.cpp)
file(GLOB PARTITION_AND_BIND_SUB_BLOCK CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/PartitionAndBindSubBlock/*.cpp)

add_bishengir_dialect_library(BiShengIRHIVMTransforms
  ${ALIGN_BUFFER}
  ${BUBBLE_UP_EXTRACT_SLICE}
  ${GRAPH_SYNC_CONFLICT_SOLVER}
  ${INJECT_SYNC}
  ${INSERT_LOAD_STORE_FOR_MIX_CV}
  ${NDDMA}
  ${TILING}
  ${TILE_AND_BIND_SUB_BLOCK}
  ${PARTITION_AND_BIND_SUB_BLOCK}
  ${NORMALIZE}
  AddFFTSToSyncBlockSetOp.cpp
  AllocExtraBuffer.cpp
  AllocToAlloca.cpp
  AllocToPointerCast.cpp
  AnnotateVFAlias.cpp
  AutoBlockifyParallelLoop.cpp
  AutoScope.cpp
  AutoInferBufferSize.cpp
  BindSyncBlockLockArg.cpp
  BindWorkSpaceArg.cpp
  BufferizableOpInterfaceImpl.cpp
  CloneTensorEmpty.cpp
  ComposeCollapseExpand.cpp
  ComposeMemRefViews.cpp
  ConstantizeBufferSize.cpp
  ConvertNonContiguousReshapeToCopy.cpp
  ConvertToHIVMOp.cpp
  InsertAnchorsAndBackup.cpp
  ConvertLayout/ShapeConversion/ConvertLayoutToFractalsUtils.cpp
  ConvertLayout/ShapeConversion/ConvertLayoutFromFractalsUtils.cpp
  ConvertLayout/ShapeConversion/ConvertLayoutOffsetUtils.cpp
  ConvertLayout/CombineOptimizedConvertLayout.cpp
  ConvertLayout/ConvertLayoutToTranspose.cpp
  ConvertLayout/ConvertLayoutUtils.cpp
  ConvertLayout/HoistConvertLayout.cpp
  ConvertLayout/InsertConvertLayout.cpp
  ConvertLayout/Propagate/PropagateConvertLayout.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutElementwise.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutScfIf.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutScfFor.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutScfWhile.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutExtractSlice.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutInsertSlice.cpp
  ConvertLayout/Propagate/PropagateConvertLayoutVBrc.cpp
  CreatePreload.cpp
  CVPipelining.cpp
  ../Utils/ShapeRegistry.cpp
  EnableHIVMCCompatiblePrint.cpp
  EnableMultiBuffer.cpp
  FlattenModule.cpp
  FlattenOps.cpp
  HIVMAggregatedDecomposeOp.cpp
  HIVMBubbleUpExtractSlice.cpp
  HIVMDecomposeOp.cpp
  HIVMInlineOTFLoadStore.cpp
  HIVMLowerToLoops.cpp
  HIVMMapForallToBlocks.cpp
  HIVMOptFuncOutput.cpp
  HIVMOptSinglePoint.cpp
  HIVMVectorizeOps.cpp
  HoistTightlyCoupledAlloc.cpp
  InferFuncCoreType.cpp
  InferHIVMDataLayout.cpp
  InferHIVMMemScope.cpp
  InferVFMode.cpp
  InferSimtVFMemEffect.cpp
  InferSimtVFMemScopeHint.cpp
  InitEntryKernel.cpp
  InjectBlockSync.cpp
  InlineFixpipe.cpp
  InlineLoadCopy.cpp
  InlineOTFBroadcast.cpp
  InsertAllocBasePlaceholder.cpp
  InsertCVTightCoupledBuffer.cpp
  InsertMemSemanticForSimtVF.cpp
  InsertFreeLockVarBeforeReturn.cpp
  InsertInferSyncBlockLockNumAndInitFunc.cpp
  InsertInferTaskTypeFunc.cpp
  InsertInferVFModeFunc.cpp
  InsertInferWorkSpaceSizeFunc.cpp
  InsertInitAndFinishForDebug.cpp
  InsertLoadStoreForMixCV.cpp
  InsertLoadStoreForScalar.cpp
  InsertNZ2NDForDebug.cpp
  InsertL12UBForDebug.cpp
  InsertWorkSpaceForMixCV.cpp
  LegalizeBoolForSimtVF.cpp
  LegalizeForLLVMExport.cpp
  LiftLowestStride.cpp
  LiftZeroRank.cpp
  LowerCreateSyncBlockLock.cpp
  LowerMultiBufferCounter.cpp
  MarkDisableLoad.cpp
  MarkMultiBuffer.cpp
  MarkSimtScopeNoInline.cpp
  MarkSyncBlockLockWithSubblock.cpp
  HIVMCanonicalizationPipeline.cpp
  MarkRealCoreType.cpp
  MarkTightlyCoupledBuffer.cpp
  MaterializeSimtVFMemScope.cpp
  MemoryDisplay.cpp
  MergeSamePreloadScopes.cpp
  NormalizeBitwiseSelect.cpp
  NormalizeConvOps.cpp
  NormalizeToTensorOp.cpp
  TensorCopyInsertion.cpp
  NormalizeMatmul.cpp
  OptMemPlanForPipeline.cpp
  OutlineAllocInVF.cpp
  OutlineCopyInVF.cpp
  PlanMemory.cpp
  regbase/PlanMemory.cpp
  RecognizeDeinterleaveOp.cpp
  RecognizeDiscontiguousStore.cpp
  ReduceRankSubview.cpp
  RemoveHIVMDataLayoutAnnotation.cpp
  RemoveCopyOps.cpp
  SetBufferSize.cpp
  SinkExclusivePreloadWork.cpp
  SinkOpToConsumerInLoop.cpp
  SIMTVFSubTiling.cpp
  SplitSimtModule.cpp
  SplitMixedIfConditionals.cpp
  SplitMixKernel.cpp
  SyncBlockHoisting.cpp
  TileAndBindSubBlock.cpp
  TileBatchMMIntoLoop.cpp
  TileCubeVectorLoop.cpp
  TightlyCoupledBufferUtils.cpp
  TritonGlobalKernelArgsToHIVMOp.cpp
  UnitFlagInfoBase.cpp
  VFOperandSubstitution.cpp
  AnalyzeArithVectorMask.cpp
  WriteBackShared.cpp
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
  BiShengIRMathExtDialect
  BiShengIRScopeDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRVectorDialect
  BiShengIRMemRefExtDialect
  BiShengIRDialectUtils
  BiShengIRTransform
  )

# Loop-shape fingerprints consumed by CVPipelining: generated from the coverage
# registry at configure time so the compiler only carries mixed keys.
set(SHAPE_REGISTRY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build-tools/coverage/shape_registry.py")
if(NOT EXISTS "${SHAPE_REGISTRY}" AND DEFINED LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR)
  set(SHAPE_REGISTRY
      "${LLVM_EXTERNAL_BISHENGIR_SOURCE_DIR}/../build-tools/coverage/shape_registry.py")
endif()
find_package(Python3 QUIET COMPONENTS Interpreter)
set(SKEW_SHAPE_TABLE ${CMAKE_CURRENT_BINARY_DIR}/SkewShapeTable.inc)
if(Python3_Interpreter_FOUND)
  add_custom_command(
    OUTPUT ${SKEW_SHAPE_TABLE}
    COMMAND ${Python3_EXECUTABLE} ${SHAPE_REGISTRY} --emit-cxx ${SKEW_SHAPE_TABLE}
    DEPENDS ${SHAPE_REGISTRY}
    COMMENT "Generating loop-shape fingerprints")
  add_custom_target(BiShengIRSkewShapeTable DEPENDS ${SKEW_SHAPE_TABLE})
  add_dependencies(BiShengIRHIVMTransforms BiShengIRSkewShapeTable)
else()
  file(WRITE ${SKEW_SHAPE_TABLE}
       "namespace {\nconstexpr unsigned kSpanKeyCount = 0;\nconstexpr uint64_t kSpanKeys[kSpanKeyCount] = {0ULL};\n}\n")
endif()
target_include_directories(BiShengIRHIVMTransforms PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
