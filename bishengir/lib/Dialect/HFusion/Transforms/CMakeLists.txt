file(GLOB AUTO_SCHEDULE ${CMAKE_CURRENT_SOURCE_DIR}/AutoSchedule/*.cpp)
file(GLOB FLATTENER ${CMAKE_CURRENT_SOURCE_DIR}/Flattener/*.cpp)
file(GLOB OP_FUSION ${CMAKE_CURRENT_SOURCE_DIR}/OpFusion/*.cpp)
file(GLOB NORMALIZE ${CMAKE_CURRENT_SOURCE_DIR}/regbase/Normalize/*.cpp)
file(GLOB NORMALIZE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/regbase/Normalize/Utils/*.cpp)

add_bishengir_dialect_library(BiShengIRHFusionTransforms
  ${AUTO_SCHEDULE}
  ${FLATTENER}
  ${OP_FUSION}
  ${NORMALIZE}
  ${NORMALIZE_UTILS}
  AdaptTritonKernel.cpp
  AddFFTSAddr.cpp
  AutoVectorize/AutoVectorizeInterpreter.cpp
  AutoVectorize/Verify.cpp
  AutoVectorizeVerifier.cpp
  BufferizableOpInterfaceImpl.cpp
  CacheIO.cpp
  CacheIOForReturnArg.cpp
  ComposeMultiReduce.cpp
  ConstantizeTilingData.cpp
  ConvertGenericToNamedOp.cpp
  Decompose.cpp
  DecomposeOpInterfaceImpl.cpp
  DecomposeMulti.cpp
  DropSymbols.cpp
  EliminateDuplicateFuncs.cpp
  FlattenOps.cpp
  FoldExtractInsertPair.cpp
  FoldExtractInsertPair.cpp
  FoldUnitDims.cpp
  FoldSymbolicDim.cpp
  GenericUnroller.cpp
  HFusionGeneralize.cpp
  HoistTensorEmpty.cpp
  InlineBrc.cpp
  InferFuncFusionKind.cpp
  InferOutShapes.cpp
  Normalize.cpp
  NormalizeSliceOps.cpp
  OpFusion.cpp
  OutlineSingleOp.cpp
  PackTilingData.cpp
  PrepareI1Nx1ForVectorization.cpp
  PullSliceIntoVectorFunction.cpp
  ReCacheIO.cpp
  RemoveCacheIO.cpp
  RemoveMaskFromUnalignedReductionLoop.cpp
  RemoveRedundantWriteAndReadPair.cpp
  ReorderOpsByBFS.cpp
  SimplifyVFArgs.cpp
  SimplifyOps.cpp
  TensorResToOutParams.cpp
  TilingInterfaceImpl.cpp
  UnfoldSymbolicDim.cpp
  UpliftWhileToFor.cpp
  LegalizeScalar.cpp
  VectorizeOps.cpp
  LegalizeBF16.cpp
  LegalizeFP8.cpp
  LegalizeBool.cpp
  DowngradeFP64CstOp.cpp
  WrapHostFunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  BiShengIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRTensorTransforms
  BiShengIRTransform
  BiShengIRHACCDialect
  BiShengIRHACCUtils
  BiShengIRHIVMDialect
  BiShengIRHFusionAnalysis
  BiShengIRMathExtDialect
  BiShengIRAnnotationDialect
  BiShengIRScopeUtils
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRDialectUtils
  BiShengIRHFusionDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRMemRefTransformOps
  MLIRSCFTransformOps
  MLIRVectorDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTensorUtils
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRPass
  BiShengIRHFusionUtils
  BiShengIRTransform
)
