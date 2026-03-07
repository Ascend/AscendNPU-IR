file(GLOB AUTO_SCHEDULE ${CMAKE_CURRENT_SOURCE_DIR}/AutoSchedule/*.cpp)
file(GLOB FLATTENER ${CMAKE_CURRENT_SOURCE_DIR}/Flattener/*.cpp)
file(GLOB OP_FUSION ${CMAKE_CURRENT_SOURCE_DIR}/OpFusion/*.cpp)

set(BISHENGIR_PRIVATE_LIBS
  BiShengIRDialectUtils
  BiShengIRSCFTransformOps
  BiShengIRScopeDialect
  BiShengIRTensorTransforms
  BishengIRTransform
  MLIRHIVMTransformOps
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRHFusionAnalysis
  MLIRMathExtDialect
  MLIRAnnotationDialect
)

add_bishengir_dialect_library(MLIRHFusionTransforms
  IS_OPEN_SOURCE

  ${AUTO_SCHEDULE}
  ${FLATTENER}
  ${OP_FUSION}
  AdaptTritonKernel.cpp
  AddFFTSAddr.cpp
  AutoVectorize.cpp
  AutoVectorizeV2.cpp
  AutoVectorizePatterns.cpp
  BufferizableOpInterfaceImpl.cpp
  CacheIO.cpp
  CacheIOForReturnArg.cpp
  ComposeMultiReduce.cpp
  ConstantizeTilingData.cpp
  ConvertGenericToNamedOp.cpp
  HFusionGeneralize.cpp
  Decompose.cpp
  DecomposeOpInterfaceImpl.cpp
  DecomposeMulti.cpp
  DropSymbols.cpp
  EliminateDuplicateFuncs.cpp
  FlattenOps.cpp
  FoldUnitDims.cpp
  FoldSymbolicDim.cpp
  GenericUnroller.cpp
  HoistTensorEmpty.cpp
  InlineBrc.cpp
  InferFuncFusionKind.cpp
  InferOutShapes.cpp
  MeshShardingInterfaceImpl.cpp
  Normalize.cpp
  NormalizeSliceOps.cpp
  OpFusion.cpp
  OutlineSingleOp.cpp
  PackTilingData.cpp
  PreVectorizationFusion.cpp
  PullSliceIntoVectorFunction.cpp
  ReCacheIO.cpp
  RemoveCacheIO.cpp
  ReorderOpsByBFS.cpp
  SimplifyOps.cpp
  TensorResToOutParams.cpp
  TilingInterfaceImpl.cpp
  TreeReduce.cpp
  UnfoldSymbolicDim.cpp
  LegalizeScalar.cpp
  LegalizeBF16.cpp
  LegalizeBool.cpp
  LegalizeFP8.cpp
  DowngradeFP64CstOp.cpp
  VectorizeOps.cpp
  WrapHostFunc.cpp
  RemoveMaskFromUnalignedReductionLoop.cpp
  SimplifyVFArgs.cpp
  MergeVecScope.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HFusion

  DEPENDS
  MLIRHFusionTransformsIncGen

  LINK_LIBS PUBLIC
  ${BISHENGIR_PRIVATE_LIBS}
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRDialectUtils
  MLIRHFusionDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRMemRefTransformOps
  MLIRSCFTransformOps
  MLIRShapeDialect
  MLIRShardingInterface
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTensorUtils
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRPass
  MLIRHFusionUtils
)
