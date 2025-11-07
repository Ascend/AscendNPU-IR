file(GLOB AUTO_SCHEDULE ${CMAKE_CURRENT_SOURCE_DIR}/AutoSchedule/*.cpp)
file(GLOB FLATTENER ${CMAKE_CURRENT_SOURCE_DIR}/Flattener/*.cpp)
file(GLOB OP_FUSION ${CMAKE_CURRENT_SOURCE_DIR}/OpFusion/*.cpp)

add_bishengir_dialect_library(BiShengIRHFusionTransforms
  ${AUTO_SCHEDULE}
  ${FLATTENER}
  ${OP_FUSION}
  AdaptTritonKernel.cpp
  AddFFTSAddr.cpp
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
  FoldSymbolicDim.cpp
  HoistTensorEmpty.cpp
  InlineBrc.cpp
  InferFuncFusionKind.cpp
  InferOutShapes.cpp
  Normalize.cpp
  NormalizeSliceOps.cpp
  OpFusion.cpp
  OutlineSingleOp.cpp
  PackTilingData.cpp
  ReCacheIO.cpp
  ReorderOpsByBFS.cpp
  SimplifyOps.cpp
  TensorResToOutParams.cpp
  TilingInterfaceImpl.cpp
  UnfoldSymbolicDim.cpp
  LegalizeBF16.cpp
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
  BiShengIRHFusionAnalysis
  BiShengIRMathExtDialect
  BiShengIRAnnotationDialect
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
