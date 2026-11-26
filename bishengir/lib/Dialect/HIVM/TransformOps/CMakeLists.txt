add_bishengir_dialect_library(BiShengIRHIVMTransformOps
  PARTIAL_SOURCES_INTENDED
  HIVMTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/TransformOps

  DEPENDS
  BiShengIRHIVMTransformOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRVectorDialect
  )

add_bishengir_dialect_library(BiShengIRHIVMVectorizeTransformOps
  PARTIAL_SOURCES_INTENDED
  Agglomerate.cpp
  AnnotateVF.cpp
  BackendCompatibility.cpp
  Groups.cpp
  VectorizeAttrs.cpp
  VectorizeTransformOps.cpp
  LiftToLinalg.cpp
  Outline.cpp
  Partition.cpp
  ExtractGroups.cpp
  Transaction.cpp
  VectorizeLinalg.cpp
  VectorizeResult.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/TransformOps

  DEPENDS
  BiShengIRHIVMVectorizeTransformOpsIncGen
  BiShengIRHIVMVectorizeAttrsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  BiShengIRAnalysis
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  MLIRAVEDialect
  BiShengIRMathExtDialect
  BiShengIRVFFusion
  MLIRArithDialect
  MLIRBytecodeWriter
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSCFTransformOps
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransformDialectUtils
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorTransforms
  )
