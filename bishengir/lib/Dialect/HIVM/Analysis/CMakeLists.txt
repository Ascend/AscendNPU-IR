add_bishengir_dialect_library(BiShengIRHIVMAnalysis
  IS_OPEN_SOURCE
  DimensionAnalyzer.cpp
  DimensionMarker.cpp
  Initializer.cpp
  VFInplaceReuseAnalyzer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  DEPENDS
  BiShengIRHIVMConversionsIncGen
  BiShengIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnalysis
  BiShengIRDialectUtils
  MLIRAffineDialect
  MLIRAffineTransforms
  BiShengIRAnnotationDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRBufferizationTransforms
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  BiShengIRHACCDialect
  BiShengIRHACCUtils
  BiShengIRHIVMDialect
  BiShengIRHIVMUtils
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  BiShengIRMemRefExtDialect
)
