add_bishengir_dialect_library(MLIRHIVMAnalysis
  IS_OPEN_SOURCE
  DimensionAnalyzer.cpp
  DimensionMarker.cpp
  Initializer.cpp
  VFInplaceReuseAnalyzer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM

  DEPENDS
  MLIRHIVMConversionsIncGen
  MLIRHIVMTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRAnalysis
  BiShengIRDialectUtils
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAnnotationDialect
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRBufferizationTransforms
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRHACCDialect
  MLIRHACCUtils
  MLIRHIVMDialect
  MLIRHIVMUtils
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefExtDialect
)