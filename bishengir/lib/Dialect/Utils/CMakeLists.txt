add_bishengir_library(BiShengIRDialectUtils
  OpInterfaceUtils.cpp
  ReachabilityAnalyzer.cpp
  UnionFind.cpp
  Util.cpp
  ValueDependencyAnalyzer.cpp

  DEPENDS
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRHIVMEnumsIncGen
  MLIRHIVMAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRAnnotationDialect
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRHFusionDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  BiShengIRMemRefDialect
  BiShengIRTensorDialect
  BiShengIRLinalgDialectExt
)
