add_bishengir_dialect_library(MLIRAVEDialect
  HIVMAVE.cpp
  HIVMAVEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVMAVE

  DEPENDS
  MLIRAVEIncGen
  MLIRAVEEnumsIncGen
  MLIRAVEAttributesIncGen
  MLIRAVEOpsIncGen
  MLIRAVETypesIncGen
  MLIRHIVMEnumsIncGen
  MLIRHIVMAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefExtDialect
  MLIRSideEffectInterfaces
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRTransformUtils
  MLIRViewLikeInterface
  BiShengIRDialectUtils 
)
