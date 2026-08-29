# Dialect-neutral schedule infrastructure: value handles, transform op
# builder, strategy base, clone-retry driver, interpreter and verifier.
# Only the value handle system is migrated so far; the remaining sources are
# added as their migrations land.
add_bishengir_dialect_library(BiShengIRSchedule
  Builder.cpp
  ValueHandle.cpp

  LINK_LIBS PUBLIC
  BiShengIRAnalysisTransforms
  BiShengIRSCFTransformOps
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
)
