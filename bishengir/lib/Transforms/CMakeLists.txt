add_bishengir_library(BiShengIRTransform
  AffineMinMaxValueBounds.cpp
  CanonicalizeModule.cpp
  DeadFunctionElimination.cpp
  EliminateSingleIterationScfFor.cpp
  DebugMemory.cpp
  InjectIR.cpp
  ExtendedCanonicalizer.cpp
  FuseReductionIntoLoop.cpp
  LowerToCPUBackend.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Transforms

  DEPENDS
  BiShengIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  BiShengIRLinalgCanonicalizations
  BiShengIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRAVEDialect
)
