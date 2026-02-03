add_bishengir_library(BiShengIRTransform
  CanonicalizeModule.cpp
  DeadFunctionElimination.cpp
  InjectIR.cpp
  ExtendedCanonicalizer.cpp
  LowerToCPUBackend.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Transforms

  DEPENDS
  BiShengIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRLinalgCanonicalizations
  BiShengIRMemRefDialect
)