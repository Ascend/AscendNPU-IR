add_bishengir_library(BishengIRTransform
  CanonicalizeModule.cpp
  DeadFunctionElimination.cpp
  LowerToCPUBackend.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Transforms

  DEPENDS
  BishengIRTransformsPassIncGen
)