add_bishengir_library(BishengIRSCFTransforms
  MapForToForall.cpp
  RemoveRedundantLoopInit.cpp
  CanonicalizeIterArg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/SCF/Transforms

  DEPENDS
  BishengIRSCFTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRSCFUtils
  MLIRSCFDialect
  MLIRIR
  MLIRDialect
  MLIRPass
)
