add_bishengir_dialect_library(BiShengIRScopeDialect
  ScopeDialect.cpp
  ScopeOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Scope

  DEPENDS
  BiShengIRScopeIncGen

  LINK_LIBS PUBLIC
  ${BS_FEAT_LINK_LIBS}
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
)
