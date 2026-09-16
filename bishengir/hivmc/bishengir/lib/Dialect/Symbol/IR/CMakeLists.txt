add_bishengir_dialect_library(BiShengIRSymbolDialect
  SymbolDialect.cpp
  SymbolOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Symbol

  DEPENDS
  BiShengIRSymbolIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  MLIRDestinationStyleOpInterface
)
