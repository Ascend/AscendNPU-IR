add_bishengir_dialect_library(MLIRSymbolTransforms
  EraseSymbol.cpp
  PropagateSymbol.cpp
  SymbolAndEncodingConverter.cpp
  UnfoldSymbolicInt.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/Symbol/Transforms

  DEPENDS
  MLIRSymbolTransformsIncGen

  LINK_LIBS PUBLIC
  BiShengIRDialectUtils
  BiShengIRSymbolDialect
  BiShengIRSymbolUtils
  MLIRDialect
  MLIRFuncDialect
  MLIRHFusionDialect
  MLIRHFusionUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
)
