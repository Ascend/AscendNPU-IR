set(LLVM_TARGET_DEFINITIONS SymbolOps.td)
mlir_tablegen(SymbolDialect.h.inc -gen-dialect-decls -dialect=symbol)
mlir_tablegen(SymbolDialect.cpp.inc -gen-dialect-defs -dialect=symbol)
mlir_tablegen(SymbolOps.h.inc -gen-op-decls)
mlir_tablegen(SymbolOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BiShengIRSymbolIncGen)

add_bishengir_doc(SymbolOps SymbolDialect Dialects/ -gen-dialect-doc -dialect=symbol)
