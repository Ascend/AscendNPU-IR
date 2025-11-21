set(LLVM_TARGET_DEFINITIONS ScopeOps.td)
mlir_tablegen(ScopeDialect.h.inc -gen-dialect-decls -dialect=scope)
mlir_tablegen(ScopeDialect.cpp.inc -gen-dialect-defs -dialect=scope)
mlir_tablegen(ScopeOps.h.inc -gen-op-decls)
mlir_tablegen(ScopeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BiShengIRScopeIncGen)

add_bishengir_doc(ScopeOps ScopeDialect Dialects/ -gen-dialect-doc -dialect=scope)
