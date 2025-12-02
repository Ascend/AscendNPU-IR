set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Scope)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Scope)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Scope)
add_public_tablegen_target(BiShengIRScopeTransformsIncGen)
add_dependencies(mlir-headers BiShengIRScopeTransformsIncGen)

add_bishengir_doc(Passes ScopePasses ./ -gen-pass-doc)