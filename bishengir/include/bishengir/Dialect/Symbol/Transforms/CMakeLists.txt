set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Symbol)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Symbol)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Symbol)
add_public_tablegen_target(MLIRSymbolTransformsIncGen)
add_dependencies(mlir-headers MLIRSymbolTransformsIncGen)

add_bishengir_doc(Passes SymbolPasses ./ -gen-pass-doc)