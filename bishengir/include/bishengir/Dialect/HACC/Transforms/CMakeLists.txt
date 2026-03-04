set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HACC)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix HACC)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix HACC)
add_public_tablegen_target(MLIRHACCTransformsIncGen)
add_dependencies(mlir-headers MLIRHACCTransformsIncGen)

add_bishengir_doc(Passes HACCPasses ./ -gen-pass-doc)