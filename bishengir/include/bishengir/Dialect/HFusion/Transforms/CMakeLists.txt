set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HFusion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix HFusion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix HFusion)
add_public_tablegen_target(MLIRHFusionTransformsIncGen)
add_dependencies(mlir-headers MLIRHFusionTransformsIncGen)

add_bishengir_doc(Passes HFusionPasses ./ -gen-pass-doc)