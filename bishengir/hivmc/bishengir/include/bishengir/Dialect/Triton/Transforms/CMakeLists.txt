set(tblgen_feat_list)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BiShengIRTriton ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRTritonTransformsIncGen)
add_dependencies(mlir-headers BiShengIRTritonTransformsIncGen)
