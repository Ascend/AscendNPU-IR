set(tblgen_feat_list)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BishengIRTransforms ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRAscendDPXTransformsIncGen)
add_dependencies(mlir-headers BiShengIRAscendDPXTransformsIncGen)
