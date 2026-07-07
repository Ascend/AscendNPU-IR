set(tblgen_feat_list)
if(BISHENGIR_ENABLE_TRITON_COMPILE)
  list(APPEND tblgen_feat_list -DBISHENGIR_ENABLE_TRITON_COMPILE)
endif()

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BiShengIRTriton ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRTritonTransformsIncGen)
add_dependencies(mlir-headers BiShengIRTritonTransformsIncGen)