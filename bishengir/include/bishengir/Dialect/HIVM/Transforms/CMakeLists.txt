set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HIVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix HIVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix HIVM)
add_public_tablegen_target(BiShengIRHIVMTransformsIncGen)
add_dependencies(mlir-headers BiShengIRHIVMTransformsIncGen)

add_bishengir_doc(Passes HIVMPasses ./ -gen-pass-doc)
