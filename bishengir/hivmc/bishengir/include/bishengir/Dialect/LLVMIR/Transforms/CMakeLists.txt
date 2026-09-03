set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BiShengIRLLVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix BiShengIRLLVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix BiShengIRLLVM)

add_public_tablegen_target(BiShengIRLLVMTransformsIncGen)
add_dependencies(mlir-headers BiShengIRLLVMTransformsIncGen)
add_bishengir_doc(Passes BiShengIRLLVMPasses ./ -gen-pass-doc)
