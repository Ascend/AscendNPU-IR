set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BishengIRLLVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix BishengIRLLVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix BishengIRLLVM)

add_public_tablegen_target(BishengIRLLVMTransformsIncGen)
add_dependencies(mlir-headers BishengIRLLVMTransformsIncGen)
add_bishengir_doc(Passes BiShengIRLLVMPasses ./ -gen-pass-doc)