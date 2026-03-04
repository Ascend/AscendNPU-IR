set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BishengIRTorch)
add_public_tablegen_target(BishengIRTorchTransformsIncGen)
add_dependencies(mlir-headers BishengIRTorchTransformsIncGen)