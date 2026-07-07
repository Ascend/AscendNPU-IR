set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BiShengIRTorch)
add_public_tablegen_target(BiShengIRTorchTransformsIncGen)
add_dependencies(mlir-headers BiShengIRTorchTransformsIncGen)
