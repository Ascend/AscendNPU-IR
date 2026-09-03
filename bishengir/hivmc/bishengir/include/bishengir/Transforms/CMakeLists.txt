set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name BishengIRTransform)
mlir_tablegen(BishengIRTransform.capi.h.inc -gen-pass-capi-header --prefix BishengIRTransform)
mlir_tablegen(BishengIRTransform.capi.cpp.inc -gen-pass-capi-impl --prefix BishengIRTransform)

add_public_tablegen_target(BishengIRTransformsPassIncGen)
add_dependencies(mlir-headers BishengIRTransformsPassIncGen)
