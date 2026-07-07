set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AVE)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix AVE)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix AVE)
add_public_tablegen_target(MLIRAVETransformsIncGen)
add_dependencies(mlir-headers MLIRAVETransformsIncGen)
 
add_mlir_doc(Passes AVEPasses ./ -gen-pass-doc)