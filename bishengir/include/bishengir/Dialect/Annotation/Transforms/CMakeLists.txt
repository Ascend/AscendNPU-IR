set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Annotation)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Annotation)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Annotation)
add_public_tablegen_target(BiShengIRAnnotationTransformsIncGen)
add_dependencies(mlir-headers BiShengIRAnnotationTransformsIncGen)

add_bishengir_doc(Passes AnnotationPasses ./ -gen-pass-doc)
