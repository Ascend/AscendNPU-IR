set(LLVM_TARGET_DEFINITIONS TransformOps.td)
mlir_tablegen(TransformOps.h.inc -gen-op-decls)
mlir_tablegen(TransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BiShengIRAnalysisTransformsIncGen)
# The doc output name differs from the td basename to avoid a target name
# collision with the upstream transform dialect's TransformOpsDocGen.
add_bishengir_doc(TransformOps AnalysisScheduleTransformOps Dialects/ -gen-op-doc)
add_dependencies(mlir-headers BiShengIRAnalysisTransformsIncGen)
