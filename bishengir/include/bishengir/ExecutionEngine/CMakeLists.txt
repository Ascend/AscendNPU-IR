if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ExecutionEngine)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ExecutionEngine)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ExecutionEngine)
add_public_tablegen_target(MLIRExecutionEngineTransformsIncGen)
add_dependencies(mlir-headers MLIRExecutionEngineTransformsIncGen)

add_bishengir_doc(Passes ExecutionEnginePasses ./ -gen-pass-doc)