set(LLVM_TARGET_DEFINITIONS HIVMAVEBase.td)
mlir_tablegen(AVEDialect.h.inc -gen-dialect-decls -dialect=ave)
mlir_tablegen(AVEDialect.cpp.inc -gen-dialect-defs -dialect=ave)
add_public_tablegen_target(MLIRAVEIncGen)

set(LLVM_TARGET_DEFINITIONS HIVMAVEOps.td)
mlir_tablegen(AVEOps.h.inc -gen-op-decls)
mlir_tablegen(AVEOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAVEOpsIncGen)
add_dependencies(mlir-headers MLIRAVEOpsIncGen)

set(LLVM_TARGET_DEFINITIONS HIVMAVEAttrs.td)
mlir_tablegen(AVEEnums.h.inc -gen-enum-decls)
mlir_tablegen(AVEEnums.cpp.inc -gen-enum-defs)

mlir_tablegen(AVEAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ave)
mlir_tablegen(AVEAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ave)
add_public_tablegen_target(MLIRAVEEnumsIncGen)
add_public_tablegen_target(MLIRAVEAttributesIncGen)
add_dependencies(mlir-headers MLIRAVEEnumsIncGen)
add_dependencies(mlir-headers MLIRAVEAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS HIVMAVETypes.td)
mlir_tablegen(AVETypes.h.inc -gen-typedef-decls -typedefs-dialect=ave)
mlir_tablegen(AVETypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ave)
add_public_tablegen_target(MLIRAVETypesIncGen)
add_dependencies(mlir-headers MLIRAVETypesIncGen)

add_mlir_doc(AVEDoc AVEDialect Dialects/ -gen-dialect-doc -dialect=ave)

