# Declare a function to generate ODS with bishengir-hfusion-ods-yaml-gen
function(add_hfusion_ods_yaml_gen yaml_ast_file output_file)
  set(YAML_AST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${yaml_ast_file})
  set(GEN_ODS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.td)
  set(GEN_CPP_FILE ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.yamlgen.cpp.inc)
  set_source_files_properties(
    ${GEN_ODS_FILE}
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${GEN_CPP_FILE}
    PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT ${GEN_ODS_FILE} ${GEN_CPP_FILE}
    COMMAND ${BISHENGIR_HFUSION_ODS_YAML_GEN_EXE} ${YAML_AST_SOURCE} -o-ods-decl=${GEN_ODS_FILE} -o-impl=${GEN_CPP_FILE}
    MAIN_DEPENDENCY
    ${YAML_AST_SOURCE}
    DEPENDS
    ${BISHENGIR_HFUSION_ODS_YAML_GEN_EXE}
    ${BISHENGIR_HFUSION_ODS_YAML_GEN_TARGET})
  add_custom_target(
    MLIR${output_file}YamlIncGen
    DEPENDS
    ${BISHENGIR_HFUSION_ODS_YAML_GEN_EXE}
    ${BISHENGIR_HFUSION_ODS_YAML_GEN_TARGET}
    ${GEN_ODS_FILE} ${GEN_CPP_FILE})
  list(APPEND LLVM_TARGET_DEPENDS ${GEN_ODS_FILE})
  set(LLVM_TARGET_DEPENDS ${LLVM_TARGET_DEPENDS} PARENT_SCOPE)
endfunction()

# NOTE: LLVM_TARGET_DEPENDS gets picked up by tablegen targets to add file
# level dependencies. This is gross but CMake requires depending on both
# targets and generated files, and it must be done when the custom target is
# declared (there is no way to add after the fact).
set(LLVM_TARGET_DEPENDS "")
add_hfusion_ods_yaml_gen(HFusionNamedStructuredOps.yaml HFusionNamedStructuredOps)

# Provide a short name for all external dependency that needs to
# include HFusion in ODS
add_custom_target(HFusionOdsGen
  DEPENDS
  MLIRHFusionNamedStructuredOpsYamlIncGen
)
add_dependencies(mlir-headers HFusionOdsGen)

add_mlir_dialect(HFusionOps hfusion)

add_bishengir_doc(HFusionDoc HFusionDialect Dialects/ -gen-dialect-doc -dialect=hfusion)
add_dependencies(HFusionDialectDocGen HFusionOdsGen)

set(LLVM_TARGET_DEFINITIONS HFusionStructuredOps.td)
mlir_tablegen(HFusionStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(HFusionStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRHFusionStructuredOpsIncGen)
add_dependencies(MLIRHFusionStructuredOpsIncGen HFusionOdsGen)
add_dependencies(mlir-headers MLIRHFusionStructuredOpsIncGen)

set(LLVM_TARGET_DEFINITIONS HFusionEnums.td)
set(LLVM_TARGET_DEFINITIONS HFusionAttrs.td)
mlir_tablegen(HFusionEnums.h.inc -gen-enum-decls)
mlir_tablegen(HFusionEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HFusionAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hfusion)
mlir_tablegen(HFusionAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hfusion)
add_public_tablegen_target(MLIRHFusionEnumsIncGen)
add_public_tablegen_target(MLIRHFusionAttributesIncGen)
add_dependencies(mlir-headers MLIRHFusionEnumsIncGen)
add_dependencies(mlir-headers MLIRHFusionAttributesIncGen)
