set(tblgen_feat_list)
if(BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  list(APPEND tblgen_feat_list -DBISHENGIR_BUILD_STANDALONE_IR_ONLY)
endif()

if(LLVM_MAJOR_VERSION_21_COMPATIBLE)
  list(APPEND tblgen_feat_list -D__LLVM_MAJOR_VERSION_21_COMPATIBLE__)
endif()
if(LLVM_MAJOR_VERSION_22_COMPATIBLE)
  list(APPEND tblgen_feat_list -D__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
endif()

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
    BiShengIR${output_file}YamlIncGen
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
  BiShengIRHFusionNamedStructuredOpsYamlIncGen
)
add_dependencies(mlir-headers HFusionOdsGen)

# TODO: Better to do it with the add_bishengir_dialect()
set(LLVM_TARGET_DEFINITIONS HFusionOps.td)
mlir_tablegen(HFusionOps.h.inc -gen-op-decls)
mlir_tablegen(HFusionOps.cpp.inc -gen-op-defs)
mlir_tablegen(HFusionOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=hfusion)
mlir_tablegen(HFusionOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hfusion)
mlir_tablegen(HFusionOpsDialect.h.inc -gen-dialect-decls -dialect=hfusion ${tblgen_feat_list})
mlir_tablegen(HFusionOpsDialect.cpp.inc -gen-dialect-defs -dialect=hfusion ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRHFusionOpsIncGen)
add_dependencies(mlir-headers BiShengIRHFusionOpsIncGen)

add_bishengir_doc(HFusionDoc HFusionDialect Dialects/ -gen-dialect-doc -dialect=hfusion)
add_dependencies(HFusionDialectDocGen HFusionOdsGen)

set(LLVM_TARGET_DEFINITIONS HFusionStructuredOps.td)
mlir_tablegen(HFusionStructuredOps.h.inc -gen-op-decls ${tblgen_feat_list})
mlir_tablegen(HFusionStructuredOps.cpp.inc -gen-op-defs ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRHFusionStructuredOpsIncGen)
add_dependencies(BiShengIRHFusionStructuredOpsIncGen HFusionOdsGen)
add_dependencies(mlir-headers BiShengIRHFusionStructuredOpsIncGen)

set(LLVM_TARGET_DEFINITIONS HFusionEnums.td)
set(LLVM_TARGET_DEFINITIONS HFusionAttrs.td)
mlir_tablegen(HFusionEnums.h.inc -gen-enum-decls)
mlir_tablegen(HFusionEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HFusionAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hfusion)
mlir_tablegen(HFusionAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hfusion)
add_public_tablegen_target(BiShengIRHFusionEnumsIncGen)
add_public_tablegen_target(BiShengIRHFusionAttributesIncGen)
add_dependencies(mlir-headers BiShengIRHFusionEnumsIncGen)
add_dependencies(mlir-headers BiShengIRHFusionAttributesIncGen)
