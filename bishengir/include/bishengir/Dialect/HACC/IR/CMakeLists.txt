add_bishengir_dialect(HACCBase hacc)
add_bishengir_doc(HACCBase HACCDialect Dialects/ -gen-dialect-doc)

# HACC IR depends on TargetSpec
add_dependencies(BiShengIRHACCBaseIncGen BiShengIRNPUTargetSpecIncGen)

# Collect TableGen feature flags
set(tblgen_feat_list)
if(LLVM_MAJOR_VERSION_21_COMPATIBLE)
  list(APPEND tblgen_feat_list -D__LLVM_MAJOR_VERSION_21_COMPATIBLE__)
endif()

set(LLVM_TARGET_DEFINITIONS HACCAttrs.td)
mlir_tablegen(HACCEnums.h.inc -gen-enum-decls ${tblgen_feat_list})
mlir_tablegen(HACCEnums.cpp.inc -gen-enum-defs ${tblgen_feat_list})
mlir_tablegen(HACCAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hacc ${tblgen_feat_list})
mlir_tablegen(HACCAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hacc ${tblgen_feat_list})
mlir_tablegen(HACCTypes.h.inc -gen-typedef-decls -typedefs-dialect=hacc ${tblgen_feat_list})
mlir_tablegen(HACCTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=hacc ${tblgen_feat_list})
add_public_tablegen_target(BiShengIRHACCEnumsIncGen)
add_public_tablegen_target(BiShengIRHACCAttributesIncGen)
add_dependencies(mlir-headers BiShengIRHACCEnumsIncGen)
add_dependencies(mlir-headers BiShengIRHACCAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS HACCInterfaces.td)
mlir_tablegen(HACCInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HACCInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(HACCAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(HACCAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(BiShengIRHACCInterfacesIncGen)
add_dependencies(mlir-headers BiShengIRHACCInterfacesIncGen)
