set(LLVM_TARGET_DEFINITIONS BiShengIRAttrs.td)
mlir_tablegen(BiShengIREnums.h.inc -gen-enum-decls)
mlir_tablegen(BiShengIREnums.cpp.inc -gen-enum-defs)
mlir_tablegen(BiShengIRAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=bishengir)
mlir_tablegen(BiShengIRAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=bishengir)
mlir_tablegen(BiShengIRTypes.h.inc -gen-typedef-decls -typedefs-dialect=bishengir)
mlir_tablegen(BiShengIRTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=bishengir)
add_public_tablegen_target(BiShengIRBiShengIREnumsIncGen)
add_public_tablegen_target(BiShengIRBiShengIRAttributesIncGen)
add_dependencies(mlir-headers BiShengIRBiShengIREnumsIncGen)
add_dependencies(mlir-headers BiShengIRBiShengIRAttributesIncGen)

add_mlir_interface(AggregatedOpInterface)

# Vendor CopyOpInterface only on LLVM 22+ (MLIR removed it in PR #157711).
# On LLVM < 22 MLIR still ships it and the shared include guard in
# CopyOpInterface.td deduplicates against MLIR's copy.
if(LLVM_MAJOR_VERSION_22_COMPATIBLE)
  add_mlir_interface(CopyOpInterface)
endif()
