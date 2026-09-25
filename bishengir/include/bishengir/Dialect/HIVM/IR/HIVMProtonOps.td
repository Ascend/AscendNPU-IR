#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_TD

def ProtonScopeBoundaryOp
    : HIVM_Op<"proton_scope_boundary", [MemoryEffects<[MemRead, MemWrite]>,
                                        CubeVectorCoreTypeTrait]> {
  let summary = "Synchronized A5 Proton scope boundary";
  let arguments = (ins I32Attr:$region_id);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = opNameDeclaration;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_TD
