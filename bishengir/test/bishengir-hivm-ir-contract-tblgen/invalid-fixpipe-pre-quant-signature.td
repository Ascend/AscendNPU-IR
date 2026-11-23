// RUN: not bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-decls %s 2>&1 | FileCheck %s

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

// A signature slot must reference an ODS Type constraint with a real
// predicate; an empty predicate is rejected instead of turning into a
// matcher that accepts everything.
def NoPredType : Type<CPred<"">, "no predicate">;
def TestBadType : ElemTypeSignature<F32, NoPredType>;
def TestBadTypeCase : HIVM_FixpipePreQuantCase<"BAD_TYPE", 96, [TestBadType]>;

// A predicate that needs more context than $_self cannot be expanded into a
// standalone matcher.
def NeedsOpType : Type<CPred<"$_op.getType().isF32()">, "needs op">;
def TestOpDepSig : ElemTypeSignature<F32, NeedsOpType>;
def TestOpDepCase : HIVM_FixpipePreQuantCase<"OP_DEP", 95, [TestOpDepSig]>;

// The same signature twice in one case is a generation-time error.
def TestDup : HIVM_FixpipePreQuantCase<"DUP", 97, [Sig_F32_F16, Sig_F32_F16]>;

// CHECK: error: predicate of 'NoPredType' cannot be expanded in a self-only context (empty predicate or unresolved placeholder)
// CHECK: error: duplicate signature 'F32->F16' in Fixpipe pre-quant case 'DUP'
// CHECK: error: predicate of 'NeedsOpType' cannot be expanded in a self-only context (empty predicate or unresolved placeholder)
// CHECK: error: Fixpipe pre-quant case 'BAD_TYPE' is not referenced by HIVM_FixpipePreQuantModeEnum
// CHECK: error: Fixpipe pre-quant case 'DUP' is not referenced by HIVM_FixpipePreQuantModeEnum
// CHECK: error: Fixpipe pre-quant case 'OP_DEP' is not referenced by HIVM_FixpipePreQuantModeEnum
