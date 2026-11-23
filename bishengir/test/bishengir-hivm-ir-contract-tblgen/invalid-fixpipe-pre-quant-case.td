// RUN: not bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-decls %s 2>&1 | FileCheck %s

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

// A pre-quant case without any typed signature would silently bypass the
// contract.
def TestNoSignature : HIVM_FixpipePreQuantCase<"NO_SIG", 99, []>;

// A typed case that is not registered in HIVM_FixpipePreQuantModeEnum would
// generate dead contract code.
def TestOrphan : HIVM_FixpipePreQuantCase<"ORPHAN", 98, [Sig_F32_F16]>;

// CHECK: error: Fixpipe pre-quant case 'NO_SIG' must declare at least one src/dst type signature
// CHECK: error: Fixpipe pre-quant case 'ORPHAN' is not referenced by HIVM_FixpipePreQuantModeEnum
