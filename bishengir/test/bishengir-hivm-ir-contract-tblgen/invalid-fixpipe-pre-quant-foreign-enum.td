// RUN: not bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-defs %s 2>&1 | FileCheck %s

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

// A typed case in an unrelated enum must not be emitted as a
// FixpipePreQuantMode case.
def Sig_Foreign_F8_F8 : ElemTypeSignature<F8E4M3FN, F8E5M2>;
def TEST_FOREIGN_FP8 :
    HIVM_FixpipePreQuantCase<"FOREIGN_FP8", 42, [Sig_Foreign_F8_F8]>;
def ForeignPreQuantModeEnum : HIVM_I32Enum<
    "ForeignPreQuantMode", "foreign pre-quant mode", [TEST_FOREIGN_FP8]>;

// CHECK: error: Fixpipe pre-quant case 'FOREIGN_FP8' is not referenced by HIVM_FixpipePreQuantModeEnum
