// RUN: bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-defs %s | FileCheck %s

include "bishengir/Dialect/HIVM/IR/Contracts/FixpipePreQuantContract.td"

// Gate: adding a dtype that is absent from the generator's C++ is a TD-only
// change. The FP8 types are not part of the production pre-quant cases; the
// predicate and the diagnostic spelling come from their ODS TypeConstraint.
def Sig_F8_F8 : ElemTypeSignature<F8E4M3FN, F8E5M2>;
def TEST_FIXPIPEPREQUANT_FP8 : HIVM_FixpipePreQuantCase<"FP8", 42, [Sig_F8_F8]>;
def HIVM_FixpipePreQuantModeEnum : I32EnumAttr<
    "FixpipePreQuantMode", "test fixpipe pre_quant mode",
    [TEST_FIXPIPEPREQUANT_FP8]> {
  let cppNamespace = "::mlir::hivm";
}

// CHECK: static bool matchTEST_FIXPIPEPREQUANT_FP8_Sig_F8_F8
// CHECK: return (srcElementType.isFloat8E4M3FN()) && (dstElementType.isFloat8E5M2());
// CHECK: static const FixpipePreQuantSignatureDesc kTEST_FIXPIPEPREQUANT_FP8Signatures[] =
// CHECK: case ::mlir::hivm::FixpipePreQuantMode::FP8: return kTEST_FIXPIPEPREQUANT_FP8Signatures;
