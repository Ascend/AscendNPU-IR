// RUN: bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-decls %s | FileCheck %s --check-prefix=DECL
// RUN: bishengir-hivm-ir-contract-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %p/../../../third-party/llvm-project/mlir/include \
// RUN:   -I %p/../../../third-party/llvm-project/llvm/include \
// RUN:   -gen-fixpipe-pre-quant-contract-defs %s | FileCheck %s --check-prefix=DEF

include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

// The typed enum cases in HIVMAttrs.td are the single source of truth; this
// test locks the generated forward verifier, descriptor tables and reverse
// candidate lookup.

// DECL: struct FixpipePreQuantSignatureDesc {
// DECL: getFixpipePreQuantSignatures(::mlir::hivm::FixpipePreQuantMode mode);
// DECL: verifyFixpipePreQuantSignature(
// DECL: getFixpipePreQuantCandidates(::mlir::Type srcElementType,

// DEF: static bool matchHIVM_FIXPIPEPREQUANT_NO_QUANT_Sig_F32_F32
// DEF: static const FixpipePreQuantSignatureDesc kHIVM_FIXPIPEPREQUANT_F322F16Signatures[] =
// DEF: case ::mlir::hivm::FixpipePreQuantMode::NO_QUANT: return kHIVM_FIXPIPEPREQUANT_NO_QUANTSignatures;
// DEF: case ::mlir::hivm::FixpipePreQuantMode::F322F16: return kHIVM_FIXPIPEPREQUANT_F322F16Signatures;
// DEF: "pre_quant mode '"
// DEF: << ::mlir::hivm::stringifyFixpipePreQuantMode(mode)
// DEF: "one of ["
// DEF: candidates.push_back(::mlir::hivm::FixpipePreQuantMode::S322I8);
