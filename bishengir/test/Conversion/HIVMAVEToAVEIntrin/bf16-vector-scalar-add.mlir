// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin %s | FileCheck %s --check-prefix=INTR
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin -convert-func-to-llvm -reconcile-unrealized-casts %s | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// Direct BF16 vector-scalar addition uses a physical 128-element data vector.
// INTR-LABEL: func.func @vadds_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vadds.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// INTR-NOT: ave.hir.vadds
// LLVM-LABEL: define <128 x bfloat> @vadds_bf16(
// LLVM-NOT: fpext
// LLVM: call <128 x bfloat> @llvm.hivm.vadds.s.x.v128bf16(<128 x bfloat> %{{.*}}, bfloat %{{.*}}, <256 x i1> %{{.*}})
// LLVM-NOT: fptrunc
func.func @vadds_bf16(%vec: vector<128xbf16>, %scalar: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %res = ave.hir.vadds %vec, %scalar, %mask : vector<128xbf16>, bf16, vector<128xi1>
  return %res : vector<128xbf16>
}

// BF16 scalar broadcasts fold to the vector-scalar addition.
// INTR-LABEL: func.func @broadcast_add_bf16
// INTR-NOT: ave.hir.scalar_broadcast
// INTR: "hivm_regbaseintrins.intr.hivm.vadds.s.x"
// INTR: "hivm_regbaseintrins.intr.hivm.vadds.s.x"
// LLVM-LABEL: define <128 x bfloat> @broadcast_add_bf16(
// LLVM-NOT: fpext
// LLVM: call <128 x bfloat> @llvm.hivm.vadds.s.x.v128bf16
// LLVM: call <128 x bfloat> @llvm.hivm.vadds.s.x.v128bf16
// LLVM-NOT: fptrunc
func.func @broadcast_add_bf16(%vec: vector<128xbf16>, %lhs: bf16, %rhs: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %lhs_b = ave.hir.scalar_broadcast %lhs : bf16 -> vector<128xbf16>
  %rhs_b = ave.hir.scalar_broadcast %rhs : bf16 -> vector<128xbf16>
  %first = ave.hir.vadd %vec, %lhs_b, %mask : vector<128xbf16>, vector<128xi1>
  %second = ave.hir.vadd %first, %rhs_b, %mask : vector<128xbf16>, vector<128xi1>
  return %second : vector<128xbf16>
}

// A masked BF16 broadcast folds when the producer and consumer masks match.
// INTR-LABEL: func.func @masked_broadcast_add_bf16
// INTR-NOT: ave.hir.broadcast
// INTR: "hivm_regbaseintrins.intr.hivm.vadds.s.x"
// LLVM-LABEL: define <128 x bfloat> @masked_broadcast_add_bf16(
// LLVM: call <128 x bfloat> @llvm.hivm.vadds.s.x.v128bf16
func.func @masked_broadcast_add_bf16(%vec: vector<128xbf16>, %scalar: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %bcast = ave.hir.broadcast %scalar, %mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vadd %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1>
  return %res : vector<128xbf16>
}

// Different producer and consumer masks prevent the BF16 fold.
// INTR-LABEL: func.func @different_mask_add_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vadd.s.x"
// INTR-NOT: "hivm_regbaseintrins.intr.hivm.vadds.s.x"
// LLVM-LABEL: define <128 x bfloat> @different_mask_add_bf16(
// LLVM: call <128 x bfloat> @llvm.hivm.vadd.s.x.v128bf16
// LLVM-NOT: @llvm.hivm.vadds.s.x.v128bf16
func.func @different_mask_add_bf16(%vec: vector<128xbf16>, %scalar: bf16) -> vector<128xbf16> {
  %producer_mask = ave.hir.pge <ALL> : vector<128xi1>
  %consumer_mask = ave.hir.pge <H> : vector<128xi1>
  %bcast = ave.hir.broadcast %scalar, %producer_mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vadd %vec, %bcast, %consumer_mask : vector<128xbf16>, vector<128xi1>
  return %res : vector<128xbf16>
}

// Existing FP16 vector-scalar addition remains supported.
// INTR-LABEL: func.func @vadds_f16
// INTR: "hivm_regbaseintrins.intr.hivm.vadds.s.x"({{.*}}) : (vector<128xf16>, f16, vector<256xi1>) -> vector<128xf16>
// LLVM-LABEL: define <128 x half> @vadds_f16(
// LLVM: call <128 x half> @llvm.hivm.vadds.s.x.v128f16
func.func @vadds_f16(%vec: vector<128xf16>, %scalar: f16) -> vector<128xf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %res = ave.hir.vadds %vec, %scalar, %mask : vector<128xf16>, f16, vector<128xi1>
  return %res : vector<128xf16>
}
