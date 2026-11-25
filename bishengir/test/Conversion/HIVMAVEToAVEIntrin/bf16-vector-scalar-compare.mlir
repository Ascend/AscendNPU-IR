// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin %s | FileCheck %s --check-prefix=INTR
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin -convert-func-to-llvm -reconcile-unrealized-casts %s | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// INTR-LABEL: func.func @vcmps_all_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.eq.s.z"
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.ne.s.z"
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.gt.s.z"
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.ge.s.z"
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.lt.s.z"
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.le.s.z"
// LLVM-LABEL: define {{.*}} @vcmps_all_bf16(
// LLVM: @llvm.hivm.vcmps.eq.s.z.v128bf16
// LLVM: @llvm.hivm.vcmps.ne.s.z.v128bf16
// LLVM: @llvm.hivm.vcmps.gt.s.z.v128bf16
// LLVM: @llvm.hivm.vcmps.ge.s.z.v128bf16
// LLVM: @llvm.hivm.vcmps.lt.s.z.v128bf16
// LLVM: @llvm.hivm.vcmps.le.s.z.v128bf16
func.func @vcmps_all_bf16(%vec: vector<128xbf16>, %rhs: bf16) -> (vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>) {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %eq = ave.hir.vcmps <EQ> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  %ne = ave.hir.vcmps <NE> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  %gt = ave.hir.vcmps <GT> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  %ge = ave.hir.vcmps <GE> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  %lt = ave.hir.vcmps <LT> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  %le = ave.hir.vcmps <LE> %vec, %rhs, %mask : vector<128xbf16>, bf16, vector<128xi1> -> vector<256xi1>
  return %eq, %ne, %gt, %ge, %lt, %le : vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>, vector<256xi1>
}

// INTR-LABEL: func.func @broadcast_vcmps_bf16
// INTR-NOT: ave.hir.broadcast
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.eq.s.z"
// LLVM-LABEL: define <256 x i1> @broadcast_vcmps_bf16(
// LLVM: @llvm.hivm.vcmps.eq.s.z.v128bf16
func.func @broadcast_vcmps_bf16(%vec: vector<128xbf16>, %rhs: bf16) -> vector<256xi1> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %bcast = ave.hir.broadcast %rhs, %mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vcmp <EQ> %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}

// INTR-LABEL: func.func @different_mask_vcmps_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vcmp.eq.s.z"
// INTR-NOT: "hivm_regbaseintrins.intr.hivm.vcmps.eq.s.z"
// LLVM-LABEL: define <256 x i1> @different_mask_vcmps_bf16(
// LLVM: @llvm.hivm.vcmp.eq.s.z.v128bf16
// LLVM-NOT: @llvm.hivm.vcmps.eq.s.z.v128bf16
func.func @different_mask_vcmps_bf16(%vec: vector<128xbf16>, %rhs: bf16) -> vector<256xi1> {
  %producer_mask = ave.hir.pge <ALL> : vector<128xi1>
  %consumer_mask = ave.hir.pge <H> : vector<128xi1>
  %bcast = ave.hir.broadcast %rhs, %producer_mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vcmp <EQ> %vec, %bcast, %consumer_mask : vector<128xbf16>, vector<128xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}

// INTR-LABEL: func.func @nonbroadcast_vcmp_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vcmp.eq.s.z"
// INTR-NOT: "hivm_regbaseintrins.intr.hivm.vcmps"
// LLVM-LABEL: define <256 x i1> @nonbroadcast_vcmp_bf16(
// LLVM: @llvm.hivm.vcmp.eq.s.z.v128bf16
func.func @nonbroadcast_vcmp_bf16(%vec: vector<128xbf16>, %rhs: vector<128xbf16>) -> vector<256xi1> {
  %mask = ave.hir.pge <H> : vector<128xi1>
  %res = ave.hir.vcmp <EQ> %vec, %rhs, %mask : vector<128xbf16>, vector<128xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}

// INTR-LABEL: func.func @scalar_broadcast_vcmps_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vcmps.eq.s.z"
// LLVM-LABEL: define <256 x i1> @scalar_broadcast_vcmps_bf16(
// LLVM: @llvm.hivm.vcmps.eq.s.z.v128bf16
func.func @scalar_broadcast_vcmps_bf16(%vec: vector<128xbf16>, %rhs: bf16) -> vector<256xi1> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %bcast = ave.hir.scalar_broadcast %rhs : bf16 -> vector<128xbf16>
  %res = ave.hir.vcmp <EQ> %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1> -> vector<256xi1>
  return %res : vector<256xi1>
}
