// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin %s | FileCheck %s --check-prefix=INTR
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -ave-replace-with-vector-scalar -convert-hivmave-to-ave-intrin -convert-func-to-llvm -reconcile-unrealized-casts %s | bishengir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// Check the scalar stays BF16, the mask is converted to a physical predicate,
// and the exact overloaded LLVM name is emitted without FP32 promotion.
// INTR-LABEL: func.func @vmaxs_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// INTR-NOT: ave.hir.vmaxs
// LLVM-LABEL: define <128 x bfloat> @vmaxs_bf16(
// LLVM-NOT: fpext
// LLVM: call <128 x bfloat> @llvm.hivm.vmaxs.s.x.v128bf16(<128 x bfloat> %{{.*}}, bfloat %{{.*}}, <256 x i1> %{{.*}})
// LLVM-NOT: fptrunc
func.func @vmaxs_bf16(%vec: vector<128xbf16>, %scalar: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %res = ave.hir.vmaxs %vec, %scalar, %mask : vector<128xbf16>, bf16, vector<128xi1>
  return %res : vector<128xbf16>
}

// INTR-LABEL: func.func @vmins_bf16
// INTR: "hivm_regbaseintrins.intr.hivm.vmins.s.x"({{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
// INTR-NOT: ave.hir.vmins
// LLVM-LABEL: define <128 x bfloat> @vmins_bf16(
// LLVM-NOT: fpext
// LLVM: call <128 x bfloat> @llvm.hivm.vmins.s.x.v128bf16(<128 x bfloat> %{{.*}}, bfloat %{{.*}}, <256 x i1> %{{.*}})
// LLVM-NOT: fptrunc
func.func @vmins_bf16(%vec: vector<128xbf16>, %scalar: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %res = ave.hir.vmins %vec, %scalar, %mask : vector<128xbf16>, bf16, vector<128xi1>
  return %res : vector<128xbf16>
}

// Existing scalar dispatch must still work.
// INTR-LABEL: func.func @vmaxs_f16
// INTR: "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"({{.*}}) : (vector<128xf16>, f16, vector<256xi1>) -> vector<128xf16>
// LLVM-LABEL: define <128 x half> @vmaxs_f16(
// LLVM: call <128 x half> @llvm.hivm.vmaxs.s.x.v128f16
func.func @vmaxs_f16(%vec: vector<128xf16>, %scalar: f16) -> vector<128xf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %res = ave.hir.vmaxs %vec, %scalar, %mask : vector<128xf16>, f16, vector<128xi1>
  return %res : vector<128xf16>
}

// INTR-LABEL: func.func @vmins_f32
// INTR: "hivm_regbaseintrins.intr.hivm.vmins.s.x"({{.*}}) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
// LLVM-LABEL: define <64 x float> @vmins_f32(
// LLVM: call <64 x float> @llvm.hivm.vmins.s.x.v64f32
func.func @vmins_f32(%vec: vector<64xf32>, %scalar: f32) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %res = ave.hir.vmins %vec, %scalar, %mask : vector<64xf32>, f32, vector<64xi1>
  return %res : vector<64xf32>
}

// Exercise the fold and its downstream lowering together, with a real
// predicate producer instead of a hand-written regbase intrinsic.
// INTR-LABEL: func.func @broadcast_minmax_bf16
// INTR-NOT: ave.hir.scalar_broadcast
// INTR: "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"
// INTR: "hivm_regbaseintrins.intr.hivm.vmins.s.x"
// INTR-NOT: ave.hir.vmin
// LLVM-LABEL: define <128 x bfloat> @broadcast_minmax_bf16(
// LLVM-NOT: fpext
// LLVM: call <128 x bfloat> @llvm.hivm.vmaxs.s.x.v128bf16
// LLVM: call <128 x bfloat> @llvm.hivm.vmins.s.x.v128bf16
// LLVM-NOT: fptrunc
func.func @broadcast_minmax_bf16(%vec: vector<128xbf16>, %lo: bf16, %hi: bf16) -> vector<128xbf16> {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %lob = ave.hir.scalar_broadcast %lo : bf16 -> vector<128xbf16>
  %hib = ave.hir.scalar_broadcast %hi : bf16 -> vector<128xbf16>
  %max = ave.hir.vmax %vec, %lob, %mask : vector<128xbf16>, vector<128xi1>
  %min = ave.hir.vmin %max, %hib, %mask : vector<128xbf16>, vector<128xi1>
  return %min : vector<128xbf16>
}
