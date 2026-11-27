// RUN: split-file %s %t
// RUN: bishengir-opt %t/ave.mlir -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-hivmave-to-ave-intrin -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | bishengir-translate -mlir-to-llvmir | FileCheck %s
// RUN: bishengir-opt %t/block-args.mlir -hacc-append-device-spec=target=Ascend960DT_968B7 -convert-vector-to-hivmave -analyze-vector-layout -remove-vector-layout-attr -convert-hivmave-to-ave-intrin -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | bishengir-translate -mlir-to-llvmir | opt -S -passes=early-cse | FileCheck %s --check-prefix=ARG
// RUN: not bishengir-opt %t/ave.mlir -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin 2>&1 | FileCheck %s --check-prefix=A5
// RUN: bishengir-translate %t/a5.mlir -mlir-to-llvmir | FileCheck %s --check-prefix=RAW

//--- ave.mlir

// CHECK-LABEL: define internal ptc_simdvf {{.*}}@halfword_readback(
// CHECK-NOT: noalias
// CHECK: {
// CHECK: call ptr addrspace(6) @llvm.launder.invariant.group
// CHECK: call <128 x half> @llvm.hivm.vldsx1
// CHECK: bitcast <128 x half> {{.*}} to <128 x i16>
// CHECK: call void @llvm.hivm.vscatter.v128s16.v300
// CHECK: call void @llvm.hivm.mem.bar.vst.vld
// CHECK: call ptr addrspace(6) @llvm.launder.invariant.group
// CHECK: call <128 x half> @llvm.hivm.vldsx1
// CHECK: declare void @llvm.hivm.vscatter.v128s16.v300(<128 x i16>, ptr addrspace(6) nocapture writeonly,
// A5: failed to legalize operation 'ave.hir.vscatter'
func.func @halfword_readback(%root: memref<256xf16, 6>, %alias: memref<256xf16, 6>, %index: vector<128xi16>, %count: index) -> vector<128xf16> attributes {hivm.vector_function} {
  %zero = arith.constant 0 : index
  %mask, %rem = ave.hir.plt %count {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>, index
  %before = ave.hir.vload <NORM> %root[%zero] : memref<256xf16, 6> into vector<128xf16>
  ave.hir.vscatter %alias[%zero][%index], %mask, %before : memref<256xf16, 6>, vector<128xi16>, vector<128xi1>, vector<128xf16>
  %bar = arith.constant 1 : i32
  ave.hir.membar %bar
  %after = ave.hir.vload <NORM> %root[%zero] : memref<256xf16, 6> into vector<128xf16>
  return %after : vector<128xf16>
}

//--- block-args.mlir

// The caller can select any subset of lanes. Address generation and the store
// share each logical bit at its corresponding B16 or B32 predicate position.
// ARG-LABEL: define void @masked_column(
// ARG: %[[PACKED:.*]] = call <256 x i1> @llvm.vector.insert.v256i1.v128i1(<256 x i1> poison, <128 x i1> {{.*}}, i64 0)
// ARG: %[[MASK:.*]] = call <256 x i1> @llvm.hivm.punpack(<256 x i1> %[[PACKED]], i32 0)
// ARG: call <128 x i16> @llvm.hivm.vmuls{{.*}}<256 x i1> %[[MASK]])
// ARG: call void @llvm.hivm.vscatter.v128s16.v300({{.*}}<256 x i1> %[[MASK]])
func.func @masked_column(%dst: memref<?xf16, strided<[128], offset: ?>, 6>, %value: vector<128xf16>, %mask: vector<128xi1>, %base: index) {
  vector.transfer_write %value, %dst[%base], %mask {in_bounds = [true]} : vector<128xf16>, memref<?xf16, strided<[128], offset: ?>, 6>
  return
}

// ARG-LABEL: define void @masked_word_column(
// ARG: %[[PACKED:.*]] = call <256 x i1> @llvm.vector.insert.v256i1.v64i1(<256 x i1> poison, <64 x i1> {{.*}}, i64 0)
// ARG: %[[HALF:.*]] = call <256 x i1> @llvm.hivm.punpack(<256 x i1> %[[PACKED]], i32 0)
// ARG: %[[MASK:.*]] = call <256 x i1> @llvm.hivm.punpack(<256 x i1> %[[HALF]], i32 0)
// ARG: call <64 x i32> @llvm.hivm.vmuls{{.*}}<256 x i1> %[[MASK]])
// ARG: call void @llvm.hivm.vscatter.v64s32.v300({{.*}}<256 x i1> %[[MASK]])
func.func @masked_word_column(%dst: memref<?xf32, strided<[2], offset: ?>, 6>, %value: vector<64xf32>, %mask: vector<64xi1>, %base: index) {
  vector.transfer_write %value, %dst[%base], %mask {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[2], offset: ?>, 6>
  return
}

//--- a5.mlir

// Direct A5 intrinsic IR retains the LLVM attributes from its toolchain.
// RAW-LABEL: define internal ptc_simdvf void @raw_halfword(
// RAW-SAME: ptr addrspace(6) noalias
// RAW: declare void @llvm.hivm.vscatter.v128s16.v300(<128 x i16>, ptr addrspace(6) nocapture readonly,
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  llvm.func @raw_halfword(%dst: !llvm.ptr<6>, %value: vector<128xi16>, %index: vector<128xi16>, %mask: vector<256xi1>) attributes {hivm.vector_function} {
    "hivm_regbaseintrins.intr.hivm.vscatter.v128s16_v300"(%value, %dst, %index, %mask) : (vector<128xi16>, !llvm.ptr<6>, vector<128xi16>, vector<256xi1>) -> ()
    llvm.return
  }
}
