// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-hivmave-to-ave-intrin -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | bishengir-translate -mlir-to-llvmir | FileCheck %s

// Expand block byte offsets to element offsets, with a lane mask and no UB spill.
// CHECK-LABEL: define <64 x float> @block_gather(
// CHECK: call <64 x i32> @llvm.hivm.vci
// CHECK: call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32({{.*}}i16 3
// CHECK: call <64 x i32> @llvm.hivm.vselr
// CHECK: call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32({{.*}}i16 2
// CHECK: call <64 x i32> @llvm.hivm.vgather2.v300.v64s32
// CHECK: bitcast <64 x i32> {{.*}} to <64 x float>
func.func @block_gather(%base: memref<1024xf32, 6>, %index: vector<64xi32>) -> vector<64xf32> {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %result = ave.hir.vgatherb %base[%c0][%index], %mask : memref<1024xf32, 6>, vector<64xi32>, vector<64xi1> into vector<64xf32>
  return %result : vector<64xf32>
}

// A scatter VF cannot claim disjoint root/view parameters. Both reads require
// the CANN reload compatibility handling, including a read before the scatter.
// CHECK-LABEL: define internal ptc_simdvf {{.*}}@scatter_readback(
// CHECK-NOT: noalias
// CHECK: {
// CHECK: call ptr addrspace(6) @llvm.launder.invariant.group
// CHECK: call <64 x i32> @llvm.hivm.vldsx1.v64s32
// CHECK: call void @llvm.hivm.vscatter.v64s32.v300
// CHECK: call void @llvm.hivm.mem.bar.vst.vld
// CHECK: call ptr addrspace(6) @llvm.launder.invariant.group
// CHECK: call <64 x i32> @llvm.hivm.vldsx1.v64s32
func.func @scatter_readback(%base: memref<128xi32, 6>, %alias: memref<128xi32, 6>, %index: vector<64xi32>) -> vector<64xi32> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %before = ave.hir.vload <NORM> %base[%c0] : memref<128xi32, 6> into vector<64xi32>
  ave.hir.vscatter %alias[%c0][%index], %mask, %before : memref<128xi32, 6>, vector<64xi32>, vector<64xi1>, vector<64xi32>
  %bar = arith.constant 1 : i32
  ave.hir.membar %bar
  %after = ave.hir.vload <NORM> %base[%c0] : memref<128xi32, 6> into vector<64xi32>
  return %after : vector<64xi32>
}

// CHECK: declare void @llvm.hivm.vscatter.v64s32.v300(<64 x i32>, ptr addrspace(6) nocapture writeonly,
