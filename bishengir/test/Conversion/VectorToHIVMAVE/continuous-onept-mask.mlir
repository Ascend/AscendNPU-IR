// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9579 -convert-vector-to-hivmave -convert-arith-to-hivmave -analyze-vector-layout -remove-vector-layout-attr -ave-normalize-ops -canonicalize -convert-hivmave-to-ave-intrin -canonicalize | FileCheck %s

// A runtime extent is restricted by the caller to 0 or 1. Both are defined:
// active=0 must write nothing; active=1 writes one element at the strided
// address. Shape one proves at most one write, not that a write is required,
// so the physical predicate must survive to the store. The rank-1 strided WORD
// write lowers to a scatter carrying that predicate (never the maskless vstus
// ONEPT fast path); its address is base + lane*stride, not a contiguous
// lane-0 write.
// CHECK-LABEL: @shape_one_runtime_mask
// CHECK: %[[PLT:.*]] = "hivm_regbaseintrins.intr.hivm.plt.b32.v300"
// CHECK: %[[MASK:.*]] = llvm.extractvalue %[[PLT]][0]
// CHECK: %[[IDX:.*]] = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"
// CHECK-NOT: intr.hivm.vstus
// CHECK-NOT: intr.hivm.vstsx1
// CHECK: "hivm_regbaseintrins.intr.hivm.vscatter_v64s32_v300"({{.*}}, %[[IDX]], %[[MASK]])
// CHECK-NOT: intr.hivm.vstus
func.func @shape_one_runtime_mask(%dst: memref<1024xf32, #hivm.address_space<ub>>, %n: index, %active: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %v = arith.constant dense<7.0> : vector<64xf32>
  scf.for %iv = %c0 to %n step %c64 {
    %sub = memref.subview %dst[%iv] [1] [2] : memref<1024xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
    %m = vector.create_mask %active : vector<64xi1>
    vector.transfer_write %v, %sub[%c0], %m {in_bounds = [true]} : vector<64xf32>, memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// Test the consumer directly too: an empty-at-runtime predicate must remain
// predicated. The guard denies the maskless ONEPT fast path, so the store keeps
// its mask operand and the hardware honors an empty predicate as "no write".
// CHECK-LABEL: @active_zero
// CHECK: %[[OFF:.*]] = "hivm_regbaseintrins.intr.hivm.pge.b32"
// CHECK-NOT: intr.hivm.vstus
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1{{.*}}"({{.*}}, %[[OFF]])
func.func @active_zero(%dst: memref<1xf32, #hivm.address_space<ub>>, %v: vector<64xf32>) {
  %c0 = arith.constant 0 : index
  %m = ave.hir.pge <ALLF> : vector<64xi1>
  ave.hir.masked_store <ONEPT_B32> %dst[%c0], %m, %v {hivm.is_continuous} : memref<1xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// A proven VL1 predicate still permits the existing maskless fast path:
// lane zero is provably active, so discarding the mask is safe here.
// CHECK-LABEL: @active_one
// CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.f32"
// CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
func.func @active_one(%dst: memref<1xf32, #hivm.address_space<ub>>, %v: vector<64xf32>) {
  %c0 = arith.constant 0 : index
  %m = ave.hir.pge <VL1> : vector<64xi1>
  ave.hir.masked_store <ONEPT_B32> %dst[%c0], %m, %v {hivm.is_continuous} : memref<1xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  return
}

// Constant active=0 specialization: the mask folds to empty, so the rank-1
// strided WORD write lowers to a scatter whose empty predicate canonicalizes
// away entirely -- no store is emitted. An empty predicate writes nothing.
// CHECK-LABEL: @shape_one_active_0
// CHECK-NOT: intr.hivm.vstus
// CHECK-NOT: intr.hivm.vstsx1
// CHECK-NOT: intr.hivm.vscatter
func.func @shape_one_active_0(%dst: memref<1024xf32, #hivm.address_space<ub>>, %n: index) {
  %active = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %v = arith.constant dense<7.0> : vector<64xf32>
  scf.for %iv = %c0 to %n step %c64 {
    %sub = memref.subview %dst[%iv] [1] [2] : memref<1024xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
    %m = vector.create_mask %active : vector<64xi1>
    vector.transfer_write %v, %sub[%c0], %m {in_bounds = [true]} : vector<64xf32>, memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// Constant active=1 specialization: one element written through the predicated
// vstsx1 path at the strided address base + index*stride, never the maskless
// continuous ONEPT write.
// CHECK-LABEL: @shape_one_active_1
// CHECK: llvm.mul
// CHECK: %[[PTR1:.*]] = llvm.getelementptr
// CHECK-NOT: intr.hivm.vstus
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1{{.*}}"({{.*}}, %[[PTR1]],{{.*}})
func.func @shape_one_active_1(%dst: memref<1024xf32, #hivm.address_space<ub>>, %n: index) {
  %active = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %v = arith.constant dense<7.0> : vector<64xf32>
  scf.for %iv = %c0 to %n step %c64 {
    %sub = memref.subview %dst[%iv] [1] [2] : memref<1024xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
    %m = vector.create_mask %active : vector<64xi1>
    vector.transfer_write %v, %sub[%c0], %m {in_bounds = [true]} : vector<64xf32>, memref<1xf32, strided<[2], offset: ?>, #hivm.address_space<ub>>
  }
  return
}
