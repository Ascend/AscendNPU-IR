// RUN: bishengir-opt %s | FileCheck %s
// CHECK-LABEL: func.func @planned_bounded
// CHECK: hivm.hir.load
// CHECK-SAME: broadcast_axes = [1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @planned_bounded(%src: memref<64x1xf32, #hivm.address_space<gm>>, %raw: index) {
 %z = arith.constant 0 : i64
 %c32 = arith.constant 32 : index
 %n = arith.minui %raw, %c32 : index
 %p = hivm.hir.pointer_cast(%z) : memref<32x8x1xf32, #hivm.address_space<ub>>
 %base = memref.subview %p[0,0,0] [32,4,1] [1,1,1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x4xf32, strided<[8,1]>, #hivm.address_space<ub>>
 %s = memref.subview %src[9,0] [%n,1] [1,1] : memref<64x1xf32, #hivm.address_space<gm>> to memref<?x1xf32, strided<[1,1], offset:9>, #hivm.address_space<gm>>
 %d = memref.subview %base[0,0] [%n,4] [1,1] : memref<32x4xf32, strided<[8,1]>, #hivm.address_space<ub>> to memref<?x4xf32, strided<[8,1]>, #hivm.address_space<ub>>
 hivm.hir.load ins(%s : memref<?x1xf32, strided<[1,1], offset:9>, #hivm.address_space<gm>>) outs(%d : memref<?x4xf32, strided<[8,1]>, #hivm.address_space<ub>>) broadcast_axes = [1]
 return
}

}
