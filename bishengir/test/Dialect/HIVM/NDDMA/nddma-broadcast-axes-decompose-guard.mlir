// RUN: bishengir-opt %s -hivm-aggregated-decompose-op | FileCheck %s

// A broadcast descriptor is one transfer: the decompose pass rebuilds the load
// from its padding fields, so it must leave the descriptor (mask and shapes)
// untouched. The second function is the control: an ordinary padded load still
// takes the decompose path.
// CHECK-LABEL: func.func @broadcast_load_not_decomposed(
// CHECK:       hivm.hir.load {{.*}} outs(%{{.*}} : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
// CHECK-NOT:   hivm.hir.vbrc
// CHECK:       return
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @broadcast_load_not_decomposed(%src: memref<1x32xf32, #hivm.address_space<gm>>) {
  %alloc = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%alloc : memref<4x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}

// CHECK-LABEL: func.func @padded_load_still_decomposed(
// CHECK:       hivm.hir.vbrc
// CHECK:       hivm.hir.load
// CHECK-SAME:  pad_mode = <PadValue>
// CHECK-NOT:   init_out_buffer = true
// CHECK:       return
func.func @padded_load_still_decomposed(%src: memref<4x32xf32, #hivm.address_space<gm>>, %cst: f32) {
  %alloc = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x32xf32, #hivm.address_space<gm>>) outs(%alloc : memref<4x32xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 init_out_buffer = true
  return
}
}
