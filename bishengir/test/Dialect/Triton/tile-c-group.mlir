// RUN: bishengir-opt --tile-dot-loads='k-tile-size=1' %s | FileCheck %s

// Input is the grouped C-chain immediately before TileDotLoads.
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<16x16xf32>) {
// CHECK: [[FIRST:%[0-9]+]] = tt.dot {{.*}}, {{.*}}, {{.*}} {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64, bishengir.dot.tiled} : tensor<16x1xf32> * tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK: [[SECOND:%[0-9]+]] = tt.dot {{.*}}, {{.*}}, [[FIRST]] {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64, bishengir.dot.tiled} : tensor<16x1xf32> * tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK: scf.yield [[SECOND]] : tensor<16x16xf32>

module {
  tt.func @tile_c_group(%a_ptr: !tt.ptr<f32>, %b0_ptr: !tt.ptr<f32>,
                         %b1_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i64
    %c1 = arith.constant 1 : i64
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %pa = tt.make_tensor_ptr %a_ptr, [%c16, %c16], [%c16, %c1], [%c0, %c0]
      {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %pb0 = tt.make_tensor_ptr %b0_ptr, [%c16, %c16], [%c16, %c1], [%c0, %c0]
      {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %pb1 = tt.make_tensor_ptr %b1_ptr, [%c16, %c16], [%c16, %c1], [%c0, %c0]
      {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %po = tt.make_tensor_ptr %out_ptr, [%c16, %c16], [%c16, %c1], [%c0, %c0]
      {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %a = tt.load %pa : !tt.ptr<tensor<16x16xf32>>
    %b0 = tt.load %pb0 : !tt.ptr<tensor<16x16xf32>>
    %b1 = tt.load %pb1 : !tt.ptr<tensor<16x16xf32>>
    %b0t = arith.addf %b0, %zero : tensor<16x16xf32>
    %b1t = arith.addf %b1, %zero : tensor<16x16xf32>
    %d0 = tt.dot %a, %b0t, %zero {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %d1 = tt.dot %a, %b1t, %d0 {bishengir.dot.c_grouped_for_overlap, bishengir.dot.group_id = 0 : i64} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.store %po, %d1 : !tt.ptr<tensor<16x16xf32>>
    tt.return
  }
}
