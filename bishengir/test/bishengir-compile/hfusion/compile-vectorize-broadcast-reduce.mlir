// RUN: bishengir-opt %s -hfusion-auto-vectorize="max-vectorize-axes=1" -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @test_broadcast_fuse_reduce_outlined_vf_1
func.func @test_broadcast_fuse_reduce() -> tensor<48x16xi64>
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
        %c0_i64 = arith.constant 0 : i64
        %13 = tensor.empty() : tensor<48x16xi64>
        %14 = tensor.empty() : tensor<48x16xi64>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%14 : tensor<48x16xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<48x16xi64>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13 : tensor<48x16xi64>) outs(%15 : tensor<48x16xi64>) {
    ^bb0(%in: i64, %out: i64):
      %17 = linalg.index 1 : index
      %18 = arith.index_cast %17 : index to i32
      %19 = arith.extsi %18 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : i32 to i64
      %20 = arith.cmpi eq, %in, %19 : i64
      %21 = arith.uitofp %20 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : i1 to f32
      %22 = arith.fptosi %21 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : f32 to i64
      %23 = arith.addi %22, %out : i64
      linalg.yield %23 : i64
    } -> tensor<48x16xi64>
        return %16 : tensor<48x16xi64>
}