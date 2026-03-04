// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s
func.func @test_extract_inline() -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %69 = bufferization.alloc_tensor() : tensor<i32>
  // CHECK:  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : tensor<1xi32>)
  %res0 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%c0_i32 : i32) outs(%69 : tensor<i32>) {
      ^bb0(%in: i32, %out: i32):
        linalg.yield %in : i32
      } -> tensor<i32>
  return %res0 : tensor<i32>
}