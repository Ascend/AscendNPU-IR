// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2="enable-multiple-consumer-fusion=true" -o %t 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: FileCheck %s --check-prefix=PASS-CHECK < %t
//
// Multi-consumer fusion retry: the row-reduction producer
// (tensor<1x16384xf32> -> tensor<1xf32>, dim 1) feeds both the sub/div
// elementwise chain and a direct extract_slice store (softmax returns amax
// alongside the result). With enable-multiple-consumer-fusion=true the first
// attempt fuses the reduction producer into the consumer group, but the
// loop-signature rebuild cannot yield the reduced value back to the external
// users, so the untiled original op survives with its label and the vectorize
// step fails on it ("Attempted to vectorize, but failed"). AutoVectorizeV2
// must retry once with multiple-consumer fusion disabled instead of failing
// the whole pass.
//
// DIAG: Attempted to vectorize, but failed
// DIAG: AutoVectorizeV2 failed; retrying with enableMultipleConsumerFusion=false

// PASS-CHECK-LABEL: func.func @softmax_mcf_retry
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> (0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @softmax_mcf_retry(%arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg5: memref<?xf32>, %arg6: i64, %arg7: i32, %arg9: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384 = arith.constant 16384 : index
    scf.for %arg16 = %c0_i32 to %arg9 step %c1_i32 : i32 {
      %iv = arith.index_cast %arg16 : i32 to index
      %zext = arith.extsi %arg16 : i32 to i64
      %5 = arith.cmpi slt, %arg16, %arg7 : i32
      %6 = arith.muli %arg6, %zext : i64
      %7 = arith.index_cast %6 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%7], sizes: [1, 16384], strides: [16384, 1] : memref<?xf32> to memref<1x16384xf32, strided<[16384, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<1x16384xf32, strided<[16384, 1], offset: ?>> into memref<16384xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1x16384xf32>
      %8 = arith.index_castui %5 : i1 to index
      %9 = arith.muli %8, %c16384 : index
      %subview = memref.subview %collapse_shape[0] [%9] [1] : memref<16384xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 0] [1, %9] [1, 1] : memref<1x16384xf32> to memref<1x?xf32, strided<[16384, 1]>>
      %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1]] : memref<1x?xf32, strided<[16384, 1]>> into memref<?xf32, strided<[1]>>
      memref.copy %subview, %collapse_shape_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %14 = bufferization.to_tensor %alloc restrict writable : memref<1x16384xf32>
      %15 = tensor.empty() : tensor<1xf32>
      %16 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<1x16384xf32>) outs(%15 : tensor<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %25 = arith.maximumf %in, %out : f32
        linalg.yield %25 : f32
      } -> tensor<1xf32>
      %18 = tensor.empty() : tensor<1x16384xf32>
      %19 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%14, %16 : tensor<1x16384xf32>, tensor<1xf32>) outs(%18 : tensor<1x16384xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %25 = arith.subf %in, %in_10 : f32
        linalg.yield %25 : f32
      } -> tensor<1x16384xf32>
      %collapsed = tensor.collapse_shape %19 [[0, 1]] : tensor<1x16384xf32> into tensor<16384xf32>
      %21 = tensor.empty() : tensor<16384xf32>
      %22 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel"]} ins(%collapsed, %16 : tensor<16384xf32>, tensor<1xf32>) outs(%21 : tensor<16384xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %25 = arith.divf %in, %in_10 : f32
        linalg.yield %25 : f32
      } -> tensor<16384xf32>
      %expanded = tensor.expand_shape %22 [[0, 1]] output_shape [1, 16384] : tensor<16384xf32> into tensor<1x16384xf32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%7], sizes: [1, 16384], strides: [16384, 1] : memref<?xf32> to memref<1x16384xf32, strided<[16384, 1], offset: ?>>
      %extracted_slice = tensor.extract_slice %expanded[0, 0] [1, %9] [1, 1] : tensor<1x16384xf32> to tensor<1x?xf32>
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [1, %9] [1, 1] : memref<1x16384xf32, strided<[16384, 1], offset: ?>> to memref<1x?xf32, strided<[16384, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_3 : (tensor<1x?xf32>, memref<1x?xf32, strided<[16384, 1], offset: ?>>) -> ()
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%iv], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
      %extracted_slice_5 = tensor.extract_slice %16[0] [%8] [1] : tensor<1xf32> to tensor<?xf32>
      %subview_6 = memref.subview %reinterpret_cast_4[0] [%8] [1] : memref<1xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_5 in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
