// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -hfusion-auto-vectorize | FileCheck %s

// -----
func.func @triton_load_store_one_grid(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %c1_i32 = arith.constant 1 : i32
  %c6422 = arith.constant 6422 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  scf.for %arg12 = %c0_i32 to %arg5 step %c16_i32  : i32 {
    %0 = arith.index_cast %arg12 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16xi32>
    %1 = arith.addi %0, %c16 : index
    %2 = arith.maxsi %0, %c6422 : index
    %3 = arith.minsi %1, %2 : index
    %4 = arith.subi %3, %0 : index
    %5 = arith.cmpi slt, %4, %c16 : index
    scf.if %5 {
      // CHECK: annotation.mark %alloc keys = ["pad_const"] values = [%c1_i32 : i32] : memref<16xi32>
      linalg.fill ins(%c1_i32 : i32) outs(%alloc : memref<16xi32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<16xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%4] [1] : memref<16xi32> to memref<?xi32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
    %6 = bufferization.to_tensor %alloc restrict writable : memref<16xi32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%0], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %6 in writable %reinterpret_cast_1 : (tensor<16xi32>, memref<16xi32, strided<[1], offset: ?>>) -> ()
  }
  return
}

// -----
func.func @triton_dot_3_None(%arg0: i64, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
  %cst = arith.constant 0.000000e00 : f32
  %0 = tensor.empty() : tensor<87x4x3xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<87x4x3xf32>) -> tensor<87x4x3xf32>
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [87, 4, 3], strides: [12, 3, 1] : memref<?xf32> to memref<87x4x3xf32, strided<[12, 3, 1]>>
  %alloc = memref.alloc() : memref<87x4x3xf32>
  memref.copy %reinterpret_cast, %alloc : memref<87x4x3xf32, strided<[12, 3, 1]>> to memref<87x4x3xf32>
  %2 = bufferization.to_tensor %alloc restrict writable : memref<87x4x3xf32>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [87, 3, 3], strides: [9, 3, 1] : memref<?xf32> to memref<87x3x3xf32, strided<[9, 3, 1]>>
  %alloc_1 = memref.alloc() : memref<87x3x3xf32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<87x3x3xf32, strided<[9, 3, 1]>> to memref<87x3x3xf32>
  %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<87x3x3xf32>
  // CHECK: linalg.batch_matmul
  %4 = linalg.batch_matmul {input_precison = "ieee"} ins(%2, %3 : tensor<87x4x3xf32>, tensor<87x3x3xf32>) outs(%1 : tensor<87x4x3xf32>) -> tensor<87x4x3xf32>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [87, 4, 3], strides: [12, 3, 1] : memref<?xf32> to memref<87x4x3xf32, strided<[12, 3, 1]>>
  bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<87x4x3xf32>, memref<87x4x3xf32, strided<[12, 3, 1]>>) -> ()
  return
} 

// -----
// CHECK-LABEL: func.func @test_extract_inline

// CHECK-NOT: tensor.extract

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @test_extract_inline(%arg0: tensor<8x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<f32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<f32>
  
  %res0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction", "reduction"]
  } ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<f32>
  
  %scalar0 = tensor.extract %res0[] : tensor<f32>
  %scalar1 = tensor.extract %arg1[%c0] : tensor<1xf32>

  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map1, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %scalar0, %scalar1 : tensor<8x8xf32>, f32, f32) 
    outs(%arg2 : tensor<8x8xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %4 = arith.addf %in, %in_0 : f32
    %5 = arith.addf %4, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<8x8xf32>

  return %3 : tensor<8x8xf32>
}