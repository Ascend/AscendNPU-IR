// RUN: bishengir-opt --hivm-inline-otf-load-store --cse --canonicalize-ext -split-input-file %s | FileCheck %s
// CHECK-LABEL:   func.func @concat_store(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<8x4xf32>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<8x4xf32>) -> tensor<8x8xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<8x8xf32>
// CHECK: %[[VAL_3:.*]] = tensor.insert_slice %[[VAL_0]] into %[[VAL_2]][0, 0] [8, 4] [1, 1] : tensor<8x4xf32> into tensor<8x8xf32>
// CHECK: %[[VAL_4:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_3]][0, 4] [8, 4] [1, 1] : tensor<8x4xf32> into tensor<8x8xf32>
// CHECK: %[[VAL_5:.*]] = hivm.hir.store ins(%[[VAL_4]] : tensor<8x8xf32>) outs(%[[VAL_2]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: return %[[VAL_5]] : tensor<8x8xf32>
func.func @concat_store(%arg0 : tensor<8x4xf32>, %arg1 : tensor<8x4xf32>) -> tensor<8x8xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %init = tensor.empty() : tensor<8x8xf32>
  %0 = hivm.hir.vconcat dim(1) ins(%arg0, %arg1 : tensor<8x4xf32>, tensor<8x4xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = tensor.empty() : tensor<8x8xf32>
  %stored = hivm.hir.store ins(%0 : tensor<8x8xf32>) outs(%1 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %stored: tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @concat_store_annotation
// CHECK: %[[VAL_9:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] {{\[}}%{{.*}}, 2] [1, 1] : tensor<?x2xf32> into tensor<1x4xf32>
// CHECK: %[[VAL_11:.*]] = tensor.insert_slice %{{.*}} into %[[VAL_9]][0, 2] {{\[}}%{{.*}}, 2] [1, 1] : tensor<?x2xf32> into tensor<1x4xf32>
// CHECK: %[[VAL_12:.*]] = hivm.hir.store ins(%[[VAL_11]] : tensor<1x4xf32>) outs(%{{.*}} : tensor<?x4xf32>) atomic = <none> -> tensor<?x4xf32>
func.func @concat_store_annotation(%arg0 : tensor<?x2xf32>, %arg1 : tensor<?x2xf32>, %arg6 : tensor<2x4xf32>, %arg3 : index, %arg4 : index) -> tensor<2x4xf32>
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %dynsize = arith.constant 1 : index
  %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, 0] [%arg4, 4] [1, 1] : tensor<2x4xf32> to tensor<?x4xf32>
  %init = tensor.empty(%dynsize) : tensor<?x4xf32>
  %31 = hivm.hir.vconcat dim(1) ins(%arg0, %arg1 : tensor<?x2xf32>, tensor<?x2xf32>) outs(%init : tensor<?x4xf32>) -> tensor<?x4xf32>
  annotation.mark %31 {buffer_size_in_byte = 39296 : i64} : tensor<?x4xf32>
  %32 = hivm.hir.store ins(%31 : tensor<?x4xf32>) outs(%extracted_slice_1 : tensor<?x4xf32>) atomic = <none> -> tensor<?x4xf32>
  %inserted_slice = tensor.insert_slice %32 into %arg6[%arg3, 0] [%arg4, 4] [1, 1] : tensor<?x4xf32> into tensor<2x4xf32>
  return %inserted_slice : tensor<2x4xf32>
}

// -----

// CHECK-LABEL:   func.func @concat_store_dynamic(
// CHECK: %[[VAL_7:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [8, %{{.*}}] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
// CHECK: %[[VAL_9:.*]] = tensor.insert_slice %{{.*}} into %[[VAL_7]][0, %{{.*}}] [8, %{{.*}}] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
// CHECK: %[[VAL_10:.*]] = hivm.hir.store ins(%[[VAL_9]] : tensor<8x?xf32>) outs(%{{.*}} : tensor<8x?xf32>) -> tensor<8x?xf32>
// CHECK: return %[[VAL_10]] : tensor<8x?xf32>
// CHECK: }
func.func @concat_store_dynamic(%arg0 : tensor<8x?xf32>, %arg1 : tensor<8x?xf32>, %arg2 : tensor<8x?xf32>) -> tensor<8x?xf32> 
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %dynsize = tensor.dim %arg2 , %c1 : tensor<8x?xf32>
  %init = tensor.empty(%dynsize) : tensor<8x?xf32>
  %0 = hivm.hir.vconcat dim(1) ins(%arg0, %arg1 : tensor<8x?xf32>, tensor<8x?xf32>) outs(%init : tensor<8x?xf32>) -> tensor<8x?xf32>
  %1 = tensor.empty(%dynsize) : tensor<8x?xf32>
  %stored = hivm.hir.store ins(%0 : tensor<8x?xf32>) outs(%1 : tensor<8x?xf32>) -> tensor<8x?xf32>
  return %stored: tensor<8x?xf32>
}
// -----

// CHECK-LABEL: func @test_tensor_to_memref
// CHECK: %[[VAL_15:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0] [256] [1] : tensor<256xi8> into tensor<510xi8>
// CHECK: %[[VAL_16:.*]] = tensor.insert_slice %{{.*}} into %[[VAL_15]][256] [254] [1] : tensor<254xi8> into tensor<510xi8>
// CHECK: hivm.hir.store ins(%[[VAL_16]] : tensor<510xi8>) outs(%{{.*}} : memref<510xi8, strided<[1]>>)
func.func @test_tensor_to_memref(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}, %arg2: memref<?xi8> {tt.divisibility = 16 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8, strided<[1]>>
  %alloc = memref.alloc() : memref<256xi8>
  hivm.hir.load ins(%reinterpret_cast : memref<256xi8, strided<[1]>>) outs(%alloc : memref<256xi8>)
  %0 = bufferization.to_tensor %alloc restrict writable : memref<256xi8>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [254], strides: [1] : memref<?xi8> to memref<254xi8, strided<[1]>>
  %alloc_1 = memref.alloc() : memref<254xi8>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<254xi8, strided<[1]>>) outs(%alloc_1 : memref<254xi8>)
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<254xi8>
  %2 = tensor.empty() : tensor<510xi8>
  %3 = hivm.hir.vconcat dim(0) ins(%0, %1 : tensor<256xi8>, tensor<254xi8>) outs(%2 : tensor<510xi8>) -> tensor<510xi8>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [510], strides: [1] : memref<?xi8> to memref<510xi8, strided<[1]>>
  hivm.hir.store ins(%3 : tensor<510xi8>) outs(%reinterpret_cast_2 : memref<510xi8, strided<[1]>>)
  return
}