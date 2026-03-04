// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9589 -hfusion-legalize-bf16 %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_floor_bf16_not_to_f32
// CHECK-NOT: hfusion.cast
func.func @test_floor_bf16_not_to_f32(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %0 = arith.index_cast %arg7 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%0], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<1xbf16>
  memref.copy %reinterpret_cast, %alloc : memref<1xbf16, strided<[1], offset: ?>> to memref<1xbf16>
  %1 = bufferization.to_tensor %alloc restrict writable : memref<1xbf16>
  %2 = tensor.empty() : tensor<1xbf16>
  %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%1 : tensor<1xbf16>) outs(%2 : tensor<1xbf16>) -> tensor<1xbf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%0], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<1xbf16>, memref<1xbf16, strided<[1], offset: ?>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @test_gather_bf16_not_to_f32
// CHECK-NOT: hfusion.cast
func.func @test_gather_bf16_not_to_f32(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<?xi32> {tt.divisibility = 16 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {WorkspaceArgIdx = 0 : i64, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16, 4, 64], strides: [256, 64, 1] : memref<?xbf16> to memref<16x4x64xbf16, strided<[256, 64, 1]>>
  %alloc = memref.alloc() : memref<16x4x64xbf16>
  memref.copy %reinterpret_cast, %alloc : memref<16x4x64xbf16, strided<[256, 64, 1]>> to memref<16x4x64xbf16>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<16x4x64xbf16>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 4, 32], strides: [128, 32, 1] : memref<?xi32> to memref<16x4x32xi32, strided<[128, 32, 1]>>
  %alloc_1 = memref.alloc() : memref<16x4x32xi32>
  memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x4x32xi32, strided<[128, 32, 1]>> to memref<16x4x32xi32>
  %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x4x32xi32>
  %2 = tensor.empty() : tensor<16x4x32xbf16>
  %3 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%0, %1 : tensor<16x4x64xbf16>, tensor<16x4x32xi32>) outs(%2 : tensor<16x4x32xbf16>) axis = 2 -> tensor<16x4x32xbf16>
  %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 4, 32], strides: [128, 32, 1] : memref<?xbf16> to memref<16x4x32xbf16, strided<[128, 32, 1]>>
  bufferization.materialize_in_destination %3 in writable %reinterpret_cast_2 : (tensor<16x4x32xbf16>, memref<16x4x32xbf16, strided<[128, 32, 1]>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @linalg_reduce_with_index
// CHECK: linalg.reduce
// CHECK-SAME: tensor<2x51x3x13x1xf32>
// CHECK: arith.cmpf olt
// CHECK-SAME: f32
// CHECK: linalg.yield
// CHECK-SAME: f32, i32
func.func @linalg_reduce_with_index(%0: tensor<2x51x3x13x1xbf16>, %1: tensor<2x51x3x13x1xi32>) -> tensor<2x51x3x13xi32> {
  %true = arith.constant true
  %2 = tensor.empty() : tensor<2x51x3x13xbf16>
  %3 = tensor.empty() : tensor<2x51x3x13xi32>
  %reduced:2 = linalg.reduce ins(%0, %1 : tensor<2x51x3x13x1xbf16>, tensor<2x51x3x13x1xi32>) outs(%2, %3 : tensor<2x51x3x13xbf16>, tensor<2x51x3x13xi32>) dimensions = [4]  {reduce_mode = "min_with_index", tie_break_left = "true", unsigned_src = "false"}
    (%in: bf16, %in_5: i32, %init: bf16, %init_6: i32) {
      %6 = arith.cmpf olt, %in, %init : bf16
      %7 = arith.cmpf oeq, %in, %init : bf16
      %8 = arith.cmpf une, %in, %in : bf16
      %9 = arith.cmpf une, %init, %init : bf16
      %10 = arith.xori %9, %true : i1
      %11 = arith.andi %8, %10 : i1
      %12 = arith.ori %6, %11 : i1
      %13 = arith.andi %8, %9 : i1
      %14 = arith.ori %7, %13 : i1
      %15 = arith.cmpi slt, %in_5, %init_6 : i32
      %16 = arith.andi %14, %15 : i1
      %17 = arith.ori %12, %16 : i1
      %18 = arith.select %17, %in, %init : bf16
      %19 = arith.select %17, %in_5, %init_6 : i32
      linalg.yield %18, %19 : bf16, i32
    }
  return %reduced#1 : tensor<2x51x3x13xi32>
}