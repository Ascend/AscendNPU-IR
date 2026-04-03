// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_1(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_2(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_3(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_4(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}

// CHECK-LABEL: func.func private @triton_unk_fused_gelu_0_1_fused_5(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func @triton_unk_fused_gelu_0_1(
// CHECK: arith.constant
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_0
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_1
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_2
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_3
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_4
// CHECK: func.call @triton_unk_fused_gelu_0_1_fused_5
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module {
  func.func @triton_unk_fused_gelu_0_1(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 26267.2246 : f32
    %cst_0 = arith.constant 13243.3662 : f32
    %cst_1 = arith.constant 3023.12476 : f32
    %cst_2 = arith.constant 398.569641 : f32
    %cst_3 = arith.constant 31.2128582 : f32
    %cst_4 = arith.constant 29639.3848 : f32
    %cst_5 = arith.constant 5063.7915 : f32
    %cst_6 = arith.constant 1393.80615 : f32
    %cst_7 = arith.constant 101.62809 : f32
    %cst_8 = arith.constant 7.55170154 : f32
    %cst_9 = arith.constant 0.0534437485 : f32
    %cst_10 = arith.constant -3.920000e+00 : f32
    %cst_11 = arith.constant 3.920000e+00 : f32
    %cst_12 = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c168522_i32 = arith.constant 168522 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %cst_13 = arith.constant 5.000000e-01 : f32
    %cst_14 = arith.constant 0.707106769 : f32
    %c1_i32 = arith.constant 1 : i32
    %c83_i32 = arith.constant 83 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_15 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048xf32>
    %1 = arith.muli %arg8, %c168522_i32 : i32
    %2 = arith.addi %1, %c168522_i32 : i32
    %3 = arith.minsi %2, %arg4 : i32
    scf.for %arg11 = %c0_i32 to %c83_i32 step %c1_i32  : i32 {
      %4 = arith.muli %arg11, %c2048_i32 : i32
      %5 = arith.addi %1, %4 : i32
      %6 = arith.index_cast %5 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<2048xf32>
      %7 = arith.addi %6, %c2048 : index
      %8 = arith.index_cast %3 : i32 to index
      %9 = arith.maxsi %6, %8 : index
      %10 = arith.minsi %7, %9 : index
      %11 = arith.subi %10, %6 : index
      %12 = arith.cmpi slt, %11, %c2048 : index
      scf.if %12 {
        linalg.fill ins(%cst_12 : f32) outs(%alloc : memref<2048xf32>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_16 = memref.subview %alloc[0] [%11] [1] : memref<2048xf32> to memref<?xf32>
      memref.copy %subview, %subview_16 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32>
      %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %cst_13 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %cst_14 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %16 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%15, %cst_11 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %17 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%16, %cst_10 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %17 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %cst_9 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %cst_8 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %cst_7 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%23, %cst_6 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %cst_5 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%27, %cst_4 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %28 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%18, %cst_3 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%31, %cst_2 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%32, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%33, %cst_1 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%34, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %cst_0 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%36, %18 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%37, %cst : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%29, %38 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%39, %cst_15 : tensor<2048xf32>, f32) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %40 : tensor<2048xf32>, tensor<2048xf32>) outs(%0 : tensor<2048xf32>) -> tensor<2048xf32>
      %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %41[0] [%11] [1] : tensor<2048xf32> to tensor<?xf32>
      %subview_18 = memref.subview %reinterpret_cast_17[0] [%11] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_18 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}