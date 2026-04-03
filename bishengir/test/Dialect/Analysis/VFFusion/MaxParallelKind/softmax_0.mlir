// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_fused_0(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.bitcast
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_fused_1(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
// CHECK: hfusion.select
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: linalg.yield

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_fused_2(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.yield

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_fused_3(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func @triton_unk_fused__softmax_0(
// CHECK: arith.constant
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.minsi
// CHECK: scf.for
// CHECK: arith.cmpi slt
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: arith.index_castui
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: tensor.empty
// CHECK: func.call @triton_unk_fused__softmax_0_fused_0
// CHECK: func.call @triton_unk_fused__softmax_0_fused_1
// CHECK: tensor.extract
// CHECK: func.call @triton_unk_fused__softmax_0_fused_2
// CHECK: func.call @triton_unk_fused__softmax_0_fused_3
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module {
 func.func @triton_unk_fused__softmax_0(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c-2139095040_i32 = arith.constant -2139095040 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %c659_i32 = arith.constant 659 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c384_i32 = arith.constant 384 : i32
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg9, %c659_i32 : i32
    %1 = arith.addi %0, %c659_i32 : i32
    %2 = arith.minsi %1, %arg4 : i32
    scf.for %arg12 = %c0_i32 to %c659_i32 step %c1_i32  : i32 {
      %3 = arith.addi %0, %arg12 : i32
      %4 = arith.cmpi slt, %3, %2 : i32
      %5 = arith.muli %3, %c384_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [384], strides: [1] : memref<?xf16> to memref<384xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<384xf16>
      %7 = arith.index_castui %4 : i1 to index
      %8 = arith.muli %7, %c384 : index
      %9 = arith.cmpi slt, %7, %c1 : index
      %10 = arith.cmpi slt, %8, %c384 : index
      %11 = arith.ori %9, %10 : i1
      scf.if %11 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<384xf16>)
      } {hivm.unlikely_condition}
      %12 = arith.muli %7, %8 : index
      %subview = memref.subview %reinterpret_cast[0] [%12] [1] : memref<384xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_2 = memref.subview %alloc[0] [%12] [1] : memref<384xf16> to memref<?xf16>
      memref.copy %subview, %subview_2 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<384xf16>
      %14 = tensor.empty() : tensor<384xf32>
      %15 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%13 : tensor<384xf16>) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %16 = tensor.empty() : tensor<f32>
      %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<f32>) -> tensor<f32>
      %18 = tensor.empty() : tensor<384xi32>
      %19 = hfusion.bitcast ins(%15 : tensor<384xf32>) outs(%18 : tensor<384xi32>) -> tensor<384xi32>
      %20 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%19, %c2147483647_i32 : tensor<384xi32>, i32) outs(%18 : tensor<384xi32>) -> tensor<384xi32>
      %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%20, %c-2139095040_i32 : tensor<384xi32>, i32) outs(%18 : tensor<384xi32>) -> tensor<384xi32>
      %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%21, %c1_i32 : tensor<384xi32>, i32) outs(%21 : tensor<384xi32>) -> tensor<384xi32>
      %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%22, %c0_i32 : tensor<384xi32>, i32) outs(%22 : tensor<384xi32>) -> tensor<384xi32>
      %24 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%23 : tensor<384xi32>) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %25 = tensor.empty() : tensor<384xi1>
      %26 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%24, %cst_1 : tensor<384xf32>, f32) outs(%25 : tensor<384xi1>) -> tensor<384xi1>
      %27 = hfusion.select ins(%26, %cst_0, %15 : tensor<384xi1>, f32, tensor<384xf32>) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %reduced = linalg.reduce ins(%27 : tensor<384xf32>) outs(%17 : tensor<f32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %34 = arith.maximumf %in, %init : f32
          linalg.yield %34 : f32
        }
      %extracted = tensor.extract %reduced[] : tensor<f32>
      %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %extracted : tensor<384xf32>, f32) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %29 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%28 : tensor<384xf32>) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %30 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<f32>) -> tensor<f32>
      %reduced_3 = linalg.reduce ins(%29 : tensor<384xf32>) outs(%30 : tensor<f32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %34 = arith.addf %in, %init : f32
          linalg.yield %34 : f32
        }
      %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
      %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%29, %extracted_4 : tensor<384xf32>, f32) outs(%14 : tensor<384xf32>) -> tensor<384xf32>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [384], strides: [1] : memref<?xf16> to memref<384xf16, strided<[1], offset: ?>>
      %32 = tensor.empty() : tensor<384xf16>
      %33 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%31 : tensor<384xf32>) outs(%32 : tensor<384xf16>) -> tensor<384xf16>
      %extracted_slice = tensor.extract_slice %33[0] [%12] [1] : tensor<384xf16> to tensor<?xf16>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%12] [1] : memref<384xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}