// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_2_fused_0(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.bitcast
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_2_fused_1(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
// CHECK: hfusion.select
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: linalg.yield
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_2_fused_2(
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_0_2_fused_3(
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: tensor.extract_slice
// CHECK: return

// CHECK-LABEL: func.func @triton_unk_fused__softmax_0_2(
// CHECK: arith.constant
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.minsi
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: tensor.empty
// CHECK: func.call @triton_unk_fused__softmax_0_2_fused_0
// CHECK: func.call @triton_unk_fused__softmax_0_2_fused_1
// CHECK: func.call @triton_unk_fused__softmax_0_2_fused_2
// CHECK: func.call @triton_unk_fused__softmax_0_2_fused_3
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module{
   func.func @triton_unk_fused__softmax_0_2(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c-2139095040_i32 = arith.constant -2139095040 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c384 = arith.constant 384 : index
    %c1152_i32 = arith.constant 1152 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c36_i32 = arith.constant 36 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg9, %c1152_i32 : i32
    %1 = arith.addi %0, %c1152_i32 : i32
    %2 = arith.minsi %1, %arg4 : i32
    scf.for %arg12 = %c0_i32 to %c36_i32 step %c1_i32  : i32 {
      %3 = arith.muli %arg12, %c32_i32 : i32
      %4 = arith.addi %0, %3 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = arith.muli %5, %c384 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [32, 384], strides: [384, 1] : memref<?xf16> to memref<32x384xf16, strided<[384, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x384xf16>
      %7 = arith.addi %5, %c32 : index
      %8 = arith.index_cast %2 : i32 to index
      %9 = arith.maxsi %5, %8 : index
      %10 = arith.minsi %7, %9 : index
      %11 = arith.subi %10, %5 : index
      %12 = arith.cmpi slt, %11, %c32 : index
      scf.if %12 {
        linalg.fill ins(%cst_1 : f16) outs(%alloc : memref<32x384xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0, 0] [%11, 384] [1, 1] : memref<32x384xf16, strided<[384, 1], offset: ?>> to memref<?x384xf16, strided<[384, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[0, 0] [%11, 384] [1, 1] : memref<32x384xf16> to memref<?x384xf16, strided<[384, 1]>>
      memref.copy %subview, %subview_2 : memref<?x384xf16, strided<[384, 1], offset: ?>> to memref<?x384xf16, strided<[384, 1]>>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<32x384xf16>
      %14 = tensor.empty() : tensor<32x384xf32>
      %15 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%13 : tensor<32x384xf16>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %16 = tensor.empty() : tensor<32xf32>
      %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<32xf32>) -> tensor<32xf32>
      %18 = tensor.empty() : tensor<32x384xi32>
      %19 = hfusion.bitcast ins(%15 : tensor<32x384xf32>) outs(%18 : tensor<32x384xi32>) -> tensor<32x384xi32>
      %20 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%19, %c2147483647_i32 : tensor<32x384xi32>, i32) outs(%18 : tensor<32x384xi32>) -> tensor<32x384xi32>
      %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%20, %c-2139095040_i32 : tensor<32x384xi32>, i32) outs(%18 : tensor<32x384xi32>) -> tensor<32x384xi32>
      %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%21, %c1_i32 : tensor<32x384xi32>, i32) outs(%21 : tensor<32x384xi32>) -> tensor<32x384xi32>
      %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%22, %c0_i32 : tensor<32x384xi32>, i32) outs(%22 : tensor<32x384xi32>) -> tensor<32x384xi32>
      %24 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%23 : tensor<32x384xi32>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %25 = tensor.empty() : tensor<32x384xi1>
      %26 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%24, %cst : tensor<32x384xf32>, f32) outs(%25 : tensor<32x384xi1>) -> tensor<32x384xi1>
      %27 = hfusion.select ins(%26, %cst_0, %15 : tensor<32x384xi1>, f32, tensor<32x384xf32>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %reduced = linalg.reduce ins(%27 : tensor<32x384xf32>) outs(%17 : tensor<32xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %34 = arith.maximumf %in, %init : f32
          linalg.yield %34 : f32
        }
      %broadcasted = linalg.broadcast ins(%reduced : tensor<32xf32>) outs(%14 : tensor<32x384xf32>) dimensions = [1] 
      %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %broadcasted : tensor<32x384xf32>, tensor<32x384xf32>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %29 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%28 : tensor<32x384xf32>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %30 = linalg.fill ins(%cst : f32) outs(%16 : tensor<32xf32>) -> tensor<32xf32>
      %reduced_3 = linalg.reduce ins(%29 : tensor<32x384xf32>) outs(%30 : tensor<32xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %34 = arith.addf %in, %init : f32
          linalg.yield %34 : f32
        }
      %broadcasted_4 = linalg.broadcast ins(%reduced_3 : tensor<32xf32>) outs(%14 : tensor<32x384xf32>) dimensions = [1] 
      %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%29, %broadcasted_4 : tensor<32x384xf32>, tensor<32x384xf32>) outs(%14 : tensor<32x384xf32>) -> tensor<32x384xf32>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [32, 384], strides: [384, 1] : memref<?xf16> to memref<32x384xf16, strided<[384, 1], offset: ?>>
      %32 = tensor.empty() : tensor<32x384xf16>
      %33 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%31 : tensor<32x384xf32>) outs(%32 : tensor<32x384xf16>) -> tensor<32x384xf16>
      %extracted_slice = tensor.extract_slice %33[0, 0] [%11, 384] [1, 1] : tensor<32x384xf16> to tensor<?x384xf16>
      %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%11, 384] [1, 1] : memref<32x384xf16, strided<[384, 1], offset: ?>> to memref<?x384xf16, strided<[384, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?x384xf16>, memref<?x384xf16, strided<[384, 1], offset: ?>>) -> ()
    }
    return
  }
}