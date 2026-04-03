// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_0(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.broadcast
// CHECK: linalg.fill
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.fill
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<log>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<trunc>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
// CHECK: hfusion.select
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_1(
// CHECK: linalg.broadcast
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<rint>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_2(
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK: hfusion.bitcast
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>}
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_3(
// CHECK: hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>}
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
// CHECK: hfusion.select
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: linalg.yield
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_4(
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK: linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_5(
// CHECK: linalg.broadcast
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: tensor.extract_slice
// CHECK: return

// CHECK-LABEL: func.func private @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_6(
// CHECK: hfusion.arange
// CHECK: linalg.broadcast
// CHECK: return

// CHECK-LABEL: func.func @triton_unk_fused__softmax_add_mul_rsub_3_01B(
// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: arith.muli
// CHECK: hfusion.arange
// CHECK: call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_6
// CHECK: scf.for
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: arith.cmpi slt
// CHECK: scf.if
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: memref.copy
// CHECK: bufferization.to_tensor
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_0
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: memref.load
// CHECK: tensor.insert
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_1
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_2
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_3
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_4
// CHECK: func.call @triton_unk_fused__softmax_add_mul_rsub_3_01B_fused_5
// CHECK: bufferization.materialize_in_destination
// CHECK: return

module{
   func.func @triton_unk_fused__softmax_add_mul_rsub_3_01B(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant -1.000000e+00 : f32
    %c-2139095040_i32 = arith.constant -2139095040 : i32
    %c2147483647_i32 = arith.constant 2147483647 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4194304_i32 = arith.constant 4194304 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c16_i64 = arith.constant 16 : i64
    %cst_2 = arith.constant 6.250000e-02 : f32
    %cst_3 = arith.constant 0.48089835 : f32
    %cst_4 = arith.constant 1.600000e+01 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant -3.40282347E+38 : f32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i64 = arith.constant 8 : i64
    %c2048 = arith.constant 2048 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c31_i64 = arith.constant 31 : i64
    %0 = tensor.empty() : tensor<2x2048xi64>
    %1 = tensor.empty() : tensor<2x2048xf32>
    %2 = tensor.empty() : tensor<2x2048xi32>
    %3 = tensor.empty() : tensor<2xi32>
    %4 = arith.muli %arg11, %c512_i32 : i32
    %5 = hfusion.arange offset[%c0] strides[%c1] outs(%3 : tensor<2xi32>) -> tensor<2xi32>
    %6 = tensor.empty() : tensor<2048xi32>
    %7 = hfusion.arange offset[%c0] strides[%c1] outs(%6 : tensor<2048xi32>) -> tensor<2048xi32>
    %8 = arith.addi %arg10, %c1_i32 : i32
    %9 = arith.minsi %8, %arg4 : i32
    %10 = arith.addi %4, %c512_i32 : i32
    %11 = arith.minsi %10, %arg5 : i32
    %broadcasted = linalg.broadcast ins(%7 : tensor<2048xi32>) outs(%2 : tensor<2x2048xi32>) dimensions = [0] 
    scf.for %arg13 = %arg10 to %9 step %c1_i32  : i32 {
      %12 = arith.muli %arg13, %c4194304_i32 : i32
      %13 = arith.extsi %arg13 : i32 to i64
      scf.for %arg14 = %c0_i32 to %c256_i32 step %c1_i32  : i32 {
        %14 = arith.muli %arg14, %c2_i32 : i32
        %15 = arith.addi %4, %14 : i32
        %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %5 : i32, tensor<2xi32>) outs(%3 : tensor<2xi32>) -> tensor<2xi32>
        %17 = arith.index_cast %15 : i32 to index
        %18 = arith.muli %17, %c2048 : index
        %19 = arith.index_cast %12 : i32 to index
        %20 = arith.addi %18, %19 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [2, 2048], strides: [2048, 1] : memref<?xf32> to memref<2x2048xf32, strided<[2048, 1], offset: ?>>
        %alloc = memref.alloc() : memref<2x2048xf32>
        %21 = arith.index_cast %arg6 : i32 to index
        %22 = arith.maxsi %21, %c0 : index
        %23 = arith.minsi %22, %c2048 : index
        %24 = arith.addi %17, %c2 : index
        %25 = arith.index_cast %11 : i32 to index
        %26 = arith.maxsi %17, %25 : index
        %27 = arith.minsi %24, %26 : index
        %28 = arith.subi %27, %17 : index
        %29 = arith.minsi %28, %c2 : index
        %30 = arith.minsi %23, %c2048 : index
        %31 = arith.cmpi slt, %29, %c2 : index
        %32 = arith.cmpi slt, %30, %c2048 : index
        %33 = arith.ori %31, %32 : i1
        scf.if %33 {
          linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<2x2048xf32>)
        } {hivm.unlikely_condition}
        %subview = memref.subview %reinterpret_cast[0, 0] [%29, %30] [1, 1] : memref<2x2048xf32, strided<[2048, 1], offset: ?>> to memref<?x?xf32, strided<[2048, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, 0] [%29, %30] [1, 1] : memref<2x2048xf32> to memref<?x?xf32, strided<[2048, 1]>>
        memref.copy %subview, %subview_7 : memref<?x?xf32, strided<[2048, 1], offset: ?>> to memref<?x?xf32, strided<[2048, 1]>>
        %34 = bufferization.to_tensor %alloc restrict writable : memref<2x2048xf32>
        %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %c-1_i32 : tensor<2xi32>, i32) outs(%3 : tensor<2xi32>) -> tensor<2xi32>
        %broadcasted_8 = linalg.broadcast ins(%35 : tensor<2xi32>) outs(%2 : tensor<2x2048xi32>) dimensions = [1] 
        %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted, %broadcasted_8 : tensor<2x2048xi32>, tensor<2x2048xi32>) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %37 = tensor.empty() : tensor<2x2048xi1>
        %38 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %39 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%36, %38 : tensor<2x2048xi32>, tensor<2x2048xi32>) outs(%37 : tensor<2x2048xi1>) -> tensor<2x2048xi1>
        %40 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%39 : tensor<2x2048xi1>) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%36, %40 : tensor<2x2048xi32>, tensor<2x2048xi32>) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%41, %c-1_i32 : tensor<2x2048xi32>, i32) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %43 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%42 : tensor<2x2048xi32>) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %44 = linalg.fill ins(%c16_i64 : i64) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %45 = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>} ins(%43, %44 : tensor<2x2048xi64>, tensor<2x2048xi64>) outs(%37 : tensor<2x2048xi1>) -> tensor<2x2048xi1>
        %46 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%42 : tensor<2x2048xi32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%46, %cst_2 : tensor<2x2048xf32>, f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %48 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%47 : tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %49 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%48, %cst_3 : tensor<2x2048xf32>, f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %50 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%49, %cst_4 : tensor<2x2048xf32>, f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %51 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<trunc>} ins(%50 : tensor<2x2048xf32>) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %52 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%51, %c16_i64 : tensor<2x2048xi64>, i64) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %53 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%52, %c31_i64 : tensor<2x2048xi64>, i64) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %54 = hfusion.select ins(%45, %43, %53 : tensor<2x2048xi1>, tensor<2x2048xi64>, tensor<2x2048xi64>) outs(%0 : tensor<2x2048xi64>) -> tensor<2x2048xi64>
        %55 = scf.for %arg15 = %c0 to %c2 step %c1 iter_args(%arg16 = %1) -> (tensor<2x2048xf32>) {
          %78 = scf.for %arg17 = %c0 to %c2048 step %c1 iter_args(%arg18 = %arg16) -> (tensor<2x2048xf32>) {
            %extracted = tensor.extract %54[%arg15, %arg17] {DiscreteMemAccess} : tensor<2x2048xi64>
            %79 = arith.muli %extracted, %c8_i64 : i64
            %80 = arith.addi %13, %79 : i64
            %81 = arith.index_cast %80 : i64 to index
            %reinterpret_cast_14 = memref.reinterpret_cast %arg3 to offset: [%81], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %82 = memref.load %reinterpret_cast_14[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %inserted = tensor.insert %82 into %arg18[%arg15, %arg17] : tensor<2x2048xf32>
            scf.yield {DiscreteMemAccess} %inserted : tensor<2x2048xf32>
          } {ExtractedLoadOrStore}
          scf.yield %78 : tensor<2x2048xf32>
        } {ExtractedLoadOrStore}
        %extracted_slice = tensor.extract_slice %55[0, 0] [%29, %30] [1, 1] : tensor<2x2048xf32> to tensor<?x?xf32>
        %56 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice into %56[0, 0] [%29, %30] [1, 1] : tensor<?x?xf32> into tensor<2x2048xf32>
        %broadcasted_9 = linalg.broadcast ins(%16 : tensor<2xi32>) outs(%2 : tensor<2x2048xi32>) dimensions = [1] 
        %57 = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>} ins(%broadcasted, %broadcasted_9 : tensor<2x2048xi32>, tensor<2x2048xi32>) outs(%37 : tensor<2x2048xi1>) -> tensor<2x2048xi1>
        %58 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, round_mode = #hfusion.round_mode<rint>} ins(%57 : tensor<2x2048xi1>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %59 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%58, %cst : tensor<2x2048xf32>, f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %60 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst_5, %59 : f32, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %61 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%60, %cst_6 : tensor<2x2048xf32>, f32) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %62 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%inserted_slice, %61 : tensor<2x2048xf32>, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %63 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%34, %62 : tensor<2x2048xf32>, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %64 = tensor.empty() : tensor<2xf32>
        %65 = linalg.fill ins(%cst_0 : f32) outs(%64 : tensor<2xf32>) -> tensor<2xf32>
        %66 = hfusion.bitcast ins(%63 : tensor<2x2048xf32>) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %67 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%66, %c2147483647_i32 : tensor<2x2048xi32>, i32) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %68 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%67, %c-2139095040_i32 : tensor<2x2048xi32>, i32) outs(%2 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %69 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%68, %c1_i32 : tensor<2x2048xi32>, i32) outs(%68 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %70 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%69, %c0_i32 : tensor<2x2048xi32>, i32) outs(%69 : tensor<2x2048xi32>) -> tensor<2x2048xi32>
        %71 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, round_mode = #hfusion.round_mode<rint>} ins(%70 : tensor<2x2048xi32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %72 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%71, %cst_1 : tensor<2x2048xf32>, f32) outs(%37 : tensor<2x2048xi1>) -> tensor<2x2048xi1>
        %73 = hfusion.select ins(%72, %cst_0, %63 : tensor<2x2048xi1>, f32, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %reduced = linalg.reduce ins(%73 : tensor<2x2048xf32>) outs(%65 : tensor<2xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %78 = arith.maximumf %in, %init : f32
            linalg.yield %78 : f32
          }
        %broadcasted_10 = linalg.broadcast ins(%reduced : tensor<2xf32>) outs(%1 : tensor<2x2048xf32>) dimensions = [1] 
        %74 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%63, %broadcasted_10 : tensor<2x2048xf32>, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %75 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%74 : tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %76 = linalg.fill ins(%cst_1 : f32) outs(%64 : tensor<2xf32>) -> tensor<2xf32>
        %reduced_11 = linalg.reduce ins(%75 : tensor<2x2048xf32>) outs(%76 : tensor<2xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %78 = arith.addf %in, %init : f32
            linalg.yield %78 : f32
          }
        %broadcasted_12 = linalg.broadcast ins(%reduced_11 : tensor<2xf32>) outs(%1 : tensor<2x2048xf32>) dimensions = [1] 
        %77 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%75, %broadcasted_12 : tensor<2x2048xf32>, tensor<2x2048xf32>) outs(%1 : tensor<2x2048xf32>) -> tensor<2x2048xf32>
        %extracted_slice_13 = tensor.extract_slice %77[0, 0] [%29, %30] [1, 1] : tensor<2x2048xf32> to tensor<?x?xf32>
        bufferization.materialize_in_destination %extracted_slice_13 in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[2048, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}