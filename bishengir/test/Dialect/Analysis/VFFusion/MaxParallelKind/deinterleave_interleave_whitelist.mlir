// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend910_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s
//
//==============================================================================
// TEST SPECIFICATION: DeinterleaveOp / InterleaveOp Fusion Whitelist
//==============================================================================
//
// PURPOSE: Verify that hfusion::DeinterleaveOp is recognized by
// isInFusionWhiteList in MaxParallelAnalyzer, so it can be fused with
// its consumer LinalgOps during VFFusion outlining.
//
// BEFORE the fix (48c6db566): DeinterleaveOp was excluded, causing it to
// remain in the caller while compute ops were outlined separately.
// AFTER the fix: DeinterleaveOp is fused with consumers into one VF.
//
// COVERAGE:
//   - test_deinterleave_fusion: DeinterleaveOp + linalg.elemwise fused together
//==============================================================================

// TEST_SPEC: deinterleave_fusion | DeinterleaveOp feeds linalg.elemwise_binary
//   Post-fix: deinterleave + mul should appear in the same _fused_ function.
// CHECK-LABEL: func.func private @triton_unk_fused_test_deinterleave_fusion_fused_0(
// CHECK: hfusion.deinterleave
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// CHECK: return

// CHECK-LABEL: func.func @triton_unk_fused_test_deinterleave_fusion(
// CHECK: func.call @triton_unk_fused_test_deinterleave_fusion_fused_0
// CHECK: return

module {
  func.func @triton_unk_fused_test_deinterleave_fusion(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 1.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4096x1xf16>
    %1 = arith.muli %arg8, %c1048576_i32 : i32
    %2 = arith.addi %1, %c8192_i32 : i32
    %3 = arith.minsi %2, %arg4 : i32
    scf.for %arg11 = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %4 = arith.muli %arg11, %c8192_i32 : i32
      %5 = arith.addi %1, %4 : i32
      %6 = arith.index_cast %5 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [8192], strides: [1] : memref<?xf16> to memref<8192xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<8192xf16>
      %7 = arith.addi %6, %c4096 : index
      %8 = arith.index_cast %3 : i32 to index
      %9 = arith.maxsi %6, %8 : index
      %10 = arith.minsi %7, %9 : index
      %11 = arith.subi %10, %6 : index
      %12 = arith.cmpi slt, %11, %c4096 : index
      scf.if %12 {
        linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<8192xf16>)
      } {hivm.unlikely_condition}
      %subview = memref.subview %reinterpret_cast[0] [%11] [1] : memref<8192xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %subview_2 = memref.subview %alloc[0] [%11] [1] : memref<8192xf16> to memref<?xf16>
      memref.copy %subview, %subview_2 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %13 = bufferization.to_tensor %alloc restrict writable : memref<8192xf16>
      %15 = tensor.expand_shape %13 [[0, 1]] output_shape [4096, 2] : tensor<8192xf16> into tensor<4096x2xf16>
      %16 = hfusion.deinterleave %15 channel<0> : tensor<4096x2xf16> -> tensor<4096x1xf16>
      %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %cst : tensor<4096x1xf16>, f16) outs(%0 : tensor<4096x1xf16>) -> tensor<4096x1xf16>
      %18 = tensor.collapse_shape %17 [[0, 1]] : tensor<4096x1xf16> into tensor<4096xf16>
      %extracted_slice = tensor.extract_slice %18[0] [%11] [1] : tensor<4096xf16> to tensor<?xf16>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%6], sizes: [4096], strides: [1] : memref<?xf16> to memref<4096xf16, strided<[1], offset: ?>>
      %subview_4 = memref.subview %reinterpret_cast_3[0] [%11] [1] : memref<4096xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_4 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
