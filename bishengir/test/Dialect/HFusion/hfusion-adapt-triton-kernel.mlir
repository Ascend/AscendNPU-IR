// RUN: bishengir-opt -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -pass-pipeline='builtin.module(adapt-triton-kernel,adapt-triton-kernel)' | FileCheck %s
// RUN: bishengir-opt %s -split-input-file -adapt-triton-kernel -mlir-print-op-generic | FileCheck %s --check-prefix=SCALAR
// RUN: bishengir-opt %s -split-input-file -adapt-triton-kernel --hfusion-normalize-ops | FileCheck %s --check-prefix=NORMALIZE

// NORMALIZE-LABEL: @rtz_broadcast_normalized
// NORMALIZE: %[[NORMAL:.*]] = hfusion.cast {{.*}}enable_saturate = true
// NORMALIZE-SAME: round_mode = #hfusion.round_mode<trunc>
// NORMALIZE-SAME: ins(%arg0 : tensor<16xf32>)
// NORMALIZE-SAME: -> tensor<16xf16>
// NORMALIZE: %[[BRC:.*]] = linalg.broadcast ins(%[[NORMAL]] : tensor<16xf16>)
// NORMALIZE: hfusion.select ins({{.*}}, %[[BRC]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @rtz_broadcast_normalized(%src: tensor<16xf32>) -> tensor<16x16xf16> {
    %empty = tensor.empty() : tensor<16x16xf32>
    %brc = linalg.broadcast ins(%src : tensor<16xf32>) outs(%empty : tensor<16x16xf32>) dimensions = [0]
    %init = tensor.empty() : tensor<16x16xf16>
    %r = hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
        ins(%brc : tensor<16x16xf32>) outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %r : tensor<16x16xf16>
  }
}

// -----

// NORMALIZE-LABEL: @rtz_fill_normalized
// NORMALIZE: %[[NORMAL:.*]] = hfusion.cast {{.*}}enable_saturate = true
// NORMALIZE-SAME: round_mode = #hfusion.round_mode<trunc>
// NORMALIZE-SAME: -> tensor<1xf16>
// NORMALIZE: %[[EXTRACT:.*]] = tensor.extract %[[NORMAL]]
// NORMALIZE: %[[POINT:.*]] = tensor.insert %[[EXTRACT]] into {{.*}} : tensor<f16>
// NORMALIZE: %[[ELEM:.*]] = tensor.extract %[[POINT]][] : tensor<f16>
// NORMALIZE: %[[BRC:.*]] = linalg.fill ins(%[[ELEM]] : f16)
// NORMALIZE: hfusion.select ins({{.*}}, %[[BRC]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @rtz_fill_normalized(%src: f32) -> tensor<16xf16> {
    %empty = tensor.empty() : tensor<16xf32>
    %fill = linalg.fill ins(%src : f32) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    %init = tensor.empty() : tensor<16xf16>
    %r = hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
        ins(%fill : tensor<16xf32>) outs(%init : tensor<16xf16>) -> tensor<16xf16>
    return %r : tensor<16xf16>
  }
}

// -----

// CHECK-LABEL: @rtz_f16
// CHECK: %[[FINITE:.*]] = hfusion.isfinite %arg0
// CHECK: %[[SPECIAL:.*]] = hfusion.cast
// CHECK-NOT: enable_saturate = true
// CHECK-SAME: round_mode = #hfusion.round_mode<rint>
// CHECK: %[[CAST:.*]] = hfusion.cast
// CHECK-SAME: enable_saturate = true
// CHECK-SAME: hfusion.rtz_processed
// CHECK-SAME: round_mode = #hfusion.round_mode<trunc>
// CHECK: hfusion.select ins(%[[FINITE]], %[[CAST]], %[[SPECIAL]]
// CHECK-NOT: hfusion.select
// CHECK: return
// SCALAR: "arith.truncf"
// SCALAR-NOT: enable_saturate = true
// SCALAR-SAME: round_mode = #hfusion.round_mode<rint>
// SCALAR: "arith.truncf"
// SCALAR-SAME: enable_saturate = true
// SCALAR-SAME: round_mode = #hfusion.round_mode<trunc>
func.func @rtz_f16(%src: tensor<16xf32>, %init: tensor<16xf16>) -> tensor<16xf16> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<16xf32>) outs(%init : tensor<16xf16>) -> tensor<16xf16>
  return %0 : tensor<16xf16>
}

// CHECK-LABEL: @rtz_bf16
// CHECK: %[[FINITE:.*]] = hfusion.isfinite %arg0
// CHECK: %[[SPECIAL:.*]] = hfusion.cast
// CHECK-NOT: enable_saturate = true
// CHECK-SAME: round_mode = #hfusion.round_mode<rint>
// CHECK: %[[CAST:.*]] = hfusion.cast
// CHECK-SAME: enable_saturate = true
// CHECK: hfusion.select ins(%[[FINITE]], %[[CAST]], %[[SPECIAL]]
func.func @rtz_bf16(%src: tensor<?xf32>, %init: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<?xf32>) outs(%init : tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}

// CHECK-LABEL: @rtz_fp8
// CHECK-NOT: hfusion.cast
// CHECK-NOT: hfusion.isfinite
// CHECK: %[[NAN:.*]] = arith.constant 0x7F : f8E4M3FN
// CHECK-NEXT: %[[ISNAN:.*]] = hfusion.isnan %arg0 : tensor<16xf16> -> tensor<16xi1>
// CHECK-NEXT: %[[CAST:.*]] = hfusion.cast
// CHECK-SAME: enable_saturate = true
// CHECK-SAME: hfusion.rtz_processed
// CHECK-SAME: round_mode = #hfusion.round_mode<trunc>
// CHECK-NEXT: %[[RESULT:.*]] = hfusion.select ins(%[[ISNAN]], %[[NAN]], %[[CAST]]
// CHECK-NEXT: return %[[RESULT]]
func.func @rtz_fp8(%src: tensor<16xf16>, %init: tensor<16xf8E4M3FN>) -> tensor<16xf8E4M3FN> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<16xf16>) outs(%init : tensor<16xf8E4M3FN>) -> tensor<16xf8E4M3FN>
  return %0 : tensor<16xf8E4M3FN>
}

// CHECK-LABEL: @rtz_fp8_e5m2
// CHECK: %[[FINITE:.*]] = hfusion.isfinite %arg0
// CHECK: %[[SPECIAL:.*]] = hfusion.cast
// CHECK-NOT: enable_saturate = true
// CHECK-SAME: round_mode = #hfusion.round_mode<rint>
// CHECK: %[[CAST:.*]] = hfusion.cast
// CHECK-SAME: enable_saturate = true
// CHECK: hfusion.select ins(%[[FINITE]], %[[CAST]], %[[SPECIAL]]
func.func @rtz_fp8_e5m2(%src: tensor<16xf32>, %init: tensor<16xf8E5M2>) -> tensor<16xf8E5M2> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<16xf32>) outs(%init : tensor<16xf8E5M2>) -> tensor<16xf8E5M2>
  return %0 : tensor<16xf8E5M2>
}

// CHECK-LABEL: @rtz_upcast_unchanged
// CHECK-NOT: hfusion.compare
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<trunc>}
// CHECK-NOT: hfusion.select
// CHECK: return
func.func @rtz_upcast_unchanged(%src: tensor<16xf16>, %init: tensor<16xf32>) -> tensor<16xf32> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<16xf16>) outs(%init : tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: @rint_unchanged
// CHECK-NOT: hfusion.compare
// CHECK: hfusion.cast {round_mode = #hfusion.round_mode<rint>}
// CHECK-NOT: hfusion.select
// CHECK: return
func.func @rint_unchanged(%src: tensor<16xf32>, %init: tensor<16xf16>) -> tensor<16xf16> {
  %0 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%src : tensor<16xf32>) outs(%init : tensor<16xf16>) -> tensor<16xf16>
  return %0 : tensor<16xf16>
}

// -----

// CHECK-LABEL: module
// CHECK-LABEL: @__hmf_atanf
// CHECK-NOT: hacc.entry
// CHECK-NOT: hacc.function_kind
// CHECK-SAME: -> f32
// CHECK: func.func @triton_elementwise_unary
// CHECK-SAME: hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>
module {
func.func private @__hmf_atanf(f32) -> f32
func.func @triton_elementwise_unary(%arg0: i32) attributes {global_kernel = "local", mix_mode = "aiv"} {
  return
}
}

// -----

module {
  // CHECK-LABEL: func.func @test_mark_wrokspace_arg
  // CHECK-SAME: %[[ARG:.*]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
  func.func @test_mark_wrokspace_arg(%arg0: memref<?xi8>, %arg1: i32) attributes {global_kernel = "", WorkspaceArgIdx = 0 : i64} {
    return
  }
}

// -----
// CHECK-LABEL: @triton_kernel_printop
// CHECK-SAME: (%[[PID_32:.*]]: i32,
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor
// CHECK-NEXT: hfusion.print "PID: " {hex = true} %[[PID_32]] : i32
// CHECK-NEXT: hfusion.print "" {hex = false} %[[TENSOR]] : tensor<1024xf32>
// CHECK-NEXT: call @a_triton_print_1
func.func private @triton_print(i32) attributes {hex = true, prefix = "PID: "}
func.func private @triton_print_0(tensor<1024xf32>) attributes {hex = false, prefix = ""}
func.func private @a_triton_print_1(tensor<1024xf32>) attributes {hex = false, prefix = "Val: "}
func.func @triton_kernel_printop(%pid: i32, %arg0: memref<1024xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<1024xf32>
  func.call @triton_print(%pid) : (i32) -> ()
  func.call @triton_print_0(%0) : (tensor<1024xf32>) -> ()
  func.call @a_triton_print_1(%0) : (tensor<1024xf32>) -> ()
  return
}

// -----
// CHECK-LABEL: @triton_kernel_gatherop
// CHECK: %[[TENSOR0:.*]] = bufferization.to_tensor %arg0
// CHECK: %[[TENSOR1:.*]] = bufferization.to_tensor %arg1
// CHECK: hfusion.gather
// CHECK-SAME: ins(%[[TENSOR0]], %[[TENSOR1]] : tensor<4x64xf32>, tensor<4x32xi32>) outs(%[[INIT2:.*]] : tensor<4x32xf32>) axis = 1 -> tensor<4x32xf32>
// CHECK: %[[TENSOR2:.*]] = bufferization.to_tensor %arg2
// CHECK: hfusion.gather
// CHECK-SAME: ins(%[[TENSOR0]], %[[TENSOR2]] : tensor<4x64xf32>, tensor<4x32xi32>) outs(%[[INIT5:.*]] : tensor<4x32xf32>) axis = 1 -> tensor<4x32xf32>
// CHECK: call @a_triton_gather_1
func.func private @triton_gather_0(tensor<4x64xf32>, tensor<4x32xi32>, i32) -> tensor<4x32xf32>
func.func private @triton_gather_1(tensor<4x64xf32>, tensor<4x32xi32>, i32) -> tensor<4x32xf32>
func.func private @triton_print_2(tensor<4x32xf32>) attributes {hex = false, prefix = ""}
func.func private @triton_print_3(tensor<4x32xf32>) attributes {hex = false, prefix = ""}
func.func private @a_triton_gather_1(tensor<4x64xf32>)
func.func @triton_kernel_gatherop(%arg0: memref<4x64xf32>, %arg1: memref<4x32xi32>, %arg2: memref<4x32xi32>) {
  %c1_i32 = arith.constant 1 : i32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x64xf32>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4x32xi32>
  %2 = call @triton_gather_0(%0, %1, %c1_i32) : (tensor<4x64xf32>, tensor<4x32xi32>, i32) -> tensor<4x32xf32>
  %3 = bufferization.to_tensor %arg2 restrict writable : memref<4x32xi32>
  %4 = call @triton_gather_1(%0, %3, %c1_i32) : (tensor<4x64xf32>, tensor<4x32xi32>, i32) -> tensor<4x32xf32>
  func.call @a_triton_gather_1(%0) : (tensor<4x64xf32>) -> ()
  func.call @triton_print_2(%2) : (tensor<4x32xf32>) -> ()
  func.call @triton_print_3(%4) : (tensor<4x32xf32>) -> ()
  return
}

// -----
// CHECK-LABEL: @triton_kernel_cumsumop
// CHECK: hfusion.cumsum
// CHECK: hfusion.cumsum
// CHECK: hfusion.cumsum
func.func private @triton_cumsum_0(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func private @triton_cumsum_1(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func private @triton_cumsum_2(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func @triton_kernel_cumsumop(%arg0: memref<4x6x8xf32>) -> (tensor<4x6x8xf32>, tensor<4x6x8xf32>, tensor<4x6x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xf32>
  %1 = call @triton_cumsum_0(%0, %c0_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  %2 = call @triton_cumsum_1(%0, %c1_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  %3 = call @triton_cumsum_2(%0, %c2_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  return %1, %2, %3 : tensor<4x6x8xf32>, tensor<4x6x8xf32>, tensor<4x6x8xf32>
}

// -----
// CHECK-LABEL: @triton_kernel_cumprodop
// CHECK: hfusion.cumprod
// CHECK: hfusion.cumprod
// CHECK: hfusion.cumprod
func.func private @triton_cumprod_0(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func private @triton_cumprod_1(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func private @triton_cumprod_2(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func @triton_kernel_cumprodop(%arg0: memref<4x6x8xi16>) -> (tensor<4x6x8xi16>, tensor<4x6x8xi16>, tensor<4x6x8xi16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xi16>
  %1 = call @triton_cumprod_0(%0, %c0_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  %2 = call @triton_cumprod_1(%0, %c1_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  %3 = call @triton_cumprod_2(%0, %c2_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  return %1, %2, %3 : tensor<4x6x8xi16>, tensor<4x6x8xi16>, tensor<4x6x8xi16>
}

// -----
func.func @matmul_mix_aiv_with_bind_sub_block_tag(%arg0: tensor<64x64xf32>, %arg1: memref<32x64xf16>) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<14xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.00999999977 : f32
  %c2 = arith.constant 2 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %0 = tensor.empty() : tensor<32x64xf32>
  scf.for %arg2 = %c0 to %c2 step %c1 : i32{
    // CHECK: arith.index_cast
    %2 = arith.muli %arg2, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %extracted_slice = tensor.extract_slice %arg0[%3, 0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
    %4 = hivm.hir.vadd ins(%extracted_slice, %cst : tensor<32x64xf32>, f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %5 = tensor.empty() : tensor<32x64xf16>
    %6 = hivm.hir.vcast ins(%4 : tensor<32x64xf32>) outs(%5 : tensor<32x64xf16>) -> tensor<32x64xf16>
    bufferization.materialize_in_destination %6 in writable %arg1 : (tensor<32x64xf16>, memref<32x64xf16>) -> ()
  } {bind_sub_block = true}
  // CHECK: } {hfusion.bind_sub_block}
  return
}

// -----
// CHECK-LABEL: @sort_kernel_2d
// CHECK: hfusion.sort
func.func private @triton_sort(tensor<512x8xf32>, i64, i1) -> tensor<512x8xf32>
func.func @sort_kernel_2d(%arg0: memref<512x8xf32>) -> (tensor<512x8xf32>) {
  %true = arith.constant true
  %c1_i64 = arith.constant 1 : i64
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<512x8xf32>
  %1 = call @triton_sort(%0, %c1_i64, %true) : (tensor<512x8xf32>, i64, i1) -> tensor<512x8xf32>
  return %1 : tensor<512x8xf32>
}

// -----
// CHECK-LABEL: @flip_kernel_2d
// CHECK: hfusion.flip
func.func private @triton_flip(tensor<512x8xf32>, i64) -> tensor<512x8xf32>
func.func @flip_kernel_2d(%arg0: memref<512x8xf32>) -> (tensor<512x8xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<512x8xf32>
  %1 = call @triton_flip(%0, %c1_i64) : (tensor<512x8xf32>, i64) -> tensor<512x8xf32>
  return %1 : tensor<512x8xf32>
}

// -----

// CHECK-LABEL: @triton_kernel_cummaxop
// CHECK: hfusion.cummax
// CHECK: hfusion.cummax
// CHECK: hfusion.cummax
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_cummax_0(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func private @triton_cummax_1(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func private @triton_cummax_2(tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
func.func @triton_kernel_cummaxop(%arg0: memref<4x6x8xf32>) -> (tensor<4x6x8xf32>, tensor<4x6x8xf32>, tensor<4x6x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xf32>
  %1 = call @triton_cummax_0(%0, %c0_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  %2 = call @triton_cummax_1(%0, %c1_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  %3 = call @triton_cummax_2(%0, %c2_i32, %c0_i32) : (tensor<4x6x8xf32>, i32, i32) -> tensor<4x6x8xf32>
  return %1, %2, %3 : tensor<4x6x8xf32>, tensor<4x6x8xf32>, tensor<4x6x8xf32>
}
}

// -----

// CHECK-LABEL: @triton_kernel_cumminop
// CHECK: hfusion.cummin
// CHECK: hfusion.cummin
// CHECK: hfusion.cummin
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_cummin_0(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func private @triton_cummin_1(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func private @triton_cummin_2(tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
func.func @triton_kernel_cumminop(%arg0: memref<4x6x8xi16>) -> (tensor<4x6x8xi16>, tensor<4x6x8xi16>, tensor<4x6x8xi16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xi16>
  %1 = call @triton_cummin_0(%0, %c0_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  %2 = call @triton_cummin_1(%0, %c1_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  %3 = call @triton_cummin_2(%0, %c2_i32, %c0_i32) : (tensor<4x6x8xi16>, i32, i32) -> tensor<4x6x8xi16>
  return %1, %2, %3 : tensor<4x6x8xi16>, tensor<4x6x8xi16>, tensor<4x6x8xi16>
}
}

// -----

// CHECK-LABEL: @triton_kernel_cummax_propagate_nan
// CHECK: hfusion.cummax %{{.*}} {propagate_nan = false}
// CHECK: hfusion.cummax %{{.*}} :
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_cummax_nonan_0(tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
func.func private @triton_cummax_propnan_0(tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
func.func @triton_kernel_cummax_propagate_nan(%arg0: memref<4x6x8xf32>) -> (tensor<4x6x8xf32>, tensor<4x6x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %false = arith.constant false
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xf32>
  %1 = call @triton_cummax_nonan_0(%0, %c0_i32, %c0_i32, %false) : (tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
  %2 = call @triton_cummax_propnan_0(%0, %c0_i32, %c0_i32, %true) : (tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
  return %1, %2 : tensor<4x6x8xf32>, tensor<4x6x8xf32>
}
}

// -----

// CHECK-LABEL: @triton_kernel_cummin_propagate_nan
// CHECK: hfusion.cummin %{{.*}} {propagate_nan = false}
// CHECK: hfusion.cummin %{{.*}} :
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_cummin_nonan_0(tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
func.func private @triton_cummin_propnan_0(tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
func.func @triton_kernel_cummin_propagate_nan(%arg0: memref<4x6x8xf32>) -> (tensor<4x6x8xf32>, tensor<4x6x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %false = arith.constant false
  %true = arith.constant true
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4x6x8xf32>
  %1 = call @triton_cummin_nonan_0(%0, %c0_i32, %c0_i32, %false) : (tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
  %2 = call @triton_cummin_propnan_0(%0, %c0_i32, %c0_i32, %true) : (tensor<4x6x8xf32>, i32, i32, i1) -> tensor<4x6x8xf32>
  return %1, %2 : tensor<4x6x8xf32>, tensor<4x6x8xf32>
}
}

// -----

// CHECK-LABEL: @triton_kernel_indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_indirect_load_0(memref<?xf32>, tensor<4x32xi32>) -> tensor<4x32xf32>
func.func private @triton_indirect_load_1(memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>) -> tensor<4x32xf32>
func.func private @triton_indirect_load_2(memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>, tensor<4x32xf32>) -> tensor<4x32xf32>
func.func private @triton_indirect_load_3(memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>, tensor<4x32xf32>) -> tensor<4x32xf32>
func.func private @triton_indirect_load_4(memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi1>, tensor<4x32xf32>) -> tensor<4x32xf32>
func.func @triton_kernel_indirect_load(%arg0: memref<?xf32>, %arg1: memref<4x32xi32>, %arg2: memref<4x32xi8>, %arg3: memref<4x32xi1>){
  %c0_f32 = arith.constant 0.0 : f32
  %c1_f32 = arith.constant 1.0 : f32
  %c2_i32 = arith.constant 0 : i32
  %b0 = tensor.empty() : tensor<4x32xf32>
  %t0_f32 = linalg.fill ins(%c0_f32 : f32) outs(%b0 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %b1 = tensor.empty() : tensor<4x32xf32>
  %t1_f32 = linalg.fill ins(%c1_f32 : f32) outs(%b1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %b2 = tensor.empty() : tensor<4x32xi32>
  %t2_i32 = linalg.fill ins(%c2_i32 : i32) outs(%b2 : tensor<4x32xi32>) -> tensor<4x32xi32>
  %0 = bufferization.to_tensor %arg1 restrict writable : memref<4x32xi32>
  %1 = bufferization.to_tensor %arg2 restrict writable : memref<4x32xi8>
  %bool_i1 = bufferization.to_tensor %arg3 restrict writable : memref<4x32xi1>
  %2 = call @triton_indirect_load_0(%arg0, %0) : (memref<?xf32>, tensor<4x32xi32>) -> tensor<4x32xf32>
  %3 = call @triton_indirect_load_1(%arg0, %0, %1) : (memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>) -> tensor<4x32xf32>
  %4 = call @triton_indirect_load_2(%arg0, %0, %1, %t0_f32) : (memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>, tensor<4x32xf32>) -> tensor<4x32xf32>
  %5 = call @triton_indirect_load_3(%arg0, %0, %1, %t1_f32) : (memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi8>, tensor<4x32xf32>) -> tensor<4x32xf32>
  %6 = call @triton_indirect_load_4(%arg0, %0, %bool_i1, %t1_f32) : (memref<?xf32>, tensor<4x32xi32>, tensor<4x32xi1>, tensor<4x32xf32>) -> tensor<4x32xf32>
  return
}
}

// -----

// CHECK-LABEL func.func @embeddinggather_test
// CHECK: DirectlyUsedGMArgIdxList = [0]
// CHECK: hfusion.embedding_gather
// CHECK-SAME: tensor<16x400x32xf32>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @triton_embedding_gather(memref<?xf32>, tensor<16x400xi32>, i32, i32, i32, i32, i32, i32, i32, i32) -> tensor<16x400x32xf32>
  func.func @embeddinggather_test(%arg0 : memref<?xf32>, %arg1 : tensor<16x400xi32>, %arg2 : tensor<16x400x32xf32>) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c4000 = arith.constant 4000 : i32
    %c9000 = arith.constant 9000 : i32
    %0 = func.call @triton_embedding_gather(%arg0, %arg1, %c9000, %c32, %c0, %c0, %c0, %c32, %c4000, %c32) : (memref<?xf32>, tensor<16x400xi32>, i32, i32, i32, i32, i32, i32, i32, i32) -> tensor<16x400x32xf32>
    func.return
  }
}

// -----

// CHECK-LABEL: @triton_kernel_indirect_store
// CHECK: hfusion.indirect_store
// CHECK: hfusion.indirect_store
// CHECK: hfusion.indirect_store
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_indirect_store_0(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>)
func.func private @triton_indirect_store_1(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi8>)
func.func private @triton_indirect_store_2(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi1>)
func.func @triton_kernel_indirect_store(%dst: memref<?xf32>, %offset: tensor<2x32xi32>, %src: tensor<2x32xf32>, %mask_bool: tensor<2x32xi1>, %mask_i8: tensor<2x32xi8>){
  call @triton_indirect_store_0(%dst, %offset, %src) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>) -> ()
  call @triton_indirect_store_1(%dst, %offset, %src, %mask_i8) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi8>) -> ()
  call @triton_indirect_store_2(%dst, %offset, %src, %mask_bool) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi1>) -> ()
  return
}
}

// -----

// CHECK-LABEL func.func @gather_out_to_ub_test
// CHECK: hfusion.gatherT
// CHECK-SAME: tensor<2x2xf32>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @triton__gather_out_to_ub(memref<?xf32>, tensor<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) -> tensor<2x2xf32>
  func.func @gather_out_to_ub_test(%arg0 : memref<?xf32>, %arg1 : tensor<2x2xi64>){
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = call @triton__gather_out_to_ub(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32) : (memref<?xf32>, tensor<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) -> tensor<2x2xf32>
    return
    }
}

// -----

// CHECK-LABEL func.func @index_put_test
// CHECK: hfusion.index_put
// CHECK-SAME: tensor<2x2xf32>
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @triton_index_put(memref<?xf32>, tensor<2xi64>, tensor<2x2xf32>, i32, i64, i64, i64, i64, i64, i64, i64)
  func.func @index_put_test(%arg0 : memref<?xf32>, %arg1 : tensor<2xi64>, %arg2 : tensor<2x2xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    call @triton_index_put(%arg0, %arg1, %arg2, %c0_i32, %c2_i64, %c4_i64, %c2_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64) : (memref<?xf32>, tensor<2xi64>, tensor<2x2xf32>, i32, i64, i64, i64, i64, i64, i64, i64) -> ()
    func.return
  }
}

// -----

// CHECK-LABEL: @scatterT_test
// CHECK: hfusion.scatterT
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_scatter_ub_to_out_0(memref<4x32xf32>, tensor<4x32xf32>, tensor<4x32xi32>, i32, i32, i32, i32, i32, i32, i32, i32)

func.func @scatterT_test(%arg0: memref<4x32xf32>, %arg1: tensor<4x32xf32>,%arg2: tensor<4x32xi32>) attributes {global_kernel = "local"} {
  %c0_i32  = arith.constant 0  : i32
  %c1_i32  = arith.constant 1  : i32
  %c4_i32  = arith.constant 4  : i32
  %c32_i32 = arith.constant 32 : i32

  call @triton_scatter_ub_to_out_0(%arg0, %arg1, %arg2, %c32_i32, %c1_i32, %c32_i32, %c1_i32, %c4_i32, %c32_i32, %c0_i32, %c0_i32) : (memref<4x32xf32>, tensor<4x32xf32>, tensor<4x32xi32>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  return
}
}

// -----

// CHECK-LABEL: @stride_load_test
// CHECK: hfusion.stride_load
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_stride_load_0(memref<?xf32>, i64, f32, i64, i64) -> tensor<32xf32>

func.func @stride_load_test(%arg0: memref<?xf32>) attributes {global_kernel = "local"} {
  %offset = arith.constant 4 : i64
  %other = arith.constant 0.000000e+00 : f32
  %stride = arith.constant 3 : i64
  %numel = arith.constant 32 : i64
  %0 = call @triton_stride_load_0(%arg0, %offset, %other, %stride, %numel) : (memref<?xf32>, i64, f32, i64, i64) -> tensor<32xf32>
  return
}
}

// -----

// CHECK-LABEL: @stride_store_test
// CHECK: hfusion.stride_store
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
func.func private @triton_stride_store_0(memref<?xf32>, tensor<32xf32>, i64, i64, i64)

func.func @stride_store_test(%arg0: memref<?xf32>, %arg1: tensor<32xf32>) attributes {global_kernel = "local"} {
  %offset = arith.constant 4 : i64
  %stride = arith.constant 3 : i64
  %numel = arith.constant 32 : i64
  call @triton_stride_store_0(%arg0, %arg1, %offset, %stride, %numel) : (memref<?xf32>, tensor<32xf32>, i64, i64, i64) -> ()
  return
}
}

// -----

// On membased arch (no target_device attr, default 910B), gm_addr_args_indices on
// a custom op must NOT propagate to hacc.arg_type<gm_addr>
// CHECK-LABEL: func.func @test_no_gm_addr_on_membased
// CHECK-SAME: %[[ARG:.*]]: memref<?xf32>
// CHECK-NOT: hacc.arg_type = #hacc.arg_type<gm_addr>
func.func @test_no_gm_addr_on_membased(%arg0 : memref<?xf32>, %out : memref<4xi32>) attributes {global_kernel = ""} {
  %c4_i64 = arith.constant 4 : i64
  hivm.hir.custom {gm_addr_args_indices = array<i32: 0>}
    "__builtin_histogram"
    ins(%arg0, %c4_i64 : memref<?xf32>, i64)
    outs(%out : memref<4xi32>)
  return
}
