// RUN: bishengir-opt -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s

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
