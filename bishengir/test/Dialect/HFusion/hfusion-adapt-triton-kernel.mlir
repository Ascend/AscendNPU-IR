// RUN: bishengir-opt -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt --canonicalize -adapt-triton-kernel %s -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CHECK-CUSTOM

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

// CHECK-CUSTOM-LABEL: test_custom
// CHECK-CUSTOM: %arg5: memref<?xf32> {hacc.arg_type = #hacc.arg_type<gm_addr>}
func.func @test_custom(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xf32>, %arg3 : memref<?xi64>, %arg4 : memref<?xf32>, %arg5 : memref<?xf32>, %out : memref<2x2xf32>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %alloc = memref.alloc() : memref<2x2xi64>
  %gather_load = hivm.hir.custom "__builtin_gather_load" ins(%arg5, %alloc, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32 : memref<?xf32>, memref<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) outs(%out : memref<2x2xf32>) -> memref<2x2xf32>
  return
}

// -----

// CHECK-CUSTOM-LABEL: test_custom_macro
// CHECK-CUSTOM: %arg2: memref<?xf32> {hacc.arg_type = #hacc.arg_type<gm_addr>}
func.func @test_custom_macro(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xf32>, %arg3 : memref<?xi64>, %arg4 : memref<?xf32>, %out : memref<2x2xf32>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %alloc = memref.alloc() : memref<2x2xi64>
  %gather_load = hivm.hir.custom_macro "__builtin_gather_load" ins(%arg2, %alloc, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32 : memref<?xf32>, memref<2x2xi64>, i64, i32, i64, i64, i32, i32, i32, i32) outs(%out : memref<2x2xf32>) -> memref<2x2xf32>
  return
}

// -----

func.func private @triton_flip(tensor<4x8x8xi32>, i32) -> tensor<4x8x8xi32>
func.func @test_flip(%arg0 : tensor<4x8x8xi32>) -> tensor<4x8x8xi32> {
  %alloc = memref.alloc() : memref<4x8x8xi32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<4x8x8xi32>
  %2 = arith.constant 2 : i32
  // CHECK:       %[[RET:.*]] = hfusion.flip
  %ret = call @triton_flip(%arg0, %2) : (tensor<4x8x8xi32>, i32) -> tensor<4x8x8xi32>
  return %ret : tensor<4x8x8xi32>
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
// CHECK-LABEL: @triton_kernel_indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
// CHECK: hfusion.indirect_load
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
module {
  // CHECK-LABEL func.func @embeddinggather_test
  // CHECK: DirectlyUsedGMArgIdxList = [0]
  // CHECK: hfusion.embedding_gather
  // CHECK-SAME: tensor<16x400x32xf32>
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
func.func private @triton_indirect_store_0(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>)
func.func private @triton_indirect_store_1(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi8>)
func.func private @triton_indirect_store_2(memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi1>)
func.func @triton_kernel_indirect_store(%dst: memref<?xf32>, %offset: tensor<2x32xi32>, %src: tensor<2x32xf32>, %mask_bool: tensor<2x32xi1>, %mask_i8: tensor<2x32xi8>){
  call @triton_indirect_store_0(%dst, %offset, %src) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>) -> ()
  call @triton_indirect_store_1(%dst, %offset, %src, %mask_i8) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi8>) -> ()
  call @triton_indirect_store_2(%dst, %offset, %src, %mask_bool) : (memref<?xf32>, tensor<2x32xi32>, tensor<2x32xf32>, tensor<2x32xi1>) -> ()
  return
}

// -----
module {
  // CHECK-LABEL func.func @gather_out_to_ub_test
  // CHECK: hfusion.gatherT
  // CHECK-SAME: tensor<2x2xf32>
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
module {
  // CHECK-LABEL func.func @index_put_test
  // CHECK: hfusion.index_put
  // CHECK-SAME: tensor<2x2xf32>
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
module  {
  // CHECK-LABEL func.func @sort_test
  // CHECK: hfusion.sort ins(%2 : tensor<1xf16>) descending = true sort_axis = 0
  func.func @sort_test(%arg0: memref<?xi8>) {
    %true = arith.constant true
    %c0_i64 = arith.constant 0 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
    %alloc = memref.alloc() : memref<1xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1xi8, strided<[1]>> to memref<1xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
    %1 = tensor.empty() : tensor<1xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%0 : tensor<1xi8>) outs(%1 : tensor<1xf16>) -> tensor<1xf16>
    %3 = call @triton_sort(%2, %c0_i64, %true) : (tensor<1xf16>, i64, i1) -> tensor<1xf16>
    return
  }
  func.func private @triton_sort(tensor<1xf16>, i64, i1) -> tensor<1xf16>
}

// -----
module  {
  func.func @sort_test_invalid_axis(%arg0: memref<?xi8>) {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
    %alloc = memref.alloc() : memref<1xi8>
    memref.copy %reinterpret_cast, %alloc : memref<1xi8, strided<[1]>> to memref<1xi8>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1xi8>
    %1 = tensor.empty() : tensor<1xf16>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%0 : tensor<1xi8>) outs(%1 : tensor<1xf16>) -> tensor<1xf16>
    // expected-error@below {{Currently only tail axis sorting is supported}}
    %3 = call @triton_sort(%2, %c1_i64, %true) : (tensor<1xf16>, i64, i1) -> tensor<1xf16>
    return
  }
  func.func private @triton_sort(tensor<1xf16>, i64, i1) -> tensor<1xf16>
}


// -----
// CHECK-LABEL: @scatterT_test
// CHECK: hfusion.scatterT
func.func private @triton_scatter_ub_to_out_0(memref<4x32xf32>, tensor<4x32xf32>, tensor<4x32xi32>, i32, i32, i32, i32, i32, i32, i32, i32)

func.func @scatterT_test(%arg0: memref<4x32xf32>, %arg1: tensor<4x32xf32>,%arg2: tensor<4x32xi32>) attributes {global_kernel = "local"} {
  %c0_i32  = arith.constant 0  : i32
  %c1_i32  = arith.constant 1  : i32
  %c4_i32  = arith.constant 4  : i32
  %c32_i32 = arith.constant 32 : i32

  call @triton_scatter_ub_to_out_0(%arg0, %arg1, %arg2, %c32_i32, %c1_i32, %c32_i32, %c1_i32, %c4_i32, %c32_i32, %c0_i32, %c0_i32) : (memref<4x32xf32>, tensor<4x32xf32>, tensor<4x32xi32>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  return
}
