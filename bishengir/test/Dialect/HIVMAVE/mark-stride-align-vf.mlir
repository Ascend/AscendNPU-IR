// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL:func.func @triton_sum_dim0
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x2xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<8x2xf32, #hivm.address_space<ub>>
// CHECK: call @triton_sum_dim0_outlined_vf_0(%[[ALLOC]]) {hivm.vector_function} : (memref<8x2xf32, #hivm.address_space<ub>>) -> ()

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_sum_dim0_outlined_vf_0(%arg0: memref<8x2xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 2] : vector<1x64xi1>
    scf.for %arg1 = %c0 to %c8 step %c1 {
      %subview = memref.subview %arg0[%arg1, 0] [1, 2] [1, 1] : memref<8x2xf32, #hivm.address_space<ub>> to memref<1x2xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x2xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @triton_sum_dim0_outlined_vf_1(%arg0: memref<27x8x2xf16, #hivm.address_space<ub>>, %arg1: memref<8x2xf32, #hivm.address_space<ub>>, %arg2: memref<8x2xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c27 = arith.constant 27 : index
    %0 = vector.constant_mask [1, 1, 2] : vector<1x1x64xi1>
    %1 = vector.constant_mask [1, 2] : vector<1x64xi1>
    %2 = vector.shape_cast %0 : vector<1x1x64xi1> to vector<1x64xi1>
    scf.for %arg3 = %c0 to %c8 step %c1 {
      %subview = memref.subview %arg0[0, %arg3, 0] [27, 1, 2] [1, 1, 1] : memref<27x8x2xf16, #hivm.address_space<ub>> to memref<27x1x2xf16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg1[%arg3, 0] [1, 2] [1, 1] : memref<8x2xf32, #hivm.address_space<ub>> to memref<1x2xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %3 = vector.transfer_read %subview_1[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x2xf32, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %4 = scf.for %arg4 = %c0 to %c27 step %c1 iter_args(%arg5 = %3) -> (vector<1x64xf32>) {
        %subview_3 = memref.subview %subview[%arg4, 0, 0] [1, 1, 2] [1, 1, 1] : memref<27x1x2xf16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x1x2xf16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>
        %6 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst_0, %0 {in_bounds = [true, true, true]} : memref<1x1x2xf16, strided<[16, 2, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf16>
        %7 = arith.extf %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xf16> to vector<1x1x64xf32>
        %8 = vector.shape_cast %7 : vector<1x1x64xf32> to vector<1x64xf32>
        %9 = arith.addf %arg5, %8 : vector<1x64xf32>
        %10 = arith.select %2, %9, %8 : vector<1x64xi1>, vector<1x64xf32>
        scf.yield %10 : vector<1x64xf32>
      }
      %subview_2 = memref.subview %arg2[%arg3, 0] [1, 2] [1, 1] : memref<8x2xf16, #hivm.address_space<ub>> to memref<1x2xf16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
      %5 = arith.truncf %4 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %5, %subview_2[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x2xf16, strided<[2, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @triton_sum_dim0(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    hivm.hir.set_ctrl false at ctrl[60]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [27, 8, 2], strides: [16, 2, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<27x8x2xf16, strided<[16, 2, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<27x8x2xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<27x8x2xf16, strided<[16, 2, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<27x8x2xf16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x2xf32, #hivm.address_space<ub>>
    call @triton_sum_dim0_outlined_vf_0(%alloc_0) {hivm.vector_function} : (memref<8x2xf32, #hivm.address_space<ub>>) -> ()
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x2xf16, #hivm.address_space<ub>>
    call @triton_sum_dim0_outlined_vf_1(%alloc, %alloc_0, %alloc_1) {hivm.vector_function} : (memref<27x8x2xf16, #hivm.address_space<ub>>, memref<8x2xf32, #hivm.address_space<ub>>, memref<8x2xf16, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8, 2], strides: [2, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<8x2xf16, strided<[2, 1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%alloc_1 : memref<8x2xf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<8x2xf16, strided<[2, 1]>, #hivm.address_space<gm>>)
    return
  }
}

// -----

// CHECK-LABEL:func.func @triton_sum_dim00
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x39xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark %[[ALLOC]] {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x39xf32, #hivm.address_space<ub>>
// CHECK: call @triton_sum_dim0_outlined_vf_00(%[[ALLOC]]) {hivm.vector_function} : (memref<1x39xf32, #hivm.address_space<ub>>) -> ()
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_sum_dim0_outlined_vf_00(%arg0: memref<1x39xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 39] : vector<1x64xi1>
    vector.transfer_write %cst, %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x39xf32, #hivm.address_space<ub>>
    return
  }
  func.func @triton_sum_dim0_outlined_vf_11(%arg0: memref<1x1x39xf32, #hivm.address_space<ub>>, %arg1: memref<1x39xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 39] : vector<1x1x64xi1>
    %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true]} : memref<1x1x39xf32, #hivm.address_space<ub>>, vector<1x1x64xf32>
    %2 = vector.constant_mask [1, 39] : vector<1x64xi1>
    %3 = vector.transfer_read %arg1[%c0, %c0], %cst, %2 {in_bounds = [true, true]} : memref<1x39xf32, #hivm.address_space<ub>>, vector<1x64xf32>
    %4 = vector.shape_cast %0 : vector<1x1x64xi1> to vector<1x64xi1>
    %5 = vector.shape_cast %1 : vector<1x1x64xf32> to vector<1x64xf32>
    %6 = arith.addf %3, %5 : vector<1x64xf32>
    %7 = arith.select %4, %6, %5 : vector<1x64xi1>, vector<1x64xf32>
    vector.transfer_write %7, %arg1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x39xf32, #hivm.address_space<ub>>
    return
  }
  func.func @triton_sum_dim00(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    hivm.hir.set_ctrl false at ctrl[60]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 39], strides: [39, 39, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x1x39xf32, strided<[39, 39, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<1x1x39xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<1x1x39xf32, strided<[39, 39, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<1x1x39xf32, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x39xf32, #hivm.address_space<ub>>
    call @triton_sum_dim0_outlined_vf_00(%alloc_0) {hivm.vector_function} : (memref<1x39xf32, #hivm.address_space<ub>>) -> ()
    call @triton_sum_dim0_outlined_vf_11(%alloc, %alloc_0) {hivm.vector_function} : (memref<1x1x39xf32, #hivm.address_space<ub>>, memref<1x39xf32, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 39], strides: [39, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x39xf32, strided<[39, 1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%alloc_0 : memref<1x39xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1x39xf32, strided<[39, 1]>, #hivm.address_space<gm>>)
    return
  }
}