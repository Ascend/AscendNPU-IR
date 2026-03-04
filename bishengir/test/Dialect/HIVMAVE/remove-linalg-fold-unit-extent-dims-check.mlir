// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-mark-stride-align -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @triton_sum_dim2_outlined_vf_0
// CHECK: %[[ARG0:.*]]: memref<1x1x39xi8, strided<[64, 64, 1]>, #hivm.address_space<ub>>
// CHECK: %[[ARG1:.*]]: memref<1x1xi8, strided<[32, 1]>, #hivm.address_space<ub>>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_sum_dim2_outlined_vf_0(%arg0: memref<1x1x39xi8, #hivm.address_space<ub>>, %arg1: memref<1x1xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 1, 39] : vector<1x1x256xi1>
    %1 = vector.transfer_read %arg0[%c0, %c0, %c0], %c0_i8, %0 {in_bounds = [true, true, true]} : memref<1x1x39xi8, #hivm.address_space<ub>>, vector<1x1x256xi8>
    %2 = vector.transfer_read %arg1[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<1x1xi8, #hivm.address_space<ub>>, vector<1x1xi8>
    %3 = vector.mask %0 { vector.multi_reduction <add>, %1, %2 [2] : vector<1x1x256xi8> to vector<1x1xi8> } : vector<1x1x256xi1> -> vector<1x1xi8>
    vector.transfer_write %3, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xi8>, memref<1x1xi8, #hivm.address_space<ub>>
    return
  }
  func.func @triton_sum_dim2(%arg0: memref<?xi8, #hivm.address_space<gm>>, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %c0 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    hivm.hir.set_ctrl false at ctrl[60]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1, 1, 39], strides: [39, 39, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1x39xi8, strided<[39, 39, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<1x1x39xi8, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x1x39xi8, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<1x1x39xi8, strided<[39, 39, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<1x1x39xi8, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8, #hivm.address_space<ub>>
    annotation.mark %alloc_0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x1xi8, #hivm.address_space<ub>>
    memref.store %c0_i8, %alloc_0[%c0, %c0] : memref<1x1xi8, #hivm.address_space<ub>>
    call @triton_sum_dim2_outlined_vf_0(%alloc, %alloc_0) {hivm.vector_function} : (memref<1x1x39xi8, #hivm.address_space<ub>>, memref<1x1xi8, #hivm.address_space<ub>>) -> ()
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<1x1xi8, strided<[1, 1]>, #hivm.address_space<gm>>
    hivm.hir.store ins(%alloc_0 : memref<1x1xi8, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1x1xi8, strided<[1, 1]>, #hivm.address_space<gm>>)
    return
  }
}