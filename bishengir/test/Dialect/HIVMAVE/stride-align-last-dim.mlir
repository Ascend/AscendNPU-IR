// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-mark-stride-align -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: @triton_eq_outlined_vf_0
// CHECK: %[[ARG0:.*]]: memref<37x5x3xi32, strided<[40, 8, 1]>, #hivm.address_space<ub>>
// CHECK: %[[ARG1:.*]]: memref<37x5x3xi32, strided<[40, 8, 1]>, #hivm.address_space<ub>>
// CHECK: %[[ARG2:.*]]: memref<37x5x3xi32, strided<[40, 8, 1]>, #hivm.address_space<ub>>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_eq_outlined_vf_0(%arg0: memref<37x5x3xi32, #hivm.address_space<ub>>, %arg1: memref<37x5x3xi32, #hivm.address_space<ub>>, %arg2: memref<37x5x3xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %0 = vector.constant_mask [1, 1, 3] : vector<1x1x64xi1>
    scf.for %arg3 = %c0 to %c37 step %c1 {
      scf.for %arg4 = %c0 to %c5 step %c1 {
        %subview = memref.subview %arg0[%arg3, %arg4, 0] [1, 1, 3] [1, 1, 1] : memref<37x5x3xi32, #hivm.address_space<ub>> to memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>
        // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ARG0:.*]]{{\[}}%[[ARG3:.*]], %[[ARG4:.*]], 0] {{\[}}1, 1, 3] {{\[}}1, 1, 1] : memref<37x5x3xi32, strided<[40, 8, 1]>, #hivm.address_space<ub>> to memref<1x1x3xi32, strided<[40, 8, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg3, %arg4, 0] [1, 1, 3] [1, 1, 1] : memref<37x5x3xi32, #hivm.address_space<ub>> to memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4, 0] [1, 1, 3] [1, 1, 1] : memref<37x5x3xi32, #hivm.address_space<ub>> to memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>
        // CHECK: %[[READ:.*]] = vector.transfer_read %subview{{\[}}%[[CONSTANT0:.*]], %[[CONSTANT0:.*]], %[[CONSTANT0:.*]]], %[[PASS_THRU:.*]], %[[MASK:.*]] {in_bounds = {{\[}}true, true, true]} : memref<1x1x3xi32, strided<{{\[}}40, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
        %1 = vector.transfer_read %subview[%c0, %c0, %c0], %c0_i32, %0 {in_bounds = [true, true, true]} : memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
        %2 = vector.transfer_read %subview_0[%c0, %c0, %c0], %c0_i32, %0 {in_bounds = [true, true, true]} : memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xi32>
        %3 = arith.cmpi eq, %1, %2 : vector<1x1x64xi32>
        %4 = arith.uitofp %3 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xi1> to vector<1x1x64xf16>
        %5 = arith.fptosi %4 {round_mode = #hfusion.round_mode<rint>} : vector<1x1x64xf16> to vector<1x1x64xi32>
        vector.transfer_write %5, %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : vector<1x1x64xi32>, memref<1x1x3xi32, strided<[15, 3, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @triton_eq(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [37, 5, 3], strides: [15, 3, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<37x5x3xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>) outs(%alloc : memref<37x5x3xi32, #hivm.address_space<ub>>)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [37, 5, 3], strides: [15, 3, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>
    %alloc_1 = memref.alloc() : memref<37x5x3xi32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.load ins(%[[IN:.*]] : memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>) outs(%[[SUBVIEW:.*]] : memref<37x5x3xi32, strided<[40, 8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.load ins(%reinterpret_cast_0 : memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>) outs(%alloc_1 : memref<37x5x3xi32, #hivm.address_space<ub>>)
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [37, 5, 3], strides: [15, 3, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<37x5x3xi32, #hivm.address_space<ub>>
    call @triton_eq_outlined_vf_0(%alloc, %alloc_1, %alloc_3) {hivm.vector_function} : (memref<37x5x3xi32, #hivm.address_space<ub>>, memref<37x5x3xi32, #hivm.address_space<ub>>, memref<37x5x3xi32, #hivm.address_space<ub>>) -> ()
    hivm.hir.store ins(%alloc_3 : memref<37x5x3xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<37x5x3xi32, strided<[15, 3, 1]>, #hivm.address_space<gm>>)
    return
  }
}