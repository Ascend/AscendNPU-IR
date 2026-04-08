// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-mark-stride-align -hivm-enable-stride-align -split-input-file %s | FileCheck %s

// CHECK-LABEL: @triton_kernel_outlined_vf_0
// CHECK: %[[ARG0:.*]]: memref<3x741xf32, strided<[744, 1]>, #hivm.address_space<ub>>
// CHECK: %[[ARG1:.*]]: memref<3x741xf32, strided<[744, 1]>, #hivm.address_space<ub>>
// CHECK: %[[ARG2:.*]]: memref<3x741xf32, strided<[744, 1]>, #hivm.address_space<ub>>

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @triton_kernel_outlined_vf_0(%arg0: memref<3x741xf32, #hivm.address_space<ub>>, %arg1: memref<3x741xf32, #hivm.address_space<ub>>, %arg2: memref<3x741xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c741 = arith.constant 741 : index
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c741 step %c64 {
        %0 = affine.min affine_map<(d0) -> (-d0 + 741, 64)>(%arg4)
        %subview = memref.subview %arg0[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4] [1, %0] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
        %2 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %3 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %4 = arith.addf %2, %3 : vector<1x64xf32>
        vector.transfer_write %4, %subview_1[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[741, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @triton_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false, false, false]> : vector<9xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv"} {
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c741 = arith.constant 741 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %arg8, %arg7 : i32
    %3 = arith.divsi %1, %2 : i32
    %4 = arith.remsi %3, %arg6 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.index_cast %arg4 : i32 to index
    %8 = arith.maxsi %6, %c0 : index
    %9 = arith.minsi %8, %c741 : index
    %10 = arith.minsi %9, %c741 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x741xf32, #hivm.address_space<ub>>
    scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %11 = arith.muli %arg9, %c3_i32 : i32
      %12 = arith.addi %5, %11 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%13, %6]
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_0 = memref.alloc() : memref<3x741xf32, #hivm.address_space<ub>>
      %15 = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%13]
      %16 = arith.maxsi %13, %7 : index
      %17 = arith.minsi %15, %16 : index
      %18 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%17, %13]
      %19 = arith.minsi %18, %c3 : index
      %subview = memref.subview %reinterpret_cast[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %alloc_0[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_3 = memref.alloc() : memref<3x741xf32, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %alloc_3[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) left_padding_num = %c0 : index
      func.call @triton_kernel_outlined_vf_0(%alloc_0, %alloc_3, %alloc) {hivm.vector_function} : (memref<3x741xf32, #hivm.address_space<ub>>, memref<3x741xf32, #hivm.address_space<ub>>, memref<3x741xf32, #hivm.address_space<ub>>) -> ()
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [3, 741], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %alloc[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>
      %subview_8 = memref.subview %reinterpret_cast_6[0, 0] [%19, %10] [1, 1] : memref<3x741xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%subview_7 : memref<?x?xf32, strided<[741, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    }
    return
  }
}