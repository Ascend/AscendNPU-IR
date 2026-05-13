// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend310B4 -convert-hivm-to-std -split-input-file| FileCheck %s

module {
  func.func private @triton_argmax_dim0_3d_outlined_vf_0(%arg0: memref<13x270xi8, #hivm.address_space<ub>>, %arg1: memref<13x270xf16, #hivm.address_space<ub>>)
  func.func @triton_argmax_dim0_3d(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xi8, #hivm.address_space<gm>> , %arg4: memref<?xi32, #hivm.address_space<gm>> , %arg5: i32, %arg6: i32, %arg7: i32) {
    %c11104_i64 = arith.constant 11104 : i64
    %c10560_i64 = arith.constant 10560 : i64
    %c3520_i64 = arith.constant 3520 : i64
    %c0_i64 = arith.constant 0 : i64
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [13, 27, 10], strides: [270, 10, 1] : memref<?xi8, #hivm.address_space<gm>> to memref<13x27x10xi8, strided<[270, 10, 1]>, #hivm.address_space<gm>>
    %collapse_shape = memref.collapse_shape %reinterpret_cast [[0], [1, 2]] : memref<13x27x10xi8, strided<[270, 10, 1]>, #hivm.address_space<gm>> into memref<13x270xi8, strided<[270, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<13x270xi8, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape : memref<13x270xi8, strided<[270, 1]>, #hivm.address_space<gm>>) outs(%0 : memref<13x270xi8, #hivm.address_space<ub>>) eviction_policy = <EvictFirst>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %1 = hivm.hir.pointer_cast(%c3520_i64) : memref<13x270xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @triton_argmax_dim0_3d_outlined_vf_0(%0, %1) {hivm.vector_function, no_inline} : (memref<13x270xi8, #hivm.address_space<ub>>, memref<13x270xf16, #hivm.address_space<ub>>) -> ()
    %2 = hivm.hir.pointer_cast(%c10560_i64) : memref<1x270xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c11104_i64) : memref<1x270xi32, #hivm.address_space<ub>>
    // CHECK: call @reduce_max_with_index_left_ra_half(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
    hivm.hir.vreduce <max_with_index> ins(%1 : memref<13x270xf16, #hivm.address_space<ub>>) outs(%2, %3 : memref<1x270xf16, #hivm.address_space<ub>>, memref<1x270xi32, #hivm.address_space<ub>>) unsigned_src = true tie_break_left = true reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %collapse_shape_0 = memref.collapse_shape %3 [[0, 1]] : memref<1x270xi32, #hivm.address_space<ub>> into memref<270xi32, #hivm.address_space<ub>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [27, 10], strides: [10, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<27x10xi32, strided<[10, 1]>, #hivm.address_space<gm>>
    %collapse_shape_2 = memref.collapse_shape %reinterpret_cast_1 [[0, 1]] : memref<27x10xi32, strided<[10, 1]>, #hivm.address_space<gm>> into memref<270xi32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape_0 : memref<270xi32, #hivm.address_space<ub>>) outs(%collapse_shape_2 : memref<270xi32, strided<[1]>, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}