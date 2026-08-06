// RUN: bishengir-opt %s -hivm-enable-stride-align \
// RUN: | FileCheck %s --implicit-check-not="unrealized_conversion_cast %expand_shape : memref<512x2xf32, #hivm.address_space<ub>> to memref<512x2xf32, strided<[8, 1]>"
//
// Regression test for EnableStrideAlign's copy-back alignment over-propagation.
//
// A stride-align mark on a copy target (genuinely needing 32B padding) was
// propagated back through `hivm.hir.copy ins(%src) outs(%dst)` onto the src,
// then onward to allocs that merely feed copies (e.g. loop 1's init, consumed
// only by collapse_shape + func.call, never dim-1 vector-accessed). That
// needlessly strided-rewrote the loop iter-arg/result into strided<[8,1]> while
// the loop body yields a separately-allocated dense expand_shape ([2,1]),
// forcing an unsafe dense->strided unrealized_conversion_cast at the yield.
// The fix: copy src (operand 0) does not receive the dst's stride-align info.
//
// Post-fix: loop 1 keeps its init/iter-arg/result as dense
// memref<512x2xf32, #hivm.address_space<ub>> (NOT strided-rewritten) and the
// body yields expand_shape directly — no cast, no copy. Genuinely-needed
// alignments (loop 2's copy targets) are unaffected.

// CHECK-LABEL: func.func @_std_dim_kernel_non_inner
// CHECK: scf.for
// CHECK: iter_args(%{{.*}} = %alloc_{{.*}}) -> (memref<512x2xf32, #hivm.address_space<ub>>)
// CHECK: scf.yield %expand_shape : memref<512x2xf32, #hivm.address_space<ub>>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @_std_dim_kernel_non_inner(%arg2: memref<512x2xf16, #hivm.address_space<gm>>, %arg3: memref<512x2xf16, #hivm.address_space<gm>>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x2xf32, #hivm.address_space<ub>>
    func.call @_vf_init(%alloc_1) {hivm.vector_function, no_inline} : (memref<512x2xf32, #hivm.address_space<ub>>) -> ()
    // loop 1: init = alloc_3, copied from alloc_1. Must stay dense (alloc_1 has
    // no stride-align mark of its own; it must NOT inherit one from loop 2's
    // copy-back). Body yields a dense expand_shape.
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<512x2xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%alloc_1 : memref<512x2xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<512x2xf32, #hivm.address_space<ub>>)
    %22 = scf.for %arg12 = %c0_i32 to %arg5 step %c512_i32 iter_args(%arg13 = %alloc_3) -> (memref<512x2xf32, #hivm.address_space<ub>>) : i32 {
      %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1024xf32, #hivm.address_space<ub>>
      func.call @_vf_fused(%alloc_17) {hivm.vector_function, no_inline} : (memref<1024xf32, #hivm.address_space<ub>>) -> ()
      %expand_shape = memref.expand_shape %alloc_17 [[0, 1]] output_shape [512, 2] : memref<1024xf32, #hivm.address_space<ub>> into memref<512x2xf32, #hivm.address_space<ub>>
      scf.yield %expand_shape : memref<512x2xf32, #hivm.address_space<ub>>
    }
    // loop 2: contains the genuinely-aligned copy targets (subview_17/19) that
    // mark alloc_16/alloc_18. Before the fix this mark back-propagated through
    // `copy ins(%alloc_1) outs(%alloc_18)` onto alloc_1, then onto alloc_3
    // (loop 1's init), strided-rewriting loop 1.
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<512x2xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%alloc_1 : memref<512x2xf32, #hivm.address_space<ub>>) outs(%alloc_6 : memref<512x2xf32, #hivm.address_space<ub>>)
    %23 = scf.for %arg12 = %c0_i32 to %arg5 step %c512_i32 iter_args(%arg13 = %alloc_6) -> (memref<512x2xf32, #hivm.address_space<ub>>) : i32 {
      %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<512x2xf32, #hivm.address_space<ub>>
      %subview_17 = memref.subview %alloc_16[0, 0] [1, 2] [1, 1] : memref<512x2xf32, #hivm.address_space<ub>> to memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
      annotation.mark %subview_17 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
      %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<512x2xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%alloc_1 : memref<512x2xf32, #hivm.address_space<ub>>) outs(%alloc_18 : memref<512x2xf32, #hivm.address_space<ub>>)
      %subview_19 = memref.subview %alloc_18[0, 0] [1, 2] [1, 1] : memref<512x2xf32, #hivm.address_space<ub>> to memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
      annotation.mark %subview_19 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_17 : memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>) outs(%subview_19 : memref<1x2xf32, strided<[2, 1]>, #hivm.address_space<ub>>)
      scf.yield %alloc_18 : memref<512x2xf32, #hivm.address_space<ub>>
    }
    return
  }

  func.func @_vf_init(%arg0: memref<512x2xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} { return }
  func.func @_vf_fused(%arg0: memref<1024xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} { return }
}
