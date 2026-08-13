// RUN: bishengir-opt -hivm-enable-stride-align %s | FileCheck %s

// The entry function is visited after the outlined vector function.  Updating
// the call operand therefore introduces a layout change into an already-visited
// callee.  Propagation must carry the aligned layout through expand_shape.

// CHECK-LABEL: func.func @outlined_vf(
// CHECK-SAME: %[[ARG:.*]]: memref<5x8x7xf16, strided<[128, 16, 1]>, #hivm.address_space<ub>>
// CHECK: %[[SLICE:.*]] = memref.subview %[[ARG]]
// CHECK-SAME: memref<1x8x7xf16, strided<[128, 16, 1], offset: ?>
// CHECK: %[[EXPANDED:.*]] = memref.expand_shape %[[SLICE]]
// CHECK-SAME: into memref<1x2x4x7xf16, strided<[128, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
// CHECK: hivm.hir.vexp ins(%[[EXPANDED]]
// CHECK-NOT: unrealized_conversion_cast

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @outlined_vf(%arg0: memref<5x8x7xf16, #hivm.address_space<ub>>, %offset: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %slice = memref.subview %arg0[%offset, 0, 0] [1, 8, 7] [1, 1, 1] : memref<5x8x7xf16, #hivm.address_space<ub>> to memref<1x8x7xf16, strided<[56, 7, 1], offset: ?>, #hivm.address_space<ub>>
    %expanded = memref.expand_shape %slice [[0], [1, 2], [3]] output_shape [1, 2, 4, 7] : memref<1x8x7xf16, strided<[56, 7, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x2x4x7xf16, strided<[56, 28, 7, 1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%expanded : memref<1x2x4x7xf16, strided<[56, 28, 7, 1], offset: ?>, #hivm.address_space<ub>>) outs(%expanded : memref<1x2x4x7xf16, strided<[56, 28, 7, 1], offset: ?>, #hivm.address_space<ub>>)
    return
  }

  func.func @kernel() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %input = memref.alloc() : memref<5x8x7xf16, #hivm.address_space<ub>>
    annotation.mark %input {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<5x8x7xf16, #hivm.address_space<ub>>
    func.call @outlined_vf(%input, %c0) {hivm.vector_function, no_inline} : (memref<5x8x7xf16, #hivm.address_space<ub>>, index) -> ()
    return
  }
}
