// RUN: bishengir-opt -hivm-enable-stride-align -split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: @test_elementwise_unary_op_unaligned
  func.func @test_elementwise_unary_op_unaligned(%arg0: index, %arg1: index) {
    // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
    // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index
    // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16x1xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW1:.*]] = memref.subview %[[ALLOC1]]{{\[}}0, 0, 0] {{\[}}2, 15, 1] {{\[}}1, 1, 1]
    %alloc = memref.alloc() : memref<2x15xf16, #hivm.address_space<ub>>
    // CHECK: %[[ALLOC2:.*]] = memref.alloc(%[[LEN:.*]]) : memref<?x16x1xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW2:.*]] = memref.subview %[[ALLOC2]]{{\[}}0, 0, 0] {{\[}}%[[LEN]], 15, 1] {{\[}}1, 1, 1]
    %alloc_0 = memref.alloc(%arg0) : memref<?x15xf16, #hivm.address_space<ub>>
    // CHECK: %[[PAD3:.*]] = arith.addi %[[LEN3:.*]], %[[C15]] : index
    // CHECK: %[[REM3:.*]] = arith.remsi %[[PAD3]], %[[C16]] : index
    // CHECK: %[[ALIGN3:.*]] = arith.subi %[[PAD3]], %[[REM3]] : index
    // CHECK: %[[ALLOC3:.*]] = memref.alloc(%[[ALIGN3]]) : memref<2x?x1xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW3:.*]] = memref.subview %[[ALLOC3]]{{\[}}0, 0, 0] {{\[}}2, %[[LEN3]], 1] {{\[}}1, 1, 1]
    %alloc_1 = memref.alloc(%arg1) : memref<2x?xf16, #hivm.address_space<ub>>
    // CHECK: %[[PAD4:.*]] = arith.addi %[[LEN4:.*]], %[[C15]] : index
    // CHECK: %[[REM4:.*]] = arith.remsi %[[PAD4]], %[[C16]] : index
    // CHECK: %[[ALIGN4:.*]] = arith.subi %[[PAD4]], %[[REM4]] : index
    // CHECK: %[[ALLOC4:.*]] = memref.alloc(%[[LEN]], %[[ALIGN4:.*]]) : memref<?x?x1xf16, #hivm.address_space<ub>>
    // CHECK: %[[SUBVIEW4:.*]] = memref.subview %[[ALLOC4]]{{\[}}0, 0, 0] {{\[}}%[[LEN]], %[[LEN4]], 1] {{\[}}1, 1, 1]
    %alloc_2 = memref.alloc(%arg0, %arg1) : memref<?x?xf16, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    bishengir_test.elementwise_unary ins(%alloc : memref<2x15xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x15xf16, #hivm.address_space<ub>>)
    annotation.mark %alloc_0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x15xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    bishengir_test.elementwise_unary ins(%alloc_0 : memref<?x15xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x15xf16, #hivm.address_space<ub>>)
    annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x?xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    bishengir_test.elementwise_unary ins(%alloc_1 : memref<2x?xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x15xf16, #hivm.address_space<ub>>)
    annotation.mark %alloc_2 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x?xf16, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf16, #hivm.address_space<ub>>
    bishengir_test.elementwise_unary ins(%alloc_2 : memref<?x?xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x15xf16, #hivm.address_space<ub>>)
    return
  }
}

// -----

// CHECK-LABEL: @test_elementwise_unary_op_aligned
func.func @test_elementwise_unary_op_aligned(%arg0: index, %arg1: index) {
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x16xf16, #hivm.address_space<ub>>
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc() : memref<2x16xf16, #hivm.address_space<ub>>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc(%[[LEN:.*]]) : memref<?x16xf16, #hivm.address_space<ub>>
  // CHECK-NOT: memref.subview
  %alloc_0 = memref.alloc(%arg0) : memref<?x16xf16, #hivm.address_space<ub>>
  // CHECK: %[[PAD3:.*]] = arith.addi %[[LEN3:.*]], %[[C15]] : index
  // CHECK: %[[REM3:.*]] = arith.remsi %[[PAD3]], %[[C16]] : index
  // CHECK: %[[ALIGN3:.*]] = arith.subi %[[PAD3]], %[[REM3]] : index
  // CHECK: %[[ALLOC3:.*]] = memref.alloc(%[[ALIGN3]]) : memref<2x?x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW3:.*]] = memref.subview %[[ALLOC3]]{{\[}}0, 0, 0] {{\[}}2, %[[LEN3]], 1] {{\[}}1, 1, 1]
  %alloc_1 = memref.alloc(%arg1) : memref<2x?xf16, #hivm.address_space<ub>>
  // CHECK: %[[PAD4:.*]] = arith.addi %[[LEN4:.*]], %[[C15]] : index
  // CHECK: %[[REM4:.*]] = arith.remsi %[[PAD4]], %[[C16]] : index
  // CHECK: %[[ALIGN4:.*]] = arith.subi %[[PAD4]], %[[REM4]] : index
  // CHECK: %[[ALLOC4:.*]] = memref.alloc(%[[LEN:.*]], %[[ALIGN4:.*]]) : memref<?x?x1xf16, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW4:.*]] = memref.subview %[[ALLOC4]]{{\[}}0, 0, 0] {{\[}}%[[LEN]], %[[LEN4]], 1] {{\[}}1, 1, 1]
  %alloc_2 = memref.alloc(%arg0, %arg1) : memref<?x?xf16, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x16xf16, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%alloc : memref<2x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x16xf16, #hivm.address_space<ub>>)
  annotation.mark %alloc_0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x16xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x16xf16, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%alloc_0 : memref<?x16xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x16xf16, #hivm.address_space<ub>>)
  annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x?xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x16xf16, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%alloc_1 : memref<2x?xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x16xf16, #hivm.address_space<ub>>)
  annotation.mark %alloc_2 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x?xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x16xf16, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%alloc_2 : memref<?x?xf16, #hivm.address_space<ub>>) outs(%alloc : memref<2x16xf16, #hivm.address_space<ub>>)
  return
}

// -----

// COM: FIXME once the view propagation fixes
module {
memref.global "private" constant @__constant_2xi32 : memref<2xi64> = dense<[255,15]> {alignment = 32 : i64}
memref.global "private" constant @__constant_4xi32 : memref<4xi64> = dense<[15,3,5,15]> {alignment = 32 : i64}
// CHECK-LABEL: @test_mark_view_like
func.func @test_mark_view_like() {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<15x15x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: unrealized_conversion_cast
  %a = memref.alloc() : memref<15x15x15xf32, #hivm.address_space<ub>>
  %cast = memref.cast %a : memref<15x15x15xf32, #hivm.address_space<ub>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
  annotation.mark %cast {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>
  hfusion.elemwise_unary ins(%cast : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>) outs(%cast : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>, #hivm.address_space<ub>>)
  // CHECK: %[[B:.*]] = memref.alloc() : memref<15x15x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: unrealized_conversion_cast
  %b = memref.alloc() : memref<15x15x15xf32, #hivm.address_space<ub>>
  %collapse = memref.collapse_shape %b [[0, 1], [2]] : memref<15x15x15xf32, #hivm.address_space<ub>> into memref<225x15xf32, #hivm.address_space<ub>>
  annotation.mark %collapse {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<225x15xf32, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%collapse : memref<225x15xf32, #hivm.address_space<ub>>) outs(%collapse : memref<225x15xf32, #hivm.address_space<ub>>)
  // CHECK: %[[C:.*]] = memref.alloc() : memref<15x15x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: unrealized_conversion_cast
  %c = memref.alloc() : memref<15x15x15xf32, #hivm.address_space<ub>>
  %expand = memref.expand_shape %c [[0], [1, 2], [3]] output_shape [15, 3, 5, 15] : memref<15x15x15xf32, #hivm.address_space<ub>> into memref<15x3x5x15xf32, #hivm.address_space<ub>>
  annotation.mark %expand {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<15x3x5x15xf32, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%expand : memref<15x3x5x15xf32, #hivm.address_space<ub>>) outs(%expand : memref<15x3x5x15xf32, #hivm.address_space<ub>>)
  // CHECK: %[[D:.*]] = memref.alloc() : memref<15x15x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: unrealized_conversion_cast
  %d = memref.alloc() : memref<15x15x15xf32, #hivm.address_space<ub>>
  %shape1 = memref.get_global @__constant_2xi32 : memref<2xi64>
  %reshape1 = memref.reshape %d(%shape1) : (memref<15x15x15xf32, #hivm.address_space<ub>>, memref<2xi64>) -> memref<225x15xf32, #hivm.address_space<ub>>
  annotation.mark %reshape1 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<225x15xf32, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%reshape1 : memref<225x15xf32, #hivm.address_space<ub>>) outs(%reshape1 : memref<225x15xf32, #hivm.address_space<ub>>)
  // CHECK: %[[E:.*]] = memref.alloc() : memref<15x15x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: annotation.mark
  // CHECK-NOT: unrealized_conversion_cast
  %e = memref.alloc() : memref<15x15x15xf32, #hivm.address_space<ub>>
  %shape2 = memref.get_global @__constant_4xi32 : memref<4xi64>
  %reshape2 = memref.reshape %e(%shape2) : (memref<15x15x15xf32, #hivm.address_space<ub>>, memref<4xi64>) -> memref<15x3x5x15xf32, #hivm.address_space<ub>>
  annotation.mark %reshape2 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<15x3x5x15xf32, #hivm.address_space<ub>>
  bishengir_test.elementwise_unary ins(%reshape2 : memref<15x3x5x15xf32, #hivm.address_space<ub>>) outs(%expand : memref<15x3x5x15xf32, #hivm.address_space<ub>>)
  return
}
}

// -----

// CHECK-LABEL: @test_mark_reduce_max_with_index_inner
func.func @test_mark_reduce_max_with_index_inner(%arg0: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg1: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg2: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg3: memref<?xi32, #hivm.address_space<gm>>, %arg4: i32) attributes {func_dyn_memref_args = dense<[true, true, true, true, false]> : vector<5xi1>, global_kernel = "local", hacc.dso_local = "", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.spir_kernel = "", hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c4096 = arith.constant 4096 : index
  %c16384_i32 = arith.constant 16384 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %0 = hivm.hir.get_block_idx -> i64
  %1 = arith.trunci %0 : i64 to i32
  %2 = arith.muli %1, %c16384_i32 : i32
  %3 = arith.index_cast %2 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [4096, 4], strides: [1, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4096x8x1xf32, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]{{\[}}0, 0, 0] {{\[}}4096, 4, 1] {{\[}}1, 1, 1] : memref<4096x8x1xf32, #hivm.address_space<ub>> to memref<4096x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc = memref.alloc() : memref<4096x4xf32, #hivm.address_space<ub>>
  %4 = arith.addi %3, %c4096 : index
  %5 = arith.index_cast %arg4 : i32 to index
  %6 = arith.maxsi %3, %5 : index
  %7 = arith.minsi %4, %6 : index
  %8 = arith.subi %7, %3 : index
  %9 = arith.cmpi slt, %8, %c4096 : index
  scf.if %9 {
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4096x4xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<4096x4xf32, #hivm.address_space<ub>>)
  }
  %subview = memref.subview %reinterpret_cast[0, 0] [%8, 4] [1, 1] : memref<4096x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_0 = memref.subview %alloc[0, 0] [%8, 4] [1, 1] : memref<4096x4xf32, #hivm.address_space<ub>> to memref<?x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>
  annotation.mark %subview_0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview : memref<?x4xf32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x4xf32, strided<[4, 1]>, #hivm.address_space<ub>>)
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [4096, 4], strides: [1, 1] : memref<?xi32, #hivm.address_space<gm>> to memref<4096x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  // CHECK: %[[ALLOC2:.*]] = memref.alloc() : memref<4096x8x1xi32, #hivm.address_space<ub>>
  // CHECK: %[[SUBVIEW2:.*]] = memref.subview %[[ALLOC2]]{{\[}}0, 0, 0] {{\[}}4096, 4, 1] {{\[}}1, 1, 1] : memref<4096x8x1xi32, #hivm.address_space<ub>> to memref<4096x4xi32, strided<[8, 1]>, #hivm.address_space<ub>>
  %alloc_2 = memref.alloc() : memref<4096x4xi32, #hivm.address_space<ub>>
  scf.if %9 {
    annotation.mark %alloc_2 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4096x4xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c0_i32 : i32) outs(%alloc_2 : memref<4096x4xi32, #hivm.address_space<ub>>)
  }
  %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%8, 4] [1, 1] : memref<4096x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_4 = memref.subview %alloc_2[0, 0] [%8, 4] [1, 1] : memref<4096x4xi32, #hivm.address_space<ub>> to memref<?x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>
  annotation.mark %subview_4 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%subview_3 : memref<?x4xi32, strided<[1, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x4xi32, strided<[4, 1]>, #hivm.address_space<ub>>)
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32, #hivm.address_space<ub>>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4096xi32, #hivm.address_space<ub>>
  %expand_shape = memref.expand_shape %alloc_5 [[0, 1]] output_shape [4096, 1] : memref<4096xf32, #hivm.address_space<ub>> into memref<4096x1xf32, #hivm.address_space<ub>>
  %expand_shape_7 = memref.expand_shape %alloc_6 [[0, 1]] output_shape [4096, 1] : memref<4096xi32, #hivm.address_space<ub>> into memref<4096x1xi32, #hivm.address_space<ub>>
  %alloc_8 = memref.alloc() : memref<16384xf32, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<4096x4xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vreduce <max_with_index_left> ins(%[[SUBVIEW]] : memref<4096x4xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
  hivm.hir.vreduce <max_with_index_left> ins(%alloc : memref<4096x4xf32, #hivm.address_space<ub>>) outs(%expand_shape, %expand_shape_7 : memref<4096x1xf32, #hivm.address_space<ub>>, memref<4096x1xi32, #hivm.address_space<ub>>) temp_buffer(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>) reduce_dims = [1]
  %10 = arith.muli %1, %c4096_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%11], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = arith.addi %11, %c4096 : index
  %13 = arith.maxsi %11, %5 : index
  %14 = arith.minsi %12, %13 : index
  %15 = arith.subi %14, %11 : index
  %subview_10 = memref.subview %alloc_5[0] [%15] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_11 = memref.subview %reinterpret_cast_9[0] [%15] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %reinterpret_cast_12 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [4096], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<4096xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_13 = memref.subview %alloc_6[0] [%15] [1] : memref<4096xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
  %subview_14 = memref.subview %reinterpret_cast_12[0] [%15] [1] : memref<4096xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.store ins(%subview_13 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_14 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  return
}

// -----

// CHECK-LABEL: func @test_unit_collapse
// CHECK: memref.alloc() : memref<3x8xf32, #hivm.address_space<ub>>
// CHECK: memref.alloc() : memref<3x8x1xf32, #hivm.address_space<ub>>
func.func @test_unit_collapse() {
  %alloc = memref.alloc() : memref<3xf32, #hivm.address_space<ub>>
  %expanded_0 = memref.expand_shape %alloc [[0, 1]] output_shape [3, 1] : memref<3xf32, #hivm.address_space<ub>> into memref<3x1xf32, #hivm.address_space<ub>>
  annotation.mark %expanded_0 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<3x1xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<3x1xf32, #hivm.address_space<ub>>
  hivm.hir.vexp ins(%expanded_0 : memref<3x1xf32, #hivm.address_space<ub>>) outs(%dst : memref<3x1xf32, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func @test_align_up_from_scf_iter_arg
func.func @test_align_up_from_scf_iter_arg() {
  %c0_i32 = arith.constant 0 : i32
  %c80_i32 = arith.constant 80 : i32
  %c32_i32 = arith.constant 32 : i32
  // CHECK: memref.alloc() : memref<32x8xf32, #hivm.address_space<ub>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
  %res = scf.for %arg0 = %c0_i32 to %c80_i32 step %c32_i32 iter_args(%arg1 = %alloc) -> (memref<32xf32, #hivm.address_space<ub>>) : i32 {
    annotation.mark %arg1 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<32xf32, #hivm.address_space<ub>>
    // CHECK: memref.alloc() : memref<32x8xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%arg1 : memref<32xf32, #hivm.address_space<ub>>) outs(%dst : memref<32xf32, #hivm.address_space<ub>>)
    scf.yield %arg1 : memref<32xf32, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func @test_align_down_from_scf_if
func.func @test_align_down_from_scf_if(%cond : i1) {
  %c0_i32 = arith.constant 0 : i32
  %c80_i32 = arith.constant 80 : i32
  %c32_i32 = arith.constant 32 : i32
  // CHECK: memref.alloc() : memref<2x16x1xf32, #hivm.address_space<ub>>
  // CHECK-NOT: builtin.unrealized_conversion_cast
  %alloc = memref.alloc() : memref<2x15xf32, #hivm.address_space<ub>>
  %alloc2 = memref.alloc() : memref<2x15xf32, #hivm.address_space<ub>>
  %res = scf.if %cond -> (memref<2x15xf32, #hivm.address_space<ub>>) {
    annotation.mark %alloc {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<2x15xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%alloc : memref<2x15xf32, #hivm.address_space<ub>>) outs(%dst : memref<2x15xf32, #hivm.address_space<ub>>)
    scf.yield %dst : memref<2x15xf32, #hivm.address_space<ub>>
  } else {
    annotation.mark %alloc2 {hivm.stride_align_dims = array<i32: 0>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<2x15xf32, #hivm.address_space<ub>>
    scf.yield %alloc2 : memref<2x15xf32, #hivm.address_space<ub>>
  }
  %dst2 = memref.alloc() : memref<2x15xf32, #hivm.address_space<ub>>
  hivm.hir.vexp ins(%res#0 : memref<2x15xf32, #hivm.address_space<ub>>) outs(%dst2 : memref<2x15xf32, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK: func @test_cumsum_unalignment
func.func @test_cumsum_unalignment(%arg0: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>, %arg1: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>, %arg2: memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %alloc = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>) outs(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>)
  // CHECK: memref.alloc() : memref<5x3x3x8x3x5x1xi32, #hivm.address_space<ub>>
  %alloc_0 = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  annotation.mark %alloc_0 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumsum ins(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) cum_dims = [2]
  // CHECK: memref.alloc() : memref<5x3x3x8x3x5x1xi32, #hivm.address_space<ub>>
  %alloc_1 = memref.alloc() : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 2>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>
  hivm.hir.vcumprod ins(%alloc : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) cum_dims = [2]
  hivm.hir.store ins(%alloc_0 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%arg1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%alloc_1 : memref<5x3x3x3x3x5xi32, #hivm.address_space<ub>>) outs(%arg2 : memref<5x3x3x3x3x5xi32, #hivm.address_space<gm>>)
  return
}

// -----

module {
  func.func @test_propagate_unrealized_conversion(%arg0: memref<1x2x3x2x5x60000x7x2xi16, #hivm.address_space<gm>>, %ub: index, %size: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %ub step %c1 {
      %alloc = memref.alloc(%size) : memref<?x7x2xi16, #hivm.address_space<ub>>
      %subview = memref.subview %alloc[0, 0, 0] [%size, 7, 2] [ 1, 1, 1] : memref<?x7x2xi16, #hivm.address_space<ub>> to memref<?x7x2xi16, strided<[14, 2, 1]>, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x7x2xi16, #hivm.address_space<ub>>
      %alloc_0 = memref.alloc(%size) : memref<?x7x2xi16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview : memref<?x7x2xi16, strided<[14, 2, 1]>, #hivm.address_space<ub>>) outs(%alloc_0 : memref<?x7x2xi16, #hivm.address_space<ub>>)
      %1 = scf.for %arg2 = %c0 to %ub step %c1 iter_args(%arg3 = %alloc_0) -> (memref<?x7x2xi16, #hivm.address_space<ub>>) {
        %2 = scf.for %arg4 = %c0 to %ub step %c1 iter_args(%arg5 = %arg3) -> (memref<?x7x2xi16, #hivm.address_space<ub>>) {
          // CHECK-NOT: builtin.unrealized_conversion_cast
          %alloc_2 = memref.alloc(%size) : memref<?x7x2xi16, #hivm.address_space<ub>>
          hivm.hir.vor ins(%alloc_2, %arg5 : memref<?x7x2xi16, #hivm.address_space<ub>>, memref<?x7x2xi16, #hivm.address_space<ub>>) outs(%alloc_2 : memref<?x7x2xi16, #hivm.address_space<ub>>)
          scf.yield %alloc_2 : memref<?x7x2xi16, #hivm.address_space<ub>>
        }
        scf.yield %2 : memref<?x7x2xi16, #hivm.address_space<ub>>
      }
      %alloc_1 = memref.alloc(%size) : memref<?x1x2xi16, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 32>} : memref<?x1x2xi16, #hivm.address_space<ub>>
    }
    return
  }
}
