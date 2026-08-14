// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s

// Test that InsertPadConstMark generates a `pad_const` annotation when the
// `linalg.fill` that pads a GM-load destination targets a collapse_shape alias
// of the buffer rather than the alloc directly. Before the fix, the producer
// only looked at direct users of the root alloc and missed fills on aliases,
// so no pad_const was emitted (and downstream DMA zero-padded, overwriting the
// fill). This mirrors the quantile_bitonic_kernel padding structure.
//
// The mark is attached to the alloc root (not the collapse alias) so it
// survives later canonicalization of the alias.

// CHECK-LABEL: func.func @test_pad_const_mark_on_collapse_alias
// CHECK: %[[PAD:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: annotation.mark %[[ALLOC]] keys = ["pad_const"] values = [%[[PAD]] : f32]
// CHECK-NOT: annotation.mark %{{.*}} keys = ["pad_const"]
func.func @test_pad_const_mark_on_collapse_alias(
    %arg0 : memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %size : index) {
  %cst = arith.constant 3.40282347E+38 : f32
  %alloc = memref.alloc() : memref<2x1x2x1x2x1x1x1x1x1x2x1xf32>
  // collapse the alloc down to a flat 16-elem memref; the fill targets THIS
  // alias (not the alloc), which is the structure InsertPadConstMark must
  // trace through.
  %collapse = memref.collapse_shape %alloc [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] : memref<2x1x2x1x2x1x1x1x1x1x2x1xf32> into memref<16xf32>
  linalg.fill ins(%cst : f32) outs(%collapse : memref<16xf32>)
  // copy from a GM function-arg subview into a subview of the collapse alias
  %sub_gm = memref.subview %arg0[0] [%size] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  %sub_ub = memref.subview %collapse[0] [%size] [1] : memref<16xf32> to memref<?xf32>
  memref.copy %sub_gm, %sub_ub : memref<?xf32, strided<[1]>> to memref<?xf32>
  return
}

// -----
// Contrast case: fill targets the alloc directly. The pad_const mark should
// still be generated on the alloc (this path worked before the fix too);
// included to guard both shapes the producer now supports.
// CHECK-LABEL: func.func @test_pad_const_mark_on_alloc_direct
// CHECK: %[[PAD:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: annotation.mark %[[ALLOC]] keys = ["pad_const"] values = [%[[PAD]] : f32]
// CHECK-NOT: annotation.mark %{{.*}} keys = ["pad_const"]
func.func @test_pad_const_mark_on_alloc_direct(
    %arg0 : memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
    %size : index) {
  %cst = arith.constant 3.40282347E+38 : f32
  %alloc = memref.alloc() : memref<16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<16xf32>)
  %sub_gm = memref.subview %arg0[0] [%size] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  %sub_ub = memref.subview %alloc[0] [%size] [1] : memref<16xf32> to memref<?xf32>
  memref.copy %sub_gm, %sub_ub : memref<?xf32, strided<[1]>> to memref<?xf32>
  return
}
