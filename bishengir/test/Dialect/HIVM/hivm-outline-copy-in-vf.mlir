// RUN: bishengir-opt %s -hivm-outline-copy-in-VF -cse -canonicalize -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----

// CHECK-LABEL: func.func @outlined_vf
// CHECK-NOT: hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
func.func @outlined_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK-NEXT: call @outlined_vf(%[[SRC]], %[[DST]]) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
func.func @caller() {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @outlined_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @guarded_vf
// CHECK:      %0 = vector.constant_mask [16] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %arg0[%c0], %cst, %0 : memref<16xf32>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %arg1[%c0], %0 : vector<64xf32>, memref<16xf32>
func.func @guarded_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
  vector.transfer_write %cst, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @guarded_caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK-NOT: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK: call @guarded_vf(%[[SRC]], %[[DST]]) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
func.func @guarded_caller() {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @guarded_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @subview_load_vf
// CHECK:      %0 = vector.constant_mask [4] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %subview[%c0], %cst, %0 : memref<4xf32, strided<[1]>>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %subview_0[%c0], %0 : vector<64xf32>, memref<4xf32, strided<[1]>>
func.func @subview_load_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %src = memref.subview %arg0[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  %dst = memref.subview %arg1[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%src : memref<4xf32, strided<[1], offset: ?>>) outs(%dst : memref<4xf32, strided<[1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL: func.func @unknown_write_vf
// CHECK:      %0 = vector.constant_mask [16] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %arg0[%c0], %cst, %0 : memref<16xf32>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %arg1[%c0], %0 : vector<64xf32>, memref<16xf32>
func.func @unknown_write_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%arg1) : (memref<16xf32>) -> ()
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @unknown_write_caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK-NOT: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK: call @unknown_write_vf(%[[SRC]], %[[DST]]) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
func.func @unknown_write_caller() {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @unknown_write_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @recursive_unknown_write_vf
// CHECK:      %0 = vector.constant_mask [16] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %arg0[%c0], %cst, %0 : memref<16xf32>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %arg1[%c0], %0 : vector<64xf32>, memref<16xf32>
func.func @recursive_unknown_write_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %flag: i1) attributes {hivm.vector_function, no_inline} {
  scf.if %flag {
    "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
  }
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @recursive_unknown_write_caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK-NOT: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK: call @recursive_unknown_write_vf(%[[SRC]], %[[DST]], %{{.*}}) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>, i1) -> ()
func.func @recursive_unknown_write_caller(%flag: i1) {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @recursive_unknown_write_vf(%src, %dst, %flag) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>, i1) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @unknown_region_capture_dst_vf
// CHECK:      %0 = vector.constant_mask [16] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %arg0[%c0], %cst, %0 : memref<16xf32>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %arg1[%c0], %0 : vector<64xf32>, memref<16xf32>
func.func @unknown_region_capture_dst_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_region"() ({
    "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
  }) : () -> ()
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @unknown_region_capture_dst_caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK-NOT: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK: call @unknown_region_capture_dst_vf(%[[SRC]], %[[DST]]) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
func.func @unknown_region_capture_dst_caller() {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @unknown_region_capture_dst_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @unknown_region_capture_src_vf
// CHECK:      %0 = vector.constant_mask [16] : vector<64xi1>
// CHECK-NEXT: %1 = vector.transfer_read %arg0[%c0], %cst, %0 : memref<16xf32>, vector<64xf32>
// CHECK-NEXT: vector.transfer_write %1, %arg1[%c0], %0 : vector<64xf32>, memref<16xf32>
func.func @unknown_region_capture_src_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_region"() ({
    "test.unknown_nested_write"(%arg0) : (memref<16xf32>) -> ()
  }) : () -> ()
  hivm.hir.copy ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>)
  return
}

// CHECK-LABEL: func.func @unknown_region_capture_src_caller
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK-NOT: hivm.hir.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK: call @unknown_region_capture_src_vf(%[[SRC]], %[[DST]]) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
func.func @unknown_region_capture_src_caller() {
  %src = memref.alloc() : memref<16xf32>
  %dst = memref.alloc() : memref<16xf32>
  func.call @unknown_region_capture_src_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xf32>, memref<16xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @large_copy_vf
// CHECK:      scf.for %arg2 = %c0 to %c1000 step %c64 {
// CHECK-NEXT:   %0 = arith.subi %c1000, %arg2 : index
// CHECK-NEXT:   %1 = arith.cmpi slt, %0, %c64 : index
// CHECK-NEXT:   %2 = arith.select %1, %0, %c64 : index
// CHECK-NEXT:   %3 = vector.create_mask %2 : vector<64xi1>
// CHECK-NEXT:   %4 = vector.transfer_read %arg0[%arg2], %cst, %3 : memref<1000xf32>, vector<64xf32>
// CHECK-NEXT:   vector.transfer_write %4, %arg1[%arg2], %3 : vector<64xf32>, memref<1000xf32>
// CHECK-NEXT: }
func.func @large_copy_vf(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) attributes {hivm.vector_function, no_inline} {
  %init = arith.constant dense<0.000000e+00> : vector<1xf32>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<1000xf32> to memref<1xf32, strided<[1], offset: ?>>
  vector.transfer_write %init, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<1000xf32>) outs(%arg1 : memref<1000xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @large_copy_vf_2_dim
// CHECK:      scf.for %arg2 = %c0 to %c1000 step %c64 {
// CHECK-NEXT:   %0 = arith.subi %c1000, %arg2 : index
// CHECK-NEXT:   %1 = arith.cmpi slt, %0, %c64 : index
// CHECK-NEXT:   %2 = arith.select %1, %0, %c64 : index
// CHECK-NEXT:   %3 = vector.create_mask %c1, %2 : vector<1x64xi1>
// CHECK-NEXT:   %4 = vector.transfer_read %arg0[%c0, %arg2], %cst, %3 {in_bounds = [true, false]} : memref<1x1000xf32>, vector<1x64xf32>
// CHECK-NEXT:   vector.transfer_write %4, %arg1[%c0, %arg2], %3 {in_bounds = [true, false]} : vector<1x64xf32>, memref<1x1000xf32>
// CHECK-NEXT: }
func.func @large_copy_vf_2_dim(%arg0: memref<1x1000xf32>, %arg1: memref<1x1000xf32>) attributes {hivm.vector_function, no_inline} {
  %init = arith.constant dense<0.000000e+00> : vector<1xf32>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0, %c0] [1, 1] [1, 1] : memref<1x1000xf32> to memref<1xf32, strided<[1], offset: ?>>
  vector.transfer_write %init, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<1x1000xf32>) outs(%arg1 : memref<1x1000xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @i1_copy_callee_vf
// CHECK-NOT: hivm.hir.copy
func.func @i1_copy_callee_vf(%arg0: memref<16xi1>, %arg1: memref<16xi1>) attributes {hivm.vector_function, no_inline} {
  %cst = arith.constant dense<false> : vector<1xi1>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<16xi1> to memref<1xi1, strided<[1], offset: ?>>
  vector.transfer_write %cst, %subview[%c0] {in_bounds = [true]} : vector<1xi1>, memref<1xi1, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<16xi1>) outs(%arg1 : memref<16xi1>)
  return
}

func.func @i1_copy_caller() {
  %src = memref.alloc() : memref<16xi1>
  %dst = memref.alloc() : memref<16xi1>
  func.call @i1_copy_callee_vf(%src, %dst) {hivm.vector_function, no_inline} : (memref<16xi1>, memref<16xi1>) -> ()
  return
}

// -----

// An i64 copy that cannot be outlined stays in the VF as a transfer pair. It
// must be tiled by the width one AVE vector operation covers, which for i64 is
// a register pair of 64 elements, because that is the only i64 load the
// template library implements. Tiling by the 32 elements that fit in one VL
// would emit a load that mangles onto the same library symbol as the supported
// width but needs a different signature.
// CHECK-LABEL: func.func @i64_copy_tiled_by_register_pair
// CHECK-NOT: hivm.hir.copy
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C64]]
// CHECK: vector.transfer_read {{.*}} vector<64xi64>
// CHECK: vector.transfer_write {{.*}} vector<64xi64>
func.func @i64_copy_tiled_by_register_pair(%arg0: memref<256xi64>, %arg1: memref<256xi64>)
    attributes {hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0> : vector<1xi64>
  %c0 = arith.constant 0 : index
  // Write through a subview first so the copy is not a pure entry DMA and
  // therefore cannot be hoisted to the caller.
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<256xi64> to memref<1xi64, strided<[1], offset: ?>>
  vector.transfer_write %cst, %subview[%c0] {in_bounds = [true]} : vector<1xi64>, memref<1xi64, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<256xi64>) outs(%arg1 : memref<256xi64>)
  return
}

func.func @i64_copy_tiled_by_register_pair_caller() {
  %src = memref.alloc() : memref<256xi64>
  %dst = memref.alloc() : memref<256xi64>
  func.call @i64_copy_tiled_by_register_pair(%src, %dst) {hivm.vector_function, no_inline} : (memref<256xi64>, memref<256xi64>) -> ()
  return
}

// -----

// Both operands may be reshape chains rooted in VF arguments.
// CHECK-LABEL: func.func @reshape_chain_copy_vf
// CHECK-NOT: hivm.hir.copy
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: hivm.hir.copy
// CHECK: return
func.func @reshape_chain_copy_vf(%src: memref<2x8xf32>, %dst: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  %collapsed = memref.collapse_shape %src [[0, 1]] : memref<2x8xf32> into memref<16xf32>
  %expanded_src = memref.expand_shape %collapsed [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  %expanded_dst = memref.expand_shape %dst [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  hivm.hir.copy ins(%expanded_src : memref<1x16xf32>) outs(%expanded_dst : memref<1x16xf32>)
  return
}

// Keep the copy when an operand traces to a local allocation instead of a VF argument.
// CHECK-LABEL: func.func @reshape_alloc_copy_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @reshape_alloc_copy_vf(%dst: memref<1x16xf32>) attributes {hivm.vector_function, no_inline} {
  %src = memref.alloc() : memref<16xf32>
  %expanded = memref.expand_shape %src [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  hivm.hir.copy ins(%expanded : memref<1x16xf32>) outs(%dst : memref<1x16xf32>)
  return
}

// -----

// Keep DMA copies with an explicit non-UB memory space.
// CHECK-LABEL: func.func @ub_to_l1_reshape_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @ub_to_l1_reshape_vf(%src: memref<16xf32, #hivm.address_space<ub>>, %dst: memref<1x16xf32, #hivm.address_space<cbuf>>) attributes {hivm.vector_function, no_inline} {
  %expanded = memref.expand_shape %src [[0, 1]] output_shape [1, 16] : memref<16xf32, #hivm.address_space<ub>> into memref<1x16xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%expanded : memref<1x16xf32, #hivm.address_space<ub>>) outs(%dst : memref<1x16xf32, #hivm.address_space<cbuf>>)
  return
}

// -----

// A prior write prevents hoisting the ui32 copy out of the VF.
// Lowering reinterprets both buffers as i32 so arith.constant can provide valid transfer padding.
// CHECK-LABEL: func.func @ui32_copy_vf
// CHECK-NOT: hivm.hir.copy
// CHECK-DAG: hivm.hir.bitcast %arg0 : memref<16xui32> -> memref<16xi32>
// CHECK-DAG: hivm.hir.bitcast %arg1 : memref<16xui32> -> memref<16xi32>
// CHECK-DAG: arith.constant 0 : i32
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<64xi32>
// CHECK: vector.transfer_write {{.*}} : vector<64xi32>, memref<16xi32>
// CHECK-NOT: hivm.hir.copy
// CHECK: return
func.func @ui32_copy_vf(%src: memref<16xui32>, %dst: memref<16xui32>, %value: ui32) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  memref.store %value, %dst[%c0] : memref<16xui32>
  hivm.hir.copy ins(%src : memref<16xui32>) outs(%dst : memref<16xui32>)
  return
}

// -----

// A write through B's signless bitcast must prevent hoisting B-to-C.
// The non-unit destination stride keeps B-to-C as a copy.
// CHECK-LABEL: func.func @bitcast_alias_vf
// CHECK: %[[B:.*]] = hivm.hir.bitcast %arg1
// CHECK: vector.transfer_write {{.*}}, %[[B]][
// CHECK: hivm.hir.copy ins(%arg1
// CHECK: return
func.func @bitcast_alias_vf(%a: memref<16xui32>, %b: memref<16xui32>, %c: memref<16xui32, strided<[2]>>) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i32
  %a_signless = hivm.hir.bitcast %a : memref<16xui32> -> memref<16xi32>
  %b_signless = hivm.hir.bitcast %b : memref<16xui32> -> memref<16xi32>
  %value = vector.transfer_read %a_signless[%c0], %zero {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
  vector.transfer_write %value, %b_signless[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
  hivm.hir.copy ins(%b : memref<16xui32>) outs(%c : memref<16xui32, strided<[2]>>)
  return
}

// CHECK-LABEL: func.func @bitcast_alias_caller
// CHECK-NOT: hivm.hir.copy
// CHECK: call @bitcast_alias_vf
// CHECK-NOT: hivm.hir.copy
// CHECK: return
func.func @bitcast_alias_caller(%a: memref<16xui32>, %b: memref<16xui32>, %c: memref<16xui32, strided<[2]>>) {
  func.call @bitcast_alias_vf(%a, %b, %c) {hivm.vector_function, no_inline} : (memref<16xui32>, memref<16xui32>, memref<16xui32, strided<[2]>>) -> ()
  return
}

// -----

// The unknown write prevents argument-to-argument copies from being outlined.
// CHECK-LABEL: func.func @dynamic_src_stride_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @dynamic_src_stride_vf(%src: memref<16xf32, strided<[?], offset: ?>>, %dst: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<16xf32>) -> ()
  hivm.hir.copy ins(%src : memref<16xf32, strided<[?], offset: ?>>) outs(%dst : memref<16xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @dynamic_dst_stride_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @dynamic_dst_stride_vf(%src: memref<16xf32>, %dst: memref<16xf32, strided<[?], offset: ?>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<16xf32, strided<[?], offset: ?>>) -> ()
  hivm.hir.copy ins(%src : memref<16xf32>) outs(%dst : memref<16xf32, strided<[?], offset: ?>>)
  return
}

// Unit lane stride remains legal with static or dynamic offsets.
// CHECK-LABEL: func.func @static_offset_unit_stride_vf
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: hivm.hir.copy
func.func @static_offset_unit_stride_vf(%src: memref<16xf32, strided<[1], offset: 4>>, %dst: memref<16xf32, strided<[1], offset: 4>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<16xf32, strided<[1], offset: 4>>) -> ()
  hivm.hir.copy ins(%src : memref<16xf32, strided<[1], offset: 4>>) outs(%dst : memref<16xf32, strided<[1], offset: 4>>)
  return
}

// CHECK-LABEL: func.func @dynamic_offset_unit_stride_vf
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: hivm.hir.copy
func.func @dynamic_offset_unit_stride_vf(%src: memref<16xf32, strided<[1], offset: ?>>, %dst: memref<16xf32, strided<[1], offset: ?>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<16xf32, strided<[1], offset: ?>>) -> ()
  hivm.hir.copy ins(%src : memref<16xf32, strided<[1], offset: ?>>) outs(%dst : memref<16xf32, strided<[1], offset: ?>>)
  return
}

// Non-unit lane stride remains unsupported.
// CHECK-LABEL: func.func @static_nonunit_stride_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
func.func @static_nonunit_stride_vf(%src: memref<16xf32, strided<[2], offset: 4>>, %dst: memref<16xf32, strided<[2], offset: 4>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<16xf32, strided<[2], offset: 4>>) -> ()
  hivm.hir.copy ins(%src : memref<16xf32, strided<[2], offset: 4>>) outs(%dst : memref<16xf32, strided<[2], offset: 4>>)
  return
}

// -----

// Dynamic outer strides are not supported by the AVE stride-store lowering.
// The preceding write prevents caller-side hoisting, so these checks exercise
// transfer legality for the source and destination separately.
// CHECK-LABEL: func.func @dynamic_outer_src_stride_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @dynamic_outer_src_stride_vf(%src: memref<2x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, %dst: memref<2x8xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<2x8xf32, #hivm.address_space<ub>>) -> ()
  hivm.hir.copy ins(%src : memref<2x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) outs(%dst : memref<2x8xf32, #hivm.address_space<ub>>)
  return
}

// CHECK-LABEL: func.func @dynamic_outer_dst_stride_vf
// CHECK-NOT: vector.transfer
// CHECK: hivm.hir.copy
// CHECK-NOT: vector.transfer
// CHECK: return
func.func @dynamic_outer_dst_stride_vf(%src: memref<2x8xf32, #hivm.address_space<ub>>, %dst: memref<2x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%dst) : (memref<2x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>) -> ()
  hivm.hir.copy ins(%src : memref<2x8xf32, #hivm.address_space<ub>>) outs(%dst : memref<2x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>)
  return
}

// -----

// A write through the dst argument does not hoist the view copy. The copy is
// lowered at its original position, after the write.
// CHECK-LABEL: func.func @view_copy_after_dst_write_vf
// CHECK: memref.store
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: hivm.hir.copy
// CHECK: return
func.func @view_copy_after_dst_write_vf(%src: memref<2x8xf32>, %dst: memref<16xf32>, %value: f32) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  memref.store %value, %dst[%c0] : memref<16xf32>
  %collapsed = memref.collapse_shape %src [[0, 1]] : memref<2x8xf32> into memref<16xf32>
  %expanded_src = memref.expand_shape %collapsed [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  %expanded_dst = memref.expand_shape %dst [[0, 1]] output_shape [1, 16] : memref<16xf32> into memref<1x16xf32>
  hivm.hir.copy ins(%expanded_src : memref<1x16xf32>) outs(%expanded_dst : memref<1x16xf32>)
  return
}

// A subview copy following a producer write is lowered in place.
// CHECK-LABEL: func.func @subview_copy_after_write_vf
// CHECK: memref.store
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK-NOT: hivm.hir.copy
// CHECK: return
func.func @subview_copy_after_write_vf(%src: memref<16xf32>, %dst: memref<16xf32>, %value: f32) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  memref.store %value, %src[%c0] : memref<16xf32>
  %src_view = memref.subview %src[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  %dst_view = memref.subview %dst[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%src_view : memref<4xf32, strided<[1], offset: ?>>) outs(%dst_view : memref<4xf32, strided<[1], offset: ?>>)
  return
}
