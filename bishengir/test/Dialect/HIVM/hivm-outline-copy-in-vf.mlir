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
