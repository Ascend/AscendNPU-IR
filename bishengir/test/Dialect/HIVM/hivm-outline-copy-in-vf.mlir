// RUN: bishengir-opt %s -hivm-outline-copy-in-VF -split-input-file -allow-unregistered-dialect | FileCheck %s

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
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[SUBVIEW:.*]] = memref.subview %arg1[%[[C0]]] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK: vector.transfer_write %[[CST]], %[[SUBVIEW]][%[[C0]]] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
// CHECK: %[[C0_LOWERED:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %arg0[%[[C0_LOWERED]]], %[[PAD]] {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %arg1[%[[C0_LOWERED]]] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
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
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %arg0[%[[C0]]] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK: %[[DST_SUBVIEW:.*]] = memref.subview %arg1[%[[C0]]] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK: %[[C0_LOWERED:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %[[SRC_SUBVIEW]][%[[C0_LOWERED]]], %[[PAD]] {in_bounds = [true]} : memref<4xf32, strided<[1], offset: ?>>, vector<4xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %{{.*}}[%[[C0_LOWERED]]] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>
func.func @subview_load_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %src = memref.subview %arg0[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  %dst = memref.subview %arg1[%c0] [4] [1] : memref<16xf32> to memref<4xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%src : memref<4xf32, strided<[1], offset: ?>>) outs(%dst : memref<4xf32, strided<[1], offset: ?>>)
  return
}

// -----

// CHECK-LABEL: func.func @unknown_write_vf
// CHECK: "test.unknown_write"(%arg1) : (memref<16xf32>) -> ()
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %arg0[%[[C0]]], %[[PAD]] {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %arg1[%[[C0]]] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
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
// CHECK: scf.if %{{.*}} {
// CHECK:   "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
// CHECK: }
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %arg0[%[[C0]]], %[[PAD]] {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %arg1[%[[C0]]] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
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
// CHECK: "test.unknown_region"() ({
// CHECK:   "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
// CHECK: }) : () -> ()
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %arg0[%[[C0]]], %[[PAD]] {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %arg1[%[[C0]]] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
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
// CHECK: "test.unknown_region"() ({
// CHECK:   "test.unknown_nested_write"(%arg0) : (memref<16xf32>) -> ()
// CHECK: }) : () -> ()
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[READ:.*]] = vector.transfer_read %arg0[%[[C0]]], %[[PAD]] {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.transfer_write %[[READ]], %arg1[%[[C0]]] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
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
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1000:.*]] = arith.constant 1000 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %[[C1000]] step %[[C64]] {
// CHECK:   %[[REMAINING:.*]] = arith.subi %[[C1000]], %[[IV]] : index
// CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[REMAINING]], %[[C64]] : index
// CHECK:   %[[CHUNK:.*]] = arith.select %[[CMP]], %[[REMAINING]], %[[C64]] : index
// CHECK:   %[[MASK:.*]] = vector.create_mask %[[CHUNK]] : vector<64xi1>
// CHECK:   %[[READ:.*]] = vector.transfer_read %arg0[%[[IV]]], %{{.*}}, %[[MASK]] : memref<1000xf32>, vector<64xf32>
// CHECK:   vector.transfer_write %[[READ]], %arg1[%[[IV]]], %[[MASK]] : vector<64xf32>, memref<1000xf32>
func.func @large_copy_vf(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>) attributes {hivm.vector_function, no_inline} {
  %init = arith.constant dense<0.000000e+00> : vector<1xf32>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<1000xf32> to memref<1xf32, strided<[1], offset: ?>>
  vector.transfer_write %init, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  hivm.hir.copy ins(%arg0 : memref<1000xf32>) outs(%arg1 : memref<1000xf32>)
  return
}
