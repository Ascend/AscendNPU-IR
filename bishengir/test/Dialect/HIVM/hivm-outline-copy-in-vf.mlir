// RUN: bishengir-opt %s -hivm-outline-copy-in-VF -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----

// CHECK-LABEL: func.func @outlined_vf
// CHECK-NOT: hivm.hir.load
func.func @outlined_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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
// CHECK-NEXT: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
func.func @guarded_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[%c0] [1] [1] : memref<16xf32> to memref<1xf32, strided<[1], offset: ?>>
  vector.transfer_write %cst, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>>
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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

// CHECK-LABEL: func.func @unknown_write_vf
// CHECK: "test.unknown_write"(%arg1) : (memref<16xf32>) -> ()
// CHECK-NEXT: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
func.func @unknown_write_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_write"(%arg1) : (memref<16xf32>) -> ()
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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
// CHECK: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
func.func @recursive_unknown_write_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %flag: i1) attributes {hivm.vector_function, no_inline} {
  scf.if %flag {
    "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
  }
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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
// CHECK: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
func.func @unknown_region_capture_dst_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_region"() ({
    "test.unknown_nested_write"(%arg1) : (memref<16xf32>) -> ()
  }) : () -> ()
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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
// CHECK: hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
func.func @unknown_region_capture_src_vf(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hivm.vector_function, no_inline} {
  "test.unknown_region"() ({
    "test.unknown_nested_write"(%arg0) : (memref<16xf32>) -> ()
  }) : () -> ()
  hivm.hir.load ins(%arg0 : memref<16xf32>) outs(%arg1 : memref<16xf32>) eviction_policy = <EvictFirst>
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
