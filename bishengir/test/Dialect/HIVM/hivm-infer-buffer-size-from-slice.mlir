// RUN: bishengir-opt %s -hivm-infer-buffer-size-from-slice -allow-unregistered-dialect -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hivm-infer-buffer-size-from-slice -hivm-set-buffer-size -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefix=CHAIN

// hivm-infer-buffer-size-from-slice marks the dst of every copy-like op
// with a buffer_size_in_byte upper bound derived from the op's src defining
// chain (256 * 16bit = 512 bytes).
// CHECK-LABEL: func.func @infer_copy_dst_static_src(
func.func @infer_copy_dst_static_src(%arg0: memref<256xf16>, %arg1: index) {
  %dst = memref.alloc(%arg1) : memref<?xf16>
  // CHECK: annotation.mark %[[DST:.*]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHECK: hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%[[DST]] : memref<?xf16>)
  // CHAIN: %[[CONST:.*]] = arith.constant 0 : index
  // CHAIN: %[[ALLOC:.*]] = memref.alloc() : memref<512xi8>
  // CHAIN: %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[CONST]]][%arg1] : memref<512xi8> to memref<?xf16>
  // CHAIN: hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%[[VIEW]] : memref<?xf16>)
  hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%dst : memref<?xf16>)
  return
}

// -----

// The bound also follows a subview in the copy src's defining chain.
// CHECK-LABEL: func.func @infer_copy_dst_subview_src(
func.func @infer_copy_dst_subview_src(%arg0: memref<256xf16>, %arg1: index) {
  %sv = memref.subview %arg0[0] [%arg1] [1] : memref<256xf16> to memref<?xf16>
  %dst = memref.alloc(%arg1) : memref<?xf16>
  // CHECK: annotation.mark %[[DST:.*]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHECK: hivm.hir.copy ins({{.*}} : memref<?xf16>) outs(%[[DST]] : memref<?xf16>)
  hivm.hir.copy ins(%sv : memref<?xf16>) outs(%dst : memref<?xf16>)
  return
}

// -----

// The dst of an `hivm.hir.load` is marked the same way as a copy dst.
// CHECK-LABEL: func.func @infer_load_dst_static_src(
func.func @infer_load_dst_static_src(%arg0: memref<256xf16, #hivm.address_space<gm>>, %arg1: index) {
  %dst = memref.alloc(%arg1) : memref<?xf16, #hivm.address_space<ub>>
  // CHECK: annotation.mark %[[DST:.*]] {buffer_size_in_byte = 512 : i64} : memref<?xf16, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load ins(%arg0 : memref<256xf16, #hivm.address_space<gm>>) outs(%[[DST]] : memref<?xf16, #hivm.address_space<ub>>)
  // CHAIN: %[[CONST:.*]] = arith.constant 0 : index
  // CHAIN: %[[ALLOC:.*]] = memref.alloc() : memref<512xi8, #hivm.address_space<ub>>
  // CHAIN: %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[CONST]]][%arg1] : memref<512xi8, #hivm.address_space<ub>> to memref<?xf16, #hivm.address_space<ub>>
  // CHAIN: hivm.hir.load ins(%arg0 : memref<256xf16, #hivm.address_space<gm>>) outs(%[[VIEW]] : memref<?xf16, #hivm.address_space<ub>>)
  hivm.hir.load ins(%arg0 : memref<256xf16, #hivm.address_space<gm>>) outs(%dst : memref<?xf16, #hivm.address_space<ub>>)
  return
}

// -----

// A dst that already carries a buffer_size_in_byte mark keeps it; no
// second (chain-derived) mark is added.
// CHECK-LABEL: func.func @infer_copy_dst_already_marked(
func.func @infer_copy_dst_already_marked(%arg0: memref<256xf16>, %arg1: index) {
  %dst = memref.alloc(%arg1) : memref<?xf16>
  annotation.mark %dst {buffer_size_in_byte = 999 : i64} : memref<?xf16>
  // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 999 : i64} : memref<?xf16>
  // CHECK-NOT: annotation.mark %{{.*}} {buffer_size_in_byte = 512 : i64}
  hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%dst : memref<?xf16>)
  return
}

// -----

// A statically shaped dst needs no bound.
// CHECK-LABEL: func.func @infer_copy_dst_static_shape(
func.func @infer_copy_dst_static_shape(%arg0: memref<256xf16>) {
  %dst = memref.alloc() : memref<256xf16>
  // CHECK-NOT: annotation.mark
  hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%dst : memref<256xf16>)
  return
}

// -----

// A src that already carries a buffer_size_in_byte mark propagates it to
// the dst directly (the cloned mark wins over the chain-derived bound), and
// the pattern stays idempotent: exactly one mark is added.
// CHECK-LABEL: func.func @infer_copy_src_marked(
func.func @infer_copy_src_marked(%arg0: index) {
  %src = memref.alloc(%arg0) : memref<?xf16>
  annotation.mark %src {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  %dst = memref.alloc(%arg0) : memref<?xf16>
  // CHECK: annotation.mark %[[DST:.*]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHECK-NOT: annotation.mark %[[DST]] {buffer_size_in_byte = 512 : i64} : memref<?xf16>
  // CHAIN: %[[CONST:.*]] = arith.constant 0 : index
  // CHAIN: %[[ALLOC:.*]] = memref.alloc() : memref<512xi8>
  // CHAIN: %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[CONST]]][%arg0] : memref<512xi8> to memref<?xf16>
  // CHAIN: hivm.hir.copy ins(%[[VIEW]] : memref<?xf16>) outs({{.*}} : memref<?xf16>)
  hivm.hir.copy ins(%src : memref<?xf16>) outs(%dst : memref<?xf16>)
  return
}

// -----

// A dst that is not a direct memref.alloc result (e.g. a subview) is not
// marked; its defining allocation owns the size.
// CHECK-LABEL: func.func @infer_copy_dst_subview(
func.func @infer_copy_dst_subview(%arg0: memref<256xf16>, %arg1: index) {
  %base = memref.alloc(%arg1) : memref<?xf16>
  %dst = memref.subview %base[0] [%arg1] [1] : memref<?xf16> to memref<?xf16>
  // CHECK-NOT: annotation.mark
  hivm.hir.copy ins(%arg0 : memref<256xf16>) outs(%dst : memref<?xf16>)
  return
}

// -----

// A src without a statically-shaped ancestor in its defining chain (e.g.
// a block argument) yields no bound.
// CHECK-LABEL: func.func @infer_copy_src_no_static_bound(
func.func @infer_copy_src_no_static_bound(%arg0: memref<?xf16>, %arg1: index) {
  %dst = memref.alloc(%arg1) : memref<?xf16>
  // CHECK-NOT: annotation.mark
  hivm.hir.copy ins(%arg0 : memref<?xf16>) outs(%dst : memref<?xf16>)
  return
}
