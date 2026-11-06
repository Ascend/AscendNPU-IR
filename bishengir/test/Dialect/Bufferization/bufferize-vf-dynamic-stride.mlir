// RUN: bishengir-opt -split-input-file \
// RUN:   -tensor-copy-insertion="analysis-heuristic=top-down allow-return-allocs-from-loops bufferize-function-boundaries" \
// RUN:   -inline-func-to-scope \
// RUN:   -bufferize-module-op="function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" \
// RUN:   -canonicalize -drop-equivalent-buffer-results \
// RUN:   -outline-scope="outline-marked-scopes-only=true" \
// RUN:   -hivm-verify-vector-function-top-level-copy %s | FileCheck %s

// End-to-end coverage of the RegBase bufferization flow for vector functions
// fed by dynamic strided slices: the VF is inlined into a scope before
// bufferization, so the exact (dynamic stride / offset) layout of the
// caller-side subview is inferred, and the VF is outlined again afterwards
// with that layout kept in its signature. The slices are taken from a static
// tensor.empty (never from a block argument), which bufferizes to a single
// static allocation. No identity-layout copy is inserted around the call, and
// the final VF top-level-copy check passes.

// -----

// Dynamic offset and dynamic stride on a 1-D slice.
// CHECK-LABEL: func.func @vf(
// CHECK-SAME: %[[ARG:.*]]: memref<8xf32, strided<[?], offset: ?>>)
// CHECK-SAME: -> memref<8xf32, strided<[?], offset: ?>>
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK: memref.load %[[ARG]]
// CHECK: memref.store {{.*}}, %[[ARG]]
// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %[[OFF:.*]]: index, %[[STRIDE:.*]]: index) -> memref<64xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: %[[SV:.*]] = memref.subview %[[BUF]][%[[OFF]]] [8] [%[[STRIDE]]]
// CHECK-SAME: to memref<8xf32, strided<[?], offset: ?>>
// CHECK: %[[RES:.*]] = call @vf(%[[SV]]) {hivm.vector_function}
// CHECK-SAME: (memref<8xf32, strided<[?], offset: ?>>) -> memref<8xf32, strided<[?], offset: ?>>
// CHECK: %[[DST:.*]] = memref.subview %[[BUF]][%[[OFF]]] [8] [%[[STRIDE]]]
// CHECK: memref.copy %[[RES]], %[[DST]]
module {
  func.func @vf(%arg0: tensor<8xf32>) -> tensor<8xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%t = %arg0) -> (tensor<8xf32>) {
      %v = tensor.extract %t[%i] : tensor<8xf32>
      %a = arith.addf %v, %cst : f32
      %n = tensor.insert %a into %t[%i] : tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    }
    return %r : tensor<8xf32>
  }
  func.func @caller(%off: index, %stride: index) -> tensor<64xf32> {
    %e = tensor.empty() : tensor<64xf32>
    %s = tensor.extract_slice %e[%off] [8] [%stride] : tensor<64xf32> to tensor<8xf32>
    %r = func.call @vf(%s) {hivm.vector_function} : (tensor<8xf32>) -> tensor<8xf32>
    %o = tensor.insert_slice %r into %e[%off] [8] [%stride] : tensor<8xf32> into tensor<64xf32>
    return %o : tensor<64xf32>
  }
}

// -----

// 2-D slice: dynamic row stride, unit column stride, dynamic offset. The
// inner unit stride is preserved in the layout.
// CHECK-LABEL: func.func @vf2d(
// CHECK-SAME: %[[ARG:.*]]: memref<4x8xf32, strided<[?, 1], offset: ?>>)
// CHECK-SAME: -> memref<4x8xf32, strided<[?, 1], offset: ?>>
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK: memref.load %[[ARG]]
// CHECK: memref.store {{.*}}, %[[ARG]]
// CHECK-LABEL: func.func @caller2d(
// CHECK-SAME: %[[O0:.*]]: index, %[[O1:.*]]: index, %[[S:.*]]: index) -> memref<32x32xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<32x32xf32>
// CHECK: %[[SV:.*]] = memref.subview %[[BUF]][%[[O0]], %[[O1]]] [4, 8] [%[[S]], 1]
// CHECK-SAME: to memref<4x8xf32, strided<[?, 1], offset: ?>>
// CHECK: call @vf2d(%[[SV]]) {hivm.vector_function}
// CHECK-SAME: (memref<4x8xf32, strided<[?, 1], offset: ?>>) -> memref<4x8xf32, strided<[?, 1], offset: ?>>
module {
  func.func @vf2d(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 2.0 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %arg0) -> (tensor<4x8xf32>) {
      %v = tensor.extract %t[%i, %c0] : tensor<4x8xf32>
      %a = arith.mulf %v, %cst : f32
      %n = tensor.insert %a into %t[%i, %c0] : tensor<4x8xf32>
      scf.yield %n : tensor<4x8xf32>
    }
    return %r : tensor<4x8xf32>
  }
  func.func @caller2d(%o0: index, %o1: index, %s: index) -> tensor<32x32xf32> {
    %e = tensor.empty() : tensor<32x32xf32>
    %x = tensor.extract_slice %e[%o0, %o1] [4, 8] [%s, 1] : tensor<32x32xf32> to tensor<4x8xf32>
    %r = func.call @vf2d(%x) {hivm.vector_function} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %o = tensor.insert_slice %r into %e[%o0, %o1] [4, 8] [%s, 1] : tensor<4x8xf32> into tensor<32x32xf32>
    return %o : tensor<32x32xf32>
  }
}

// -----

// The same VF called at two sites with different layouts (dynamic offset +
// dynamic stride vs. static zero offset + dynamic stride, each sliced from its
// own tensor.empty). Each call site is
// inlined and outlined independently, so each outlined function keeps the
// exact layout of its own caller instead of being merged into one function
// with a common, more dynamic layout.
// CHECK-LABEL: func.func @vf(
// CHECK-SAME: %{{.*}}: memref<8xf32, strided<[?]>>)
// CHECK-SAME: -> memref<8xf32, strided<[?]>>
// CHECK-LABEL: func.func @two_sites_scope_0(
// CHECK-SAME: %{{.*}}: memref<8xf32, strided<[?], offset: ?>>)
// CHECK-SAME: -> memref<8xf32, strided<[?], offset: ?>>
// CHECK-LABEL: func.func @two_sites(
// CHECK-SAME: %[[OFF:.*]]: index, %[[S0:.*]]: index, %[[S1:.*]]: index) -> memref<64xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: %[[A:.*]] = memref.subview %[[BUF]][%[[OFF]]] [8] [%[[S0]]]
// CHECK-SAME: to memref<8xf32, strided<[?], offset: ?>>
// CHECK: call @two_sites_scope_0(%[[A]])
// CHECK: %[[BUF2:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: %[[B:.*]] = memref.subview %[[BUF2]][0] [8] [%[[S1]]]
// CHECK-SAME: to memref<8xf32, strided<[?]>>
// CHECK: call @vf(%[[B]])
// CHECK-SAME: (memref<8xf32, strided<[?]>>) -> memref<8xf32, strided<[?]>>
module {
  func.func @vf(%arg0: tensor<8xf32>) -> tensor<8xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%t = %arg0) -> (tensor<8xf32>) {
      %v = tensor.extract %t[%i] : tensor<8xf32>
      %a = arith.addf %v, %cst : f32
      %n = tensor.insert %a into %t[%i] : tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    }
    return %r : tensor<8xf32>
  }
  func.func @two_sites(%off: index, %s0: index, %s1: index) -> tensor<64xf32> {
    %e = tensor.empty() : tensor<64xf32>
    %a = tensor.extract_slice %e[%off] [8] [%s0] : tensor<64xf32> to tensor<8xf32>
    %ra = func.call @vf(%a) {hivm.vector_function} : (tensor<8xf32>) -> tensor<8xf32>
    %o = tensor.insert_slice %ra into %e[%off] [8] [%s0] : tensor<8xf32> into tensor<64xf32>
    %e2 = tensor.empty() : tensor<64xf32>
    %b = tensor.extract_slice %e2[0] [8] [%s1] : tensor<64xf32> to tensor<8xf32>
    %rb = func.call @vf(%b) {hivm.vector_function} : (tensor<8xf32>) -> tensor<8xf32>
    %o2 = tensor.insert_slice %rb into %o[0] [8] [%s1] : tensor<8xf32> into tensor<64xf32>
    return %o2 : tensor<64xf32>
  }
}

// -----

// The slice offset is computed per iteration (loop induction variable) and the
// stride is a function argument; the call lives inside the loop body and its
// slice comes from a loop-local tensor.empty. The layout is still exact, and
// the result is written back into the loop-carried buffer.
// CHECK-LABEL: func.func @vf(
// CHECK-SAME: %{{.*}}: memref<8xf32, strided<[?], offset: ?>>)
// CHECK-SAME: -> memref<8xf32, strided<[?], offset: ?>>
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK-LABEL: func.func @loop(
// CHECK-SAME: %[[N:.*]]: index, %[[S:.*]]: index) -> memref<64xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: scf.for %[[IV:.*]] = {{.*}} to %[[N]]
// CHECK: %[[TMP:.*]] = memref.alloc() {{.*}} : memref<64xf32>
// CHECK: %[[SV:.*]] = memref.subview %[[TMP]][%[[IV]]] [8] [%[[S]]]
// CHECK-SAME: to memref<8xf32, strided<[?], offset: ?>>
// CHECK: %[[RES:.*]] = {{(func\.)?}}call @vf(%[[SV]]) {hivm.vector_function}
// CHECK: %[[DST:.*]] = memref.subview %[[BUF]][%[[IV]]] [8] [%[[S]]]
// CHECK: memref.copy %[[RES]], %[[DST]]
module {
  func.func @vf(%arg0: tensor<8xf32>) -> tensor<8xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%t = %arg0) -> (tensor<8xf32>) {
      %v = tensor.extract %t[%i] : tensor<8xf32>
      %a = arith.addf %v, %cst : f32
      %n = tensor.insert %a into %t[%i] : tensor<8xf32>
      scf.yield %n : tensor<8xf32>
    }
    return %r : tensor<8xf32>
  }
  func.func @loop(%n: index, %s: index) -> tensor<64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %e = tensor.empty() : tensor<64xf32>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %e) -> (tensor<64xf32>) {
      %t = tensor.empty() : tensor<64xf32>
      %x = tensor.extract_slice %t[%i] [8] [%s] : tensor<64xf32> to tensor<8xf32>
      %y = func.call @vf(%x) {hivm.vector_function} : (tensor<8xf32>) -> tensor<8xf32>
      %o = tensor.insert_slice %y into %acc[%i] [8] [%s] : tensor<8xf32> into tensor<64xf32>
      scf.yield %o : tensor<64xf32>
    }
    return %r : tensor<64xf32>
  }
}

// -----

// Fully static slice parameters: 126x32 rows taken from the head of a 128x32
// buffer. Offsets are zero, so the layout has the row stride 32 and no offset.
// CHECK-LABEL: func.func @vf_head(
// CHECK-SAME: %[[ARG:.*]]: memref<126x32xf32, strided<[32, 1]>>)
// CHECK-SAME: -> memref<126x32xf32, strided<[32, 1]>>
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK-LABEL: func.func @caller_head(
// CHECK-SAME: -> memref<128x32xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<128x32xf32>
// CHECK: %[[SV:.*]] = memref.subview %[[BUF]][0, 0] [126, 32] [1, 1]
// CHECK-SAME: to memref<126x32xf32, strided<[32, 1]>>
// CHECK: %[[RES:.*]] = call @vf_head(%[[SV]]) {hivm.vector_function}
// CHECK-SAME: (memref<126x32xf32, strided<[32, 1]>>) -> memref<126x32xf32, strided<[32, 1]>>
// CHECK: %[[DST:.*]] = memref.subview %[[BUF]][0, 0] [126, 32] [1, 1]
// CHECK: memref.copy %[[RES]], %[[DST]]
module {
  func.func @vf_head(%arg0: tensor<126x32xf32>) -> tensor<126x32xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %cn = arith.constant 126 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %cn step %c1 iter_args(%t = %arg0) -> (tensor<126x32xf32>) {
      %v = tensor.extract %t[%i, %c0] : tensor<126x32xf32>
      %a = arith.addf %v, %cst : f32
      %n = tensor.insert %a into %t[%i, %c0] : tensor<126x32xf32>
      scf.yield %n : tensor<126x32xf32>
    }
    return %r : tensor<126x32xf32>
  }
  func.func @caller_head() -> tensor<128x32xf32> {
    %e = tensor.empty() : tensor<128x32xf32>
    %s = tensor.extract_slice %e[0, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
    %r = func.call @vf_head(%s) {hivm.vector_function} : (tensor<126x32xf32>) -> tensor<126x32xf32>
    %o = tensor.insert_slice %r into %e[0, 0] [126, 32] [1, 1] : tensor<126x32xf32> into tensor<128x32xf32>
    return %o : tensor<128x32xf32>
  }
}

// -----

// Fully static slice parameters with a row offset: rows [2, 128) of a 128x32
// buffer. The static offset 2 * 32 = 64 is folded into the layout.
// CHECK-LABEL: func.func @vf_rows(
// CHECK-SAME: %[[ARG:.*]]: memref<126x32xf32, strided<[32, 1], offset: 64>>)
// CHECK-SAME: -> memref<126x32xf32, strided<[32, 1], offset: 64>>
// CHECK-SAME: attributes {hivm.vector_function}
// CHECK-LABEL: func.func @caller_rows(
// CHECK-SAME: -> memref<128x32xf32>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<128x32xf32>
// CHECK: %[[SV:.*]] = memref.subview %[[BUF]][2, 0] [126, 32] [1, 1]
// CHECK-SAME: to memref<126x32xf32, strided<[32, 1], offset: 64>>
// CHECK: %[[RES:.*]] = call @vf_rows(%[[SV]]) {hivm.vector_function}
// CHECK-SAME: (memref<126x32xf32, strided<[32, 1], offset: 64>>) -> memref<126x32xf32, strided<[32, 1], offset: 64>>
// CHECK: %[[DST:.*]] = memref.subview %[[BUF]][2, 0] [126, 32] [1, 1]
// CHECK: memref.copy %[[RES]], %[[DST]]
module {
  func.func @vf_rows(%arg0: tensor<126x32xf32>) -> tensor<126x32xf32> attributes {hivm.vector_function} {
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %cn = arith.constant 126 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %cn step %c1 iter_args(%t = %arg0) -> (tensor<126x32xf32>) {
      %v = tensor.extract %t[%i, %c0] : tensor<126x32xf32>
      %a = arith.addf %v, %cst : f32
      %n = tensor.insert %a into %t[%i, %c0] : tensor<126x32xf32>
      scf.yield %n : tensor<126x32xf32>
    }
    return %r : tensor<126x32xf32>
  }
  func.func @caller_rows() -> tensor<128x32xf32> {
    %e = tensor.empty() : tensor<128x32xf32>
    %s = tensor.extract_slice %e[2, 0] [126, 32] [1, 1] : tensor<128x32xf32> to tensor<126x32xf32>
    %r = func.call @vf_rows(%s) {hivm.vector_function} : (tensor<126x32xf32>) -> tensor<126x32xf32>
    %o = tensor.insert_slice %r into %e[2, 0] [126, 32] [1, 1] : tensor<126x32xf32> into tensor<128x32xf32>
    return %o : tensor<128x32xf32>
  }
}
