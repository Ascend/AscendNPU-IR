// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefixes=CHECK,SEPARATE
// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefixes=CHECK,SEPARATE
// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-merge-same-preload-scopes="bypass-shape-registry=true" | FileCheck %s --check-prefixes=CHECK,MERGED
// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" | FileCheck %s --check-prefix=OFF
// RUN: sed 's/hivm.preload_num = 0/hivm.preload_num = 1/g' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=LOOKAHEAD
// RUN: sed 's/hivm.preload_num = 0/hivm.preload_num = 1/g' %s | bishengir-opt -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s --check-prefix=LOOKAHEAD

// A copy-independent multiply moves to the existing same-stage consumer.
// A shared static tensor.empty is recreated there instead of being exported.
// CHECK-LABEL: func.func @reuse_consumer
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK-NOT: hivm.hir.vmul
// CHECK: hivm.hir.copy
// CHECK: scope.return
// CHECK: } {{.*}}hivm.independent_preload_prefix
// CHECK: scope.scope
// CHECK: tensor.empty
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vsub
// LOOKAHEAD-LABEL: func.func @reuse_consumer
// LOOKAHEAD: hivm.hir.vadd
// LOOKAHEAD: hivm.hir.vmul
// LOOKAHEAD: hivm.hir.copy
// LOOKAHEAD: } {{.*}}hivm.preload_num = 1
// OFF-LABEL: func.func @reuse_consumer
// OFF: hivm.hir.vadd
// OFF: hivm.hir.vmul
// OFF: hivm.hir.copy
func.func @reuse_consumer(%p: tensor<16xf16>, %q: tensor<16xf16>, %dst: memref<16xf16, #hivm.address_space<cbuf>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %r = scope.scope : () -> tensor<16xf16> {
      %e = tensor.empty() : tensor<16xf16>
      %copy = hivm.hir.vadd ins(%p, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      %extra = hivm.hir.vmul ins(%q, %q : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      hivm.hir.copy ins(%copy : tensor<16xf16>) outs(%dst : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %extra : tensor<16xf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<16xf16>
      %next = hivm.hir.vsub ins(%r, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Both copies retain their full dependencies. A new same-stage scope runs
// unrelated work after the copies but before the following cube scope.
// CHECK-LABEL: func.func @two_copies
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.copy
// CHECK-NOT: hivm.hir.vmul
// CHECK: } {{.*}}hivm.independent_preload_prefix
// CHECK: scope.scope
// CHECK: hivm.hir.vmul
// CHECK: } {{.*}}hivm.independent_preload_prefix{{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK: } {{.*}}hivm.tcore_type<CUBE>
func.func @two_copies(%p: tensor<16xf16>, %q: tensor<16xf16>, %a: memref<16xf16, #hivm.address_space<cbuf>>, %b: memref<16xf16, #hivm.address_space<cbuf>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %q) -> tensor<16xf16> {
    %r = scope.scope : () -> tensor<16xf16> {
      %e = tensor.empty() : tensor<16xf16>
      %copy = hivm.hir.vadd ins(%p, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      hivm.hir.copy ins(%copy : tensor<16xf16>) outs(%a : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %extra = hivm.hir.vmul ins(%state, %state : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      %copy2 = hivm.hir.vexp ins(%copy : tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      hivm.hir.copy ins(%copy2 : tensor<16xf16>) outs(%b : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %extra : tensor<16xf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %r : tensor<16xf16>
  }
  return
}

// The first prefix has no copy. Follow scope results to retain the add that
// feeds the later copy, and move only the unrelated multiply past that copy.
// CHECK-LABEL: func.func @forwarded_copy_dependencies
// CHECK: scope.scope
// CHECK: hivm.hir.vadd
// CHECK-NOT: hivm.hir.vmul
// CHECK: } {{.*}}hivm.independent_preload_prefix
// CHECK: scope.scope
// CHECK: hivm.hir.copy
// CHECK: } {{.*}}hivm.independent_preload_prefix
// CHECK: scope.scope
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vsub
func.func @forwarded_copy_dependencies(%p: tensor<16xf16>, %q: tensor<16xf16>, %dst: memref<16xf16, #hivm.address_space<cbuf>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %q) -> tensor<16xf16> {
    %r:2 = scope.scope : () -> (tensor<16xf16>, tensor<16xf16>) {
      %e = tensor.empty() : tensor<16xf16>
      %copy = hivm.hir.vadd ins(%p, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      %extra = hivm.hir.vmul ins(%state, %state : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      scope.return %copy, %extra : tensor<16xf16>, tensor<16xf16>
    } {hivm.independent_preload_prefix, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    %next = scope.scope : () -> tensor<16xf16> {
      %e = tensor.empty() : tensor<16xf16>
      %extra = hivm.hir.vsub ins(%r#1, %q : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      hivm.hir.copy ins(%r#0 : tensor<16xf16>) outs(%dst : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %extra : tensor<16xf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next : tensor<16xf16>
  }
  return
}

// A tensor view of mutable storage cannot be delayed, even through bitcast.
// CHECK-LABEL: func.func @preserve_mutable_read
// CHECK: hivm.hir.vadd
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.bitcast
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.copy
// CHECK: scope.return
func.func @preserve_mutable_read(%p: tensor<16xf16>, %q: tensor<16xf16>, %dst: memref<16xf16, #hivm.address_space<cbuf>>, %buf: memref<16xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %r = scope.scope : () -> tensor<16xf16> {
      %e = tensor.empty() : tensor<16xf16>
      %copy = hivm.hir.vadd ins(%p, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      %view = bufferization.to_tensor %buf restrict writable : memref<16xf16>
      %cast = hivm.hir.bitcast %view : tensor<16xf16> -> tensor<16xf16>
      %extra = hivm.hir.vmul ins(%cast, %q : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      hivm.hir.copy ins(%copy : tensor<16xf16>) outs(%dst : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %extra : tensor<16xf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<16xf16>
      %next = hivm.hir.vsub ins(%r, %p : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// An intermediate scope return is still a use: do not move its definition
// past the copy when the forwarding scope precedes that copy.
// CHECK-LABEL: func.func @preserve_intermediate_forwarding
// CHECK: %[[R:.*]] = scope.scope
// CHECK: hivm.hir.vmul
// CHECK: scope.return
// CHECK: %[[FORWARD:.*]] = scope.scope
// SEPARATE-NEXT: scope.return %[[R]]
// SEPARATE: hivm.hir.copy
// MERGED-NEXT: hivm.hir.copy
// MERGED-NEXT: scope.return %[[R]]
func.func @preserve_intermediate_forwarding(%p: tensor<16xf16>, %q: tensor<16xf16>, %dst: memref<16xf16, #hivm.address_space<cbuf>>, %n: index) -> tensor<16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %q) -> tensor<16xf16> {
    %r = scope.scope : () -> tensor<16xf16> {
      %e = tensor.empty() : tensor<16xf16>
      %extra = hivm.hir.vmul ins(%state, %state : tensor<16xf16>, tensor<16xf16>) outs(%e : tensor<16xf16>) -> tensor<16xf16>
      scope.return %extra : tensor<16xf16>
    } {hivm.independent_preload_prefix, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    %forward = scope.scope : () -> tensor<16xf16> {
      scope.return %r : tensor<16xf16>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.copy ins(%p : tensor<16xf16>) outs(%dst : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %forward : tensor<16xf16>
  }
  return %out : tensor<16xf16>
}
