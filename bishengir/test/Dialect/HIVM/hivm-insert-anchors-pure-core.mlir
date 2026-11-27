// RUN: bishengir-opt --allow-unregistered-dialect --hivm-insert-anchors-and-backup %s | FileCheck %s
// RUN: bishengir-opt --allow-unregistered-dialect --hivm-insert-anchors-and-backup="insert-anchor-ops-before-mem-effect-ops" %s | FileCheck %s --check-prefix=EFFECT

// Neutral ops leave the loop cube-only, with anchors only at its boundaries.
// CHECK-LABEL: func.func @pure_loop(
func.func @pure_loop(%lb: index, %ub: index, %step: index, %input: tensor<16xf32>, %out: memref<16xi32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK-NEXT: hivm.hir.anchor
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    // CHECK-NOT: hivm.hir.anchor
    %tmp = memref.alloc() : memref<i32>
    %bits = hivm.hir.bitcast %input : tensor<16xf32> -> tensor<16xi32>
    annotation.mark %bits {test_marker} : tensor<16xi32>
    hivm.hir.store ins(%bits : tensor<16xi32>) outs(%out : memref<16xi32>)
      {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    "test.unknown"() : () -> ()
    memref.dealloc %tmp : memref<i32>
  // CHECK: } {hivm.keep_control_flow}
  }
  // CHECK-NEXT: hivm.hir.anchor
  // CHECK-NEXT: return
  return
}

// Different core types in separate blocks keep anchors in both blocks.
// CHECK-LABEL: func.func @mixed_blocks(
func.func @mixed_blocks(%cond: i1, %buf: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.execute_region
  scf.execute_region {
    cf.cond_br %cond, ^left, ^right
  // CHECK: ^bb1:
  ^left:
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.load
    %cube = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<i32>
    scf.yield
  // CHECK: ^bb2:
  ^right:
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.load
    %vector = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<i32>
    scf.yield
  }
  // CHECK: return
  return
}

// Unresolved placement on a memory copy prevents the region from being cube-only.
// CHECK-LABEL: func.func @unresolved_copy(
func.func @unresolved_copy(%cond: i1, %src: memref<i32>, %dst: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.load
    %cube = memref.load %src[] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<i32>
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.copy
    memref.copy %src, %dst : memref<i32> to memref<i32>
  }
  return
}

// Memory-effect mode distinguishes reported reads and writes from allocation,
// deallocation, and unregistered ops without an effects interface.
// EFFECT-LABEL: func.func @memory_effects(
func.func @memory_effects(%src: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // EFFECT: arith.constant
  %i = arith.constant 0 : index
  // EFFECT-NEXT: memref.alloc
  %tmp = memref.alloc() : memref<i32>
  // EFFECT-NEXT: "test.unknown"
  "test.unknown"(%i) : (index) -> ()
  // EFFECT-NEXT: hivm.hir.anchor
  // EFFECT-NEXT: memref.copy
  memref.copy %src, %tmp : memref<i32> to memref<i32>
  // EFFECT-NEXT: memref.dealloc
  memref.dealloc %tmp : memref<i32>
  // EFFECT-NEXT: hivm.hir.anchor
  // EFFECT-NEXT: return
  return
}

// Preload scopes keep their anchors. Only the mixed if needs an else branch.
// CHECK-LABEL: func.func @preload(
func.func @preload(%cond: i1, %buf: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scope.scope
  scope.scope : () -> () {
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: scf.if
    scf.if %cond {
      // CHECK-NEXT: memref.load
      %pure = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<i32>
    // CHECK-NEXT: } {hivm.keep_control_flow}
    }
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: scf.if
    scf.if %cond {
      // CHECK-NEXT: hivm.hir.anchor
      // CHECK-NEXT: memref.load
      %vector = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<i32>
      %cube = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<i32>
    // CHECK: } else {
    // CHECK-NEXT: hivm.hir.anchor
    }
    // CHECK: scope.return
    scope.return
  } {hivm.preload_num = 1 : i32}
  return
}

// A single-core CV pipelining loop still needs interior anchors.
// CHECK-LABEL: func.func @pipeline(
func.func @pipeline(%lb: index, %ub: index, %step: index, %buf: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step {
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.load
    %v = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<i32>
  } {cv_unrolled_loop}
  return
}

// Unresolved op and marker attributes yield to the callee's concrete type.
func.func private @cube_helper() -> i32
    attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: func.func @typed_call(
func.func @typed_call(%cond: i1, %buf: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: memref.load
    %v = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<i32>
    // CHECK-NEXT: call @cube_helper
    %value = func.call @cube_helper() {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> i32
    // CHECK-NEXT: annotation.mark
    annotation.mark %value {hivm.tcore_type_marker = #hivm.tcore_type_marker<CUBE_OR_VECTOR>} : i32
  // CHECK-NEXT: } {hivm.keep_control_flow}
  }
  return
}

// An unresolved scope attribute and result marker yield to a concrete marker.
// CHECK-LABEL: func.func @concrete_marker(
func.func @concrete_marker(%cond: i1, %buf: memref<i32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NEXT: hivm.hir.anchor
    // CHECK-NEXT: memref.load
    %v = memref.load %buf[] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<i32>
    %value = scope.scope : () -> i32 {
      %zero = arith.constant 0 : i32
      scope.return %zero : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE_OR_VECTOR>}
    annotation.mark %value {hivm.tcore_type_marker = #hivm.tcore_type_marker<VECTOR>} : i32
    annotation.mark %value {hivm.tcore_type_marker = #hivm.tcore_type_marker<CUBE_OR_VECTOR>} : i32
  }
  return
}
