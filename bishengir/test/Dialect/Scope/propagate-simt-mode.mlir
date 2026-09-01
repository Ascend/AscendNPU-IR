// RUN: bishengir-opt -outline-scope -propagate-simt-mode -split-input-file %s | FileCheck %s

// Test: SIMT scope with sub-block tiling reverted attribute is wrapped in an
// scf.if guard. The caller directly invokes the SIMT VF, so it is marked MIX.
// CHECK-LABEL: func.func @test_simt_scope_with_reverted_attr_scope_0(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>} {
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simt_scope_with_reverted_attr(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>}
// CHECK: hivm.hir.get_sub_block_idx
// CHECK: arith.index_cast
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: call @test_simt_scope_with_reverted_attr_scope_0
// CHECK: limit_sub_block_id0
module attributes {"hivm.tile_and_bind_subblock_reverted"} {
  func.func @test_simt_scope_with_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// -----

// Test: SIMT scope without the reverted attribute produces a plain call. The
// caller directly invokes the SIMT VF, so it is marked MIX.
// CHECK-LABEL: func.func @test_simt_scope_without_reverted_attr_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simt_scope_without_reverted_attr(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>}
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK: call @test_simt_scope_without_reverted_attr_scope_0
module {
  func.func @test_simt_scope_without_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// -----

// Test: A kernel marked SIMD that directly invokes the SIMT VF is re-marked
// MIX (a stale SIMD mark is simply overwritten).
// CHECK-LABEL: func.func @test_simd_kernel_with_simt_scope_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simd_kernel_with_simt_scope(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>}
// CHECK: call @test_simd_kernel_with_simt_scope_scope_0
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_simd_kernel_with_simt_scope(%arg0: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMD>} {
    scope.scope : () -> () {
      %cst = arith.constant 1.0 : f32
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>, outline = true}
    return
  }
}

// -----

// Test: A caller feeding a constant into the SIMT scope directly invokes the
// SIMT VF after outlining, so it is marked MIX.
// CHECK-LABEL: func.func @test_simt_only_kernel_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_simt_only_kernel(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>}
// CHECK: call @test_simt_only_kernel_scope_0
// CHECK: return
module {
  func.func @test_simt_only_kernel(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    } {hivm.vf_mode = #hivm.vf_mode<SIMT>}
    return
  }
}

// -----

// Test: Non-SIMT scope: no SIMT callee anywhere, so no vf_mode is added to
// any function.
// CHECK-LABEL: func.func @test_non_simt_scope_with_reverted_attr_scope_0(
// CHECK: memref.store
// CHECK: return

// CHECK-LABEL: func.func @test_non_simt_scope_with_reverted_attr(
// CHECK-SAME: %{{.*}}: memref<f32>) {
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK: call @test_non_simt_scope_with_reverted_attr_scope_0
module attributes {"hivm.tile_and_bind_subblock_reverted"} {
  func.func @test_non_simt_scope_with_reverted_attr(%arg0: memref<f32>) {
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      memref.store %cst, %arg0[] : memref<f32>
      scope.return
    }
    return
  }
}

// -----

// Test: propagation checks direct callees only: the helper directly invokes
// the SIMT VF and is marked MIX, while the kernel merely calls the (now MIX)
// helper and stays unmarked.
// CHECK-LABEL: func.func @test_transitive_simt_kernel(
// CHECK-SAME: %{{.*}}: memref<f32>) {
// CHECK: call @test_transitive_simt_helper
// CHECK: return

// CHECK-LABEL: func.func @test_transitive_simt_helper(
// CHECK-SAME: %{{.*}}: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<MIX>}
// CHECK: call @test_transitive_simt_vf
// CHECK: return
module {
  func.func private @test_transitive_simt_vf(%arg0: memref<f32>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>}
  func.func @test_transitive_simt_kernel(%arg0: memref<f32>) {
    call @test_transitive_simt_helper(%arg0) : (memref<f32>) -> ()
    return
  }
  func.func @test_transitive_simt_helper(%arg0: memref<f32>) {
    call @test_transitive_simt_vf(%arg0) : (memref<f32>) -> ()
    return
  }
}
