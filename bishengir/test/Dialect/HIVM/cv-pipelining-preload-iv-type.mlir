// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -allow-unregistered-dialect %s | FileCheck %s

// Verify that cv-pipelining in skew (preload) mode correctly handles a
// non-index (i32) induction variable in createNewLoopsForPreloadWithScopes().
//
// Previously the scopeMap cast the i32 IV to index via arith::IndexCastOp,
// causing type mismatches when cloned body ops used the IV alongside i32-typed
// constants. The arith.index_cast in the loop body below (i32 → index for
// memref.subview indexing) exercises the fix: before the fix, scopeMap's
// bogus IndexCastOp makes the cloned arith.index_cast receive an index-typed
// operand instead of i32, triggering a type-mismatch crash.
//
// After the fix the IV is mapped directly (origIV → origIV), preserving the
// original i32 type throughout all scope bodies.

// CHECK-LABEL: func.func @test_preload_iv_i32
// Verify scopes are created around CUBE and VECTOR work items.
// The arith.index_cast (i32 IV → index) must survive inside the CUBE scope
// with its i32 operand preserved — before the fix the scopeMap injected a
// bogus IndexCastOp that made the cloned arith.index_cast fail verification.
// CHECK: scope.scope
// CHECK: arith.index_cast %{{.*}} : i32 to index
// CHECK-DAG: memref.subview
// CHECK-DAG: hivm.hir.mmadL1
// CHECK-DAG: hivm.hir.fixpipe
// CHECK: {{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>
// CHECK: scope.scope
// CHECK: hivm.hir.vexp
// CHECK: {{.*}}hivm.loop_core_type = #hivm.tcore_type<VECTOR>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_preload_iv_i32(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %k_mem = "some_op"() : () -> memref<256x16xf16>
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %bound = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %init = tensor.empty() : tensor<16x16xf32>
    %result = scf.for %i = %c0 to %bound step %step iter_args(%acc = %init) -> tensor<16x16xf32> : i32 {
      // Convert i32 IV to index to feed memref.subview.
      // This arith.index_cast carries the i32 IV into the dependency chain
      // of the CUBE work item (via memref.subview → to_tensor → mmadL1).
      // The scopeMap fix ensures the cloned arith.index_cast still receives
      // an i32 operand rather than a spurious index-typed IndexCastOp.
      %iv_idx = arith.index_cast %i : i32 to index
      %k_subview = memref.subview %k_mem[%iv_idx, %c0_idx] [16, 16] [%c1_idx, %c1_idx] : memref<256x16xf16> to memref<16x16xf16, strided<[?, ?], offset: ?>>
      %k_tensor = bufferization.to_tensor %k_subview : memref<16x16xf16, strided<[?, ?], offset: ?>>

      %dot0_init = tensor.empty() : tensor<16x16xf32>
      %dot0 = hivm.hir.mmadL1 ins(%a, %k_tensor, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %ws0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      annotation.mark %ws0 {hivm.multi_buffer = 4 : i32} : memref<16x16xf32>
      %ws0_tensor = bufferization.to_tensor %ws0 restrict writable : memref<16x16xf32>
      %fix0 = hivm.hir.fixpipe ins(%dot0 : tensor<16x16xf32>) outs(%ws0_tensor : tensor<16x16xf32>) -> tensor<16x16xf32>

      %load0_init = tensor.empty() : tensor<16x16xf32>
      %load0 = hivm.hir.load ins(%fix0 : tensor<16x16xf32>) outs(%load0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %v0_init = tensor.empty() : tensor<16x16xf32>
      %v0 = hivm.hir.vexp ins(%load0 : tensor<16x16xf32>) outs(%v0_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.yield %v0 : tensor<16x16xf32>
    }
    "some_consume"(%result) : (tensor<16x16xf32>) -> ()
    return
  }
}
