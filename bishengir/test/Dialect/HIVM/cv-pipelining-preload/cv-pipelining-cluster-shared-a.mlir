// Two matmuls that read the same left matrix should issue back to back,
// ahead of an intervening matmul that reads a different left matrix.
// hivm-reposition-preload-work does this, and only for a registered kernel.
// This LIT bypasses the registry on the first run.
// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" -allow-unregistered-dialect | FileCheck %s
// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=2" -allow-unregistered-dialect | FileCheck %s --check-prefix=OFF

// CHECK-LABEL: func.func @cluster_shared_left_matrix
// CHECK: hivm.hir.mmadL1{{.*}}ins(%[[A:[^,]+]],
// CHECK: hivm.hir.mmadL1{{.*}}ins(%[[A]],
// CHECK: hivm.hir.mmadL1{{.*}}ins(%[[OTHER:[^,]+]],
// CHECK-NOT: hivm.hir.mmadL1{{.*}}ins(%[[A]],
//
// OFF-LABEL: func.func @cluster_shared_left_matrix
// OFF: hivm.hir.mmadL1{{.*}}ins(%[[A:[^,]+]],
// OFF: hivm.hir.mmadL1{{.*}}ins(%[[OTHER:[^,]+]],
// OFF: hivm.hir.mmadL1{{.*}}ins(%[[A]],
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cluster_shared_left_matrix(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<true> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a_mem = "some_op"() : () -> memref<16x16xf16>
    %a = bufferization.to_tensor %a_mem : memref<16x16xf16>
    %b0_mem = "some_op"() : () -> memref<16x16xf16>
    %b0 = bufferization.to_tensor %b0_mem : memref<16x16xf16>
    %b1_mem = "some_op"() : () -> memref<16x16xf16>
    %b1 = bufferization.to_tensor %b1_mem : memref<16x16xf16>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %trip = "some_op"() : () -> i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c_init = tensor.empty() : tensor<16x16xf32>
    %other_mem = "some_op"() : () -> memref<16x16xf16>
    %other = bufferization.to_tensor %other_mem : memref<16x16xf16>
    scf.for %iv = %c0_i32 to %trip step %c1_i32 : i32 {
      %qk_init = tensor.empty() : tensor<16x16xf32>
      %qk = hivm.hir.mmadL1 ins(%a, %b0, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%qk_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %qk_ws = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      %qk_dst = bufferization.to_tensor %qk_ws restrict writable : memref<16x16xf32>
      %qk_fp = hivm.hir.fixpipe ins(%qk : tensor<16x16xf32>) outs(%qk_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
      "some_user"(%qk_fp) : (tensor<16x16xf32>) -> ()

      %sv_init = tensor.empty() : tensor<16x16xf32>
      %sv = hivm.hir.mmadL1 ins(%other, %b1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%sv_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %sv_ws = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      %sv_dst = bufferization.to_tensor %sv_ws restrict writable : memref<16x16xf32>
      %sv_fp = hivm.hir.fixpipe ins(%sv : tensor<16x16xf32>) outs(%sv_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
      "some_user"(%sv_fp) : (tensor<16x16xf32>) -> ()

      %qw_init = tensor.empty() : tensor<16x16xf32>
      %qw = hivm.hir.mmadL1 ins(%a, %b1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%qw_init : tensor<16x16xf32>) -> tensor<16x16xf32>
      %qw_ws = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf32>
      %qw_dst = bufferization.to_tensor %qw_ws restrict writable : memref<16x16xf32>
      %qw_fp = hivm.hir.fixpipe ins(%qw : tensor<16x16xf32>) outs(%qw_dst : tensor<16x16xf32>) -> tensor<16x16xf32>
      "some_user"(%qw_fp) : (tensor<16x16xf32>) -> ()
    }
    "some_consume"(%c_init) : (tensor<16x16xf32>) -> ()
    return
  }
}
