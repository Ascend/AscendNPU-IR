// RUN: bishengir-opt %s --hivm-v2-split-kernel --split-input-file --verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s --hivm-split-mix-kernel --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @vector_only(
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>}
// CHECK: hivm.hir.vexp
// CHECK-NOT: @vector_only_mix_
// CHECK-LABEL: func.func @host(
// CHECK: call @vector_only(
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vector_only(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.core_ratio = #hivm.core_ratio<0, 1>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.vexp ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>)
    return
  }
  func.func @host(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    func.call @vector_only(%src, %dst) : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: func.func @cube_only(
// CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>}
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
// CHECK-NOT: @cube_only_mix_
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @cube_only()
      attributes {hivm.core_ratio = #hivm.core_ratio<1, 0>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // expected-error@+1 {{hivm.core_ratio<1, 0> reserves no vector core, but the kernel has vector work}}
  func.func @missing_vector(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {hivm.core_ratio = #hivm.core_ratio<1, 0>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.vexp ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>)
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // expected-error@+1 {{hivm.core_ratio<0, 1> reserves no cube core, but the kernel has cube work}}
  func.func @missing_cube()
      attributes {hivm.core_ratio = #hivm.core_ratio<0, 1>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    return
  }
}
