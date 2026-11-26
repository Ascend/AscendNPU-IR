// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory=mem-plan-mode=global-work-space-plan | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory | FileCheck %s --check-prefix=LOCAL
// Global workspace planning must not index local scratch buffers in its
// bufferInfos map. They are allocated by the later local-memory pass.
module {
  // CHECK-LABEL: func.func @macro_with_local_scratch
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<2x64xf32, #hivm.address_space<ub>>
  // CHECK: %[[A:.*]] = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  // CHECK: %[[B:.*]] = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.custom_macro
  // CHECK-SAME: outs(%[[OUT]]
  // CHECK-SAME: tmps(%[[A]], %[[B]]
  // LOCAL-LABEL: func.func @macro_with_local_scratch
  // LOCAL-NOT: memref.alloc
  // LOCAL: hivm.hir.pointer_cast
  // LOCAL: hivm.hir.pointer_cast
  // LOCAL: hivm.hir.pointer_cast
  // LOCAL: hivm.hir.custom_macro
  // LOCAL-SAME: tmps(
  func.func @macro_with_local_scratch(%a: memref<2x64xf32, #hivm.address_space<gm>>,
                                     %b: memref<2x64xf32, #hivm.address_space<gm>>,
                                     %out: memref<2x64xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %dst = memref.alloc() : memref<2x64xf32, #hivm.address_space<ub>>
    %tmp_a = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    %tmp_b = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "macro_add"}
        "user.macro" ins(%a, %b : memref<2x64xf32, #hivm.address_space<gm>>, memref<2x64xf32, #hivm.address_space<gm>>)
        outs(%dst : memref<2x64xf32, #hivm.address_space<ub>>)
        tmps(%tmp_a, %tmp_b : memref<4x32xf32, #hivm.address_space<ub>>, memref<4x32xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%dst : memref<2x64xf32, #hivm.address_space<ub>>)
                   outs(%out : memref<2x64xf32, #hivm.address_space<gm>>)
    return
  }
}
