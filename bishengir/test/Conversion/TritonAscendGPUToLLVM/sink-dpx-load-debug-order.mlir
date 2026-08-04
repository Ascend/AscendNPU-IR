// RUN: bishengir-opt --convert-debug-to-ascenddpx \
// RUN:   --allocate-ascend-shared-memory \
// RUN:   --convert-triton-ascend-gpu-to-llvm \
// RUN:   --sink-dpx-load %s | FileCheck %s
//
// A failing device_assert terminates the kernel. SinkDPXLoad must therefore
// preserve the assert runtime call before stores that follow the source assert.

module attributes {
    "ttg.simt-optimization-mode" = 1111 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared = 0 : i32,
    ttg.target = "cuda:80"
} {
  tt.func public @assert_then_store(
      %condition: i1, %output: !tt.ptr<i32>, %value: i32)
      attributes {noinline = false} {
    tt.assert %condition, "stop before store" : i1
    tt.store %output, %value : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL: llvm.func @assert_then_store
// CHECK-NOT: ascend_dpx.store
// CHECK: scf.if
// CHECK: llvm.call @_mlir_ciface_assert_scalar_bool_gm_simt
// CHECK: ascend_dpx.store
