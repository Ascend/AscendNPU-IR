// The CPU-runner hook stops the RegBase flow after CV pipelining, before
// hivmc-a5. This device-only test has no host entry, so the hook's crash is
// expected; FileCheck validates the IR emitted before that point.
// RUN: not --crash bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --enable-auto-multi-buffer=True \
// RUN:   --enable-preload=False \
// RUN:   --enable-lazy-loading=true \
// RUN:   --enable-auto-bind-sub-block=True \
// RUN:   --disable-ffts \
// RUN:   --enable-hivm-graph-sync-solver=True \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit \
// RUN:   --enable-mixed-cv=True \
// RUN:   --enable-flatten=False \
// RUN:   --enable-hfusion-compile=true \
// RUN:   --enable-triton-kernel-compile=true \
// RUN:   --enable-lir-compile=false \
// RUN:   --set-cv-pipeline-mode=unroll \
// RUN:   --enable-cpu-runner-before=hivm-partition-and-bind-sub-block \
// RUN:   --mlir-print-ir-after=cv-pipelining \
// RUN:   2>&1 | FileCheck %s --check-prefix=LAZY-ON
//
// RUN: not --crash bishengir-compile %s \
// RUN:   --target=Ascend950PR_9579 \
// RUN:   --enable-auto-multi-buffer=True \
// RUN:   --enable-preload=False \
// RUN:   --enable-lazy-loading=false \
// RUN:   --enable-auto-bind-sub-block=True \
// RUN:   --disable-ffts \
// RUN:   --enable-hivm-graph-sync-solver=True \
// RUN:   --limit-auto-multi-buffer-of-local-buffer=no-limit \
// RUN:   --enable-mixed-cv=True \
// RUN:   --enable-flatten=False \
// RUN:   --enable-hfusion-compile=true \
// RUN:   --enable-triton-kernel-compile=true \
// RUN:   --enable-lir-compile=false \
// RUN:   --set-cv-pipeline-mode=unroll \
// RUN:   --enable-cpu-runner-before=hivm-partition-and-bind-sub-block \
// RUN:   --mlir-print-ir-after=cv-pipelining \
// RUN:   2>&1 | FileCheck %s --check-prefix=LAZY-OFF

// This case is reduced from bwd_qkv_kernel.
//
// A Q tile loaded from GM is consumed by two CUBE stages:
//   CUBE0: Q * K^T
//   VECTOR: exp2 + truncate + transpose
//   CUBE2: vector_result^T * Q
//
// With lazy loading enabled, the Q GM->L1 load is rematerialized in CUBE2.
// Therefore there are three nd2nz ops in total: Q + K in CUBE0, Q in CUBE2.
// With lazy loading disabled, Q is loaded only once and forwarded to CUBE2,
// so only two nd2nz ops remain.

// LAZY-ON-LABEL: func.func @test_lazy_q_tile
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: hivm.hir.nd2nz
// LAZY-ON: hivm.hir.nd2nz
// LAZY-ON-NOT: hivm.hir.nd2nz
// LAZY-ON: hivm.hir.mmadL1
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// LAZY-ON-NOT: hivm.hir.nd2nz
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: hivm.hir.nd2nz
// LAZY-ON-NOT: hivm.hir.nd2nz
// LAZY-ON: hivm.hir.mmadL1
// LAZY-ON-NOT: hivm.hir.nd2nz
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// LAZY-ON-NOT: hivm.hir.nd2nz
// LAZY-ON-NOT: hivm.hir.mmadL1
// LAZY-ON: return

// LAZY-OFF-LABEL: func.func @test_lazy_q_tile
// LAZY-OFF-NOT: hivm.hir.mmadL1
// LAZY-OFF: hivm.hir.nd2nz
// LAZY-OFF: hivm.hir.nd2nz
// LAZY-OFF-NOT: hivm.hir.nd2nz
// LAZY-OFF: hivm.hir.mmadL1
// LAZY-OFF-NOT: hivm.hir.mmadL1
// LAZY-OFF: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// LAZY-OFF-NOT: hivm.hir.nd2nz
// LAZY-OFF-NOT: hivm.hir.mmadL1
// LAZY-OFF: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// LAZY-OFF-NOT: hivm.hir.nd2nz
// LAZY-OFF: hivm.hir.mmadL1
// LAZY-OFF-NOT: hivm.hir.nd2nz
// LAZY-OFF-NOT: hivm.hir.mmadL1
// LAZY-OFF: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// LAZY-OFF-NOT: hivm.hir.nd2nz
// LAZY-OFF-NOT: hivm.hir.mmadL1
// LAZY-OFF: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_q_tile(
      %arg0: memref<?xi8>,
      %arg1: memref<?xi8>,
      %q: memref<?xbf16> {tt.tensor_kind = 0 : i32},
      %k: memref<?xbf16> {tt.tensor_kind = 0 : i32},
      %out: memref<?xbf16> {tt.tensor_kind = 1 : i32},
      %program_num_x: i32,
      %program_num_y: i32,
      %program_num_z: i32,
      %program_id_x: i32,
      %program_id_y: i32,
      %program_id_z: i32)
      attributes {
        SyncBlockLockArgIdx = 0 : i64,
        WorkspaceArgIdx = 1 : i64,
        global_kernel = "local",
        mix_mode = "mix",
        parallel_mode = "simd"
      } {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64 = arith.constant 64 : index
    %cst0 = arith.constant 0.000000e+00 : f32

    %acc0 = tensor.empty() : tensor<128x128xf32>
    %zero0 = linalg.fill
        ins(%cst0 : f32)
        outs(%acc0 : tensor<128x128xf32>)
        -> tensor<128x128xf32>

    %acc1 = tensor.empty() : tensor<128x64xf32>
    %zero1 = linalg.fill
        ins(%cst0 : f32)
        outs(%acc1 : tensor<128x64xf32>)
        -> tensor<128x64xf32>

    scf.for %iv = %c0_i32 to %c256_i32 step %c128_i32 : i32 {
      %iv_idx = arith.index_cast %iv : i32 to index
      %offset = arith.muli %iv_idx, %c64 : index

      // Q tile. This is the lazy-loading candidate shared by two matmuls.
      %q_src = memref.reinterpret_cast %q
          to offset: [%offset], sizes: [128, 64], strides: [64, 1]
          : memref<?xbf16>
            to memref<128x64xbf16, strided<[64, 1], offset: ?>>
      %q_buf = memref.alloc() : memref<128x64xbf16>
      memref.copy %q_src, %q_buf
          : memref<128x64xbf16, strided<[64, 1], offset: ?>>
            to memref<128x64xbf16>
      %q_tensor = bufferization.to_tensor %q_buf restrict writable
          : memref<128x64xbf16>

      // K tile is used only by the first CUBE stage.
      %k_src = memref.reinterpret_cast %k
          to offset: [%offset], sizes: [128, 64], strides: [64, 1]
          : memref<?xbf16>
            to memref<128x64xbf16, strided<[64, 1], offset: ?>>
      %k_buf = memref.alloc() : memref<128x64xbf16>
      memref.copy %k_src, %k_buf
          : memref<128x64xbf16, strided<[64, 1], offset: ?>>
            to memref<128x64xbf16>
      %k_tensor = bufferization.to_tensor %k_buf restrict writable
          : memref<128x64xbf16>

      %kt_init = tensor.empty() : tensor<64x128xbf16>
      %k_transposed = linalg.transpose
          ins(%k_tensor : tensor<128x64xbf16>)
          outs(%kt_init : tensor<64x128xbf16>)
          permutation = [1, 0]

      // CUBE0.
      %score = linalg.matmul {input_precision = "ieee"}
          ins(%q_tensor, %k_transposed
              : tensor<128x64xbf16>, tensor<64x128xbf16>)
          outs(%zero0 : tensor<128x128xf32>)
          -> tensor<128x128xf32>

      // VECTOR stage between the two CUBE stages.
      %prob = math.exp2 %score : tensor<128x128xf32>
      %prob_bf16 = arith.truncf %prob
          : tensor<128x128xf32> to tensor<128x128xbf16>
      %prob_t_init = tensor.empty() : tensor<128x128xbf16>
      %prob_transposed = linalg.transpose
          ins(%prob_bf16 : tensor<128x128xbf16>)
          outs(%prob_t_init : tensor<128x128xbf16>)
          permutation = [1, 0]

      // CUBE2. Reuse the same Q tile, which makes Q eligible for lazy loading.
      %result = linalg.matmul {input_precision = "ieee"}
          ins(%prob_transposed, %q_tensor
              : tensor<128x128xbf16>, tensor<128x64xbf16>)
          outs(%zero1 : tensor<128x64xf32>)
          -> tensor<128x64xf32>

      %result_bf16 = arith.truncf %result
          : tensor<128x64xf32> to tensor<128x64xbf16>

      // Keep the full CUBE->VECTOR->CUBE chain observable.
      %out_dst = memref.reinterpret_cast %out
          to offset: [%offset], sizes: [128, 64], strides: [64, 1]
          : memref<?xbf16>
            to memref<128x64xbf16, strided<[64, 1], offset: ?>>
      bufferization.materialize_in_destination
          %result_bf16 in writable %out_dst
          : (tensor<128x64xbf16>,
             memref<128x64xbf16, strided<[64, 1], offset: ?>>) -> ()
    }

    return
  }
}
