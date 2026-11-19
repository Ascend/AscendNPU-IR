// The late device-backend stage may be unavailable in published builds; this
// test checks the marked MIX IR emitted before sub-block partitioning.
// RUN: (bishengir-compile %s --target=Ascend950 \
// RUN:   --enable-hfusion-compile=true --enable-triton-kernel-compile=true \
// RUN:   --enable-auto-multi-buffer=true --enable-auto-bind-sub-block=true \
// RUN:   --custom-aic-number=28 --custom-aiv-number=56 --disable-ffts \
// RUN:   --enable-auto-blockify-loop --use-pipelinev2=true \
// RUN:   --enable-lir-compile=false --mlir-disable-threading \
// RUN:   --mlir-print-ir-before=hivm-partition-and-bind-sub-block \
// RUN:   -o %t.ll 2>&1 || true) | FileCheck %s

// This registered symbol skips the early coupled-buffer marking. CV v2 must
// identify coupled buffers before marking their stage-dependent slot counts.
// The CUBE stage computes dot(i) ahead of the VECTOR recurrence that consumes
// dot(i - 1). Its cross-core buffer needs two slots before the MIX function is
// split, when both producer and consumer stages remain visible.
// CHECK-LABEL: func.func @chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %[[BUFFER]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
// CHECK: hivm.hir.fixpipe {{.*}} outs(%[[BUFFER]] : memref<32x32xf32, #hivm.address_space<ub>>)
// CHECK: hivm.preload_num = 1 : i32
// CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[BUFFER]]
// CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
// CHECK: hivm.hir.vadd ins(%{{.*}}, %[[TENSOR]]
// CHECK: hivm.preload_num = 0 : i32

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  func.func @chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(%lock: memref<?xi8>, %workspace: memref<?xi8>, %lhs: memref<?xf16>, %rhs: memref<?xf16>, %output: memref<?xf32>, %n: i32, %pid0: i32, %pid1: i32, %pid2: i32, %num0: i32, %num1: i32, %num2: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%state = %init) -> tensor<32x32xf32> : i32 {
      %idx = arith.index_cast %i : i32 to index
      %offset = arith.muli %idx, %c1024 : index
      %lhs_view = memref.reinterpret_cast %lhs to offset: [%offset], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %rhs_view = memref.reinterpret_cast %rhs to offset: [%offset], sizes: [32, 32], strides: [32, 1] : memref<?xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %a = memref.alloc() : memref<32x32xf16>
      %b = memref.alloc() : memref<32x32xf16>
      memref.copy %lhs_view, %a : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
      memref.copy %rhs_view, %b : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
      %a_tensor = bufferization.to_tensor %a restrict writable : memref<32x32xf16>
      %b_tensor = bufferization.to_tensor %b restrict writable : memref<32x32xf16>
      %dot = linalg.matmul {indexing_maps = [#map, #map1, #map2], input_precision = "ieee"} ins(%a_tensor, %b_tensor : tensor<32x32xf16>, tensor<32x32xf16>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
      %next = arith.addf %state, %dot : tensor<32x32xf32>
      %output_view = memref.reinterpret_cast %output to offset: [%offset], sizes: [32, 32], strides: [32, 1] : memref<?xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      bufferization.materialize_in_destination %next in writable %output_view : (tensor<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
      scf.yield %next : tensor<32x32xf32>
    }
    return
  }
}
