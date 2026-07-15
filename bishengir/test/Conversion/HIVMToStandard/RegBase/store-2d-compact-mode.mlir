// RUN: bishengir-opt %s \
// RUN:   -hacc-append-device-spec=target=Ascend950PR_9589 \
// RUN:   -convert-hivm-to-std | FileCheck %s
// RUN: %if hivmc-a5 %{ bishengir-compile %s \
// RUN:   --target=Ascend950PR_9589 \
// RUN:   --enable-hfusion-compile=false \
// RUN:   --enable-hivm-compile=true \
// RUN:   -o %t.o %}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @store_2d_compact_mode(
      %sync: memref<?xi8>,
      %workspace: memref<?xi8>,
      %dst_base: memref<?xf32, #hivm.address_space<gm>>) attributes {
        SyncBlockLockArgIdx = 0 : i64,
        WorkspaceArgIdx = 1 : i64,
        global_kernel = "local",
        mix_mode = "aiv",
        parallel_mode = "simd"
      } {
    %dst = memref.reinterpret_cast %dst_base to
        offset: [0], sizes: [16, 4], strides: [156, 1]
        : memref<?xf32, #hivm.address_space<gm>> to
          memref<16x4xf32, strided<[156, 1]>, #hivm.address_space<gm>>
    %src = memref.alloc()
        : memref<16x4xf32, #hivm.address_space<ub>>

    // UB rows are packed (stride0 == size1), while GM rows are strided.
    hivm.hir.store
        ins(%src : memref<16x4xf32, #hivm.address_space<ub>>)
        outs(%dst
            : memref<16x4xf32, strided<[156, 1]>, #hivm.address_space<gm>>)
    return
  }
}

// CHECK-LABEL: func.func @store_2d_compact_mode
// CHECK: %[[DST:.*]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 4], strides: [156, 1]
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16x4xf32, #hivm.address_space<ub>>
// CHECK: %[[SRC_CAST:.*]] = memref.cast %[[SRC]] : memref<16x4xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
// CHECK: %[[DST_CAST:.*]] = memref.cast %[[DST]] : memref<16x4xf32, strided<[156, 1]>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
// CHECK: call @store_ubuf_to_gm_2d_float(%[[SRC_CAST]], %[[DST_CAST]], %{{.*}})
