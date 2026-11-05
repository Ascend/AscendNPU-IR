// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --enable-tla-mode --disable-ffts --mlir-print-ir-after=one-shot-bufferize,convert-tensor-view-to-hivm -o %t.o 2>&1 | FileCheck %s
// RUN: test -s %t.o

// CHECK: IR Dump After OneShotBufferize
// CHECK: tv.view_load {{.*}} outs({{.*}} : memref<8xf32>)
// CHECK: tv.view_store {{.*}} : {{.*}}, memref<8xf32>, index
// CHECK: IR Dump After ConvertTensorViewToHIVM
// CHECK-NOT: tv.
// CHECK: hivm.hir.load
// CHECK: hivm.hir.store
// CHECK-NOT: tv.

!base = !tv.tensor_view<16xf32, strides=[1]>
!gather = !tv.tensor_view<16xf32, strides=[1], #tv.gather_scatter_view<tile=[8], sparse_dim=[0], padding_value=zero>>
!partition = !tv.tensor_view<16xf32, strides=[1], #tv.partition_view<tile=[8], dim_map=[0], padding_value=zero>>

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @tensorview_copy(
      %sync: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
      %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %input: memref<?xf32, #hivm.address_space<gm>>,
      %indicesBase: memref<?xi32, #hivm.address_space<gm>>,
      %output: memref<?xf32, #hivm.address_space<gm>>,
      %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32)
      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                  hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIV>,
                  mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %indexGM = memref.reinterpret_cast %indicesBase to offset: [0], sizes: [8], strides: [1]
        : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, #hivm.address_space<gm>>
    %indexUB = memref.alloc() : memref<8xi32, #hivm.address_space<ub>>
    memref.copy %indexGM, %indexUB
        : memref<8xi32, #hivm.address_space<gm>> to memref<8xi32, #hivm.address_space<ub>>
    %indices = bufferization.to_tensor %indexUB restrict writable : memref<8xi32, #hivm.address_space<ub>>
    %source = tv.make_tensor_view %input, sizes=[%c16], strides=[%c1]
        : memref<?xf32, #hivm.address_space<gm>> -> !base
    %target = tv.make_tensor_view %output, sizes=[%c16], strides=[%c1]
        : memref<?xf32, #hivm.address_space<gm>> -> !base
    %g = tv.make_gather_scatter_view %source : !base -> !gather
    %p = tv.make_partition_view %target : !base -> !partition
    %init = tensor.empty() : tensor<8xf32>
    %value = tv.view_load %g[%indices] outs(%init : tensor<8xf32>)
        : !gather, tensor<8xi32> -> tensor<8xf32>
    tv.view_store %p[%c0], %value : !partition, tensor<8xf32>, index
    return
  }
}
