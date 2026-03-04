// RUN: bishengir-compile %s --target=Ascend950PR_9589 --enable-auto-multi-buffer=true --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-lir-compile=false

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @triton_atomic_or_3D(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c51 = arith.constant 51 : index
    %c51_i32 = arith.constant 51 : i32
    %c24_i32 = arith.constant 24 : i32
    %0 = arith.muli %arg7, %c24_i32 : i32
    %1 = arith.muli %arg9, %c51_i32 : i32
    %2 = arith.muli %arg8, %c51_i32 : i32
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %2 : i32 to index
    %5 = arith.index_cast %0 : i32 to index
    %6 = arith.muli %5, %c51 : index
    %7 = arith.addi %4, %6 : index
    %8 = arith.addi %7, %3 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [24, 1, 51], strides: [51, 51, 1] : memref<?xi32> to memref<24x1x51xi32, strided<[51, 51, 1], offset: ?>>
    %alloc = memref.alloc() : memref<24x1x51xi32>
    memref.copy %reinterpret_cast, %alloc : memref<24x1x51xi32, strided<[51, 51, 1], offset: ?>> to memref<24x1x51xi32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [24, 1, 51], strides: [51, 51, 1] : memref<?xi32> to memref<24x1x51xi32, strided<[51, 51, 1]>>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reinterpret_cast_0, %alloc : memref<24x1x51xi32, strided<[51, 51, 1]>>, memref<24x1x51xi32>) outs(%reinterpret_cast_0 : memref<24x1x51xi32, strided<[51, 51, 1]>>) attrs =  {GenericAtomicRMW = "or", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %9 = arith.ori %in, %in_1 : i32
      linalg.yield %9 : i32
    }
    return
  }
}