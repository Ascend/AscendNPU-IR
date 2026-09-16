// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s --convert-hivm-to-tritongpu --rewrite-slice-op-to-triton --convert-triton-to-tritongpu="target=cuda:80 num-warps=1 threads-per-warp=32 shared-memory-size=221184" --split-input-file --verify-diagnostics -o /dev/null

// CHECK-LABEL: tt.func @simple_indirect_load_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: !tt.ptr<f32>, %arg17: !tt.ptr<f32>, %arg18: i64, %arg19: i64, %arg20: i64)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT: %c1_i64_1 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_2 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_3 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tensor.empty() : tensor<8xf32>
// CHECK-NEXT: %c0_i64_4 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %5 = tt.splat %arg11 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %6 = tt.addptr %5, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: %7 = tt.load %6 evictionPolicy = evict_last : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %8 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %9 = tt.splat %arg17 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %10 = tt.addptr %9, %8 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: tt.store %10, %7 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xf32>
    %2 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) outs(%1 : tensor<8xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %2 : tensor<8xf32>)
    return
  }
}

// -----
// CHECK-LABEL: tt.func @simple_indirect_store_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !tt.ptr<f32>, %arg16: !tt.ptr<f32>, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i32)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT: %c1_i64_1 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_2 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_3 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %c0_i64_4 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %5 = tt.splat %arg11 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: %7 = tt.load %6 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %8 = tt.splat %arg16 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %9 = tt.addptr %8, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: tt.store %9, %7 evictionPolicy = evict_last : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<8xf32>, %arg3: memref<?xf32>, %arg4: i32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.local_load ins(%arg2 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%0 : tensor<8xi64>, %1 : tensor<8xf32>, %arg4 : i32) outs(%arg3 : memref<?xf32>) {cache = #hivm.cache_modifier<none>, evict = #hivm.eviction_policy<EvictLast>}
    return
  }
}


// -----
// CHECK-LABEL: tt.func @load_check_strided_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !tt.ptr<f32>, %arg8: !tt.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !tt.ptr<f32>, %arg13: !tt.ptr<f32>, %arg14: i64, %arg15: i64, %arg16: i64)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c16_i64_0 = arith.constant 16 : i64
// CHECK-NEXT: %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT: %c4_i64 = arith.constant 4 : i64
// CHECK-NEXT: %c32_i64_1 = arith.constant 32 : i64
// CHECK-NEXT: %c64_i64_2 = arith.constant 64 : i64
// CHECK-NEXT: %c4_i64_3 = arith.constant 4 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.reshape %0 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %cst = arith.constant dense<64> : tensor<16x1xi32>
// CHECK-NEXT: %2 = arith.muli %1, %cst : tensor<16x1xi32>
// CHECK-NEXT: %cst_4 = arith.constant dense<32> : tensor<16x1xi32>
// CHECK-NEXT: %3 = arith.addi %2, %cst_4 : tensor<16x1xi32>
// CHECK-NEXT: %4 = tt.splat %arg8 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NEXT: %5 = tt.addptr %4, %3 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.reshape %6 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %cst_5 = arith.constant dense<4> : tensor<1x16xi32>
// CHECK-NEXT: %8 = arith.muli %7, %cst_5 : tensor<1x16xi32>
// CHECK-NEXT: %9 = tt.broadcast %5 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %11 = tt.addptr %9, %10 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NEXT: %12 = tt.load %11 evictionPolicy = evict_first : tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %c32_i64_8 = arith.constant 32 : i64
// CHECK-NEXT: %c16_i64_9 = arith.constant 16 : i64
// CHECK-NEXT: %c16_i64_10 = arith.constant 16 : i64
// CHECK-NEXT: %c64_i64_11 = arith.constant 64 : i64
// CHECK-NEXT: %c4_i64_12 = arith.constant 4 : i64
// CHECK-NEXT: %13 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT: %c32_i64_13 = arith.constant 32 : i64
// CHECK-NEXT: %c64_i64_14 = arith.constant 64 : i64
// CHECK-NEXT: %c4_i64_15 = arith.constant 4 : i64
// CHECK-NEXT: %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %15 = tt.reshape %14 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %cst_16 = arith.constant dense<64> : tensor<16x1xi32>
// CHECK-NEXT: %16 = arith.muli %15, %cst_16 : tensor<16x1xi32>
// CHECK-NEXT: %cst_17 = arith.constant dense<32> : tensor<16x1xi32>
// CHECK-NEXT: %17 = arith.addi %16, %cst_17 : tensor<16x1xi32>
// CHECK-NEXT: %18 = tt.splat %arg13 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NEXT: %19 = tt.addptr %18, %17 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
// CHECK-NEXT: %20 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %21 = tt.reshape %20 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %cst_18 = arith.constant dense<4> : tensor<1x16xi32>
// CHECK-NEXT: %22 = arith.muli %21, %cst_18 : tensor<1x16xi32>
// CHECK-NEXT: %23 = tt.broadcast %19 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %24 = tt.broadcast %22 : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %25 = tt.addptr %23, %24 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NEXT: tt.store %25, %13 : tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @load_check_strided_2d(%alloc : memref<16x16xf32>, %arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [32], sizes: [16, 16], strides: [64, 4]
        : memref<?xf32> to memref<16x16xf32, strided<[64, 4], offset: 32>>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[64, 4], offset: 32>>)
                  outs(%alloc : memref<16x16xf32>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [32], sizes: [16, 16], strides: [64, 4]
        : memref<?xf32> to memref<16x16xf32, strided<[64, 4], offset: 32>>
    %1 = tensor.empty() : tensor<16x16xf32>
    hivm.hir.store ins(%1 : tensor<16x16xf32>)
                   outs(%reinterpret_cast_0 : memref<16x16xf32, strided<[64, 4], offset: 32>>)
    return
    }
}



// -----
// CHECK-LABEL: tt.func @check_store_atomicadd(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw add, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicadd(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <add>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicmax(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw max, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicmax(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <max>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicmin(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw min, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicmin(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <min>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicand(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw and, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicand(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <and>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw or, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicor(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <or>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicxor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>)
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_0 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_1 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_2 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_3 = arith.constant 0 : i64
// CHECK-NEXT: %c16_i64_4 = arith.constant 16 : i64
// CHECK-NEXT: %c1_i64_5 = arith.constant 1 : i64
// CHECK-NEXT: %c0_i64_6 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_7 = arith.constant 1 : i64
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %c0_i64_8 = arith.constant 0 : i64
// CHECK-NEXT: %c1_i64_9 = arith.constant 1 : i64
// CHECK-NEXT: %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %5 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %7 = tt.atomic_rmw xor, acq_rel, gpu, %6, %3 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicxor(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<16xi32>
  hivm.hir.store ins(%0 : tensor<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <xor>
  return
}

// -----
// CHECK-LABEL: tt.func @index_arg_kernel_scope_0(%arg0: i64)
// CHECK-NEXT: %0 = arith.trunci %arg0 : i64 to i32
// CHECK-NEXT: %1 = arith.index_castui %0 : i32 to index
// CHECK-NEXT: %2 = arith.index_cast %1 : index to i32
// CHECK-NEXT: %3 = arith.addi %2, %2 : i32
// CHECK-NEXT: tt.return

module {
  func.func @index_arg_kernel_scope_0(%arg0: index) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %0 = arith.index_cast %arg0 : index to i32
    %1 = arith.addi %0, %0 : i32
    return
  }
}

// -----

// The staging buffer is a function argument, not a memref.alloc: buffers are
// hoisted out of the function before this pass, and Triton has no MemRef
// semantics for one to lower to.
// CHECK-LABEL: tt.func @merge_16x16_to_64x64_inverse_kernel_mix_mix_aiv_scope_0
// CHECK-NOT: memref.
module {
  func.func @merge_16x16_to_64x64_inverse_kernel_mix_mix_aiv_scope_0(%arg0: i32, %arg12: index, %arg13: memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<gm>}, %arg14: memref<16xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<write>, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}, %arg15: memref<16xf32, #hivm.address_space<ub>> {hivm.memory_effect = #hivm.memory_effect<write>, hivm.simt_mem_scope_hint = #hivm.simt_mem_scope_hint<ub>}) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, noinline, outline, vector_mode = "simt"} {
    %cst_1 = arith.constant -1.000000e+00 : f32
    %218 = arith.index_cast %arg0 : i32 to index
    %219 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg12, %218]
    %reinterpret_cast = memref.reinterpret_cast %arg13 to offset: [%219], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%arg15 : memref<16xf32, #hivm.address_space<ub>>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %220 = bufferization.to_tensor %arg15 restrict writable : memref<16xf32, #hivm.address_space<ub>>
    %221 = tensor.empty() : tensor<16xf32>
    %222 = hivm.hir.vmul ins(%220, %cst_1 : tensor<16xf32>, f32) outs(%221 : tensor<16xf32>) -> tensor<16xf32>
    hivm.hir.local_store ins(%arg14 : memref<16xf32, #hivm.address_space<ub>>, %222 : tensor<16xf32>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @padded_load(
// CHECK: %[[SRC_DESC_OFFSET:.*]] = arith.addi {{.*}} : i64
// CHECK: %[[DST_DESC_OFFSET:.*]] = arith.addi {{.*}} : i64
// CHECK: %[[VALID_SIZE:.*]] = arith.index_cast {{.*}} : index to i64
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: %[[DST_OFFSET:.*]] = arith.trunci %[[DST_DESC_OFFSET]] : i64 to i32
// CHECK: %[[DST_OFFSET_TENSOR:.*]] = tt.splat %[[DST_OFFSET]] : i32 -> tensor<8xi32>
// CHECK: %[[SIZE:.*]] = arith.trunci %[[VALID_SIZE]] : i64 to i32
// CHECK: %[[SIZE_TENSOR:.*]] = tt.splat %[[SIZE]] : i32 -> tensor<8xi32>
// CHECK: %[[END:.*]] = arith.addi %[[DST_OFFSET_TENSOR]], %[[SIZE_TENSOR]] : tensor<8xi32>
// CHECK: %[[AFTER_START:.*]] = arith.cmpi sge, %[[RANGE]], %[[DST_OFFSET_TENSOR]] : tensor<8xi32>
// CHECK: %[[BEFORE_END:.*]] = arith.cmpi slt, %[[RANGE]], %[[END]] : tensor<8xi32>
// CHECK: %[[MASK:.*]] = arith.andi %[[AFTER_START]], %[[BEFORE_END]] : tensor<8xi1>
// CHECK: %[[LOCAL_INDEX:.*]] = arith.subi %[[RANGE]], %[[DST_OFFSET_TENSOR]] : tensor<8xi32>
// CHECK: %[[SRC_OFFSET:.*]] = arith.trunci %[[SRC_DESC_OFFSET]] : i64 to i32
// CHECK: %[[SRC_OFFSET_TENSOR:.*]] = tt.splat %[[SRC_OFFSET]] : i32 -> tensor<8xi32>
// CHECK: %[[SOURCE_INDEX:.*]] = arith.addi %[[SRC_OFFSET_TENSOR]], %[[LOCAL_INDEX]] : tensor<8xi32>
// CHECK: %[[SOURCE_PTRS:.*]] = tt.addptr {{.*}}, %[[SOURCE_INDEX]]
// CHECK: %[[PAD:.*]] = tt.splat {{.*}} : i64 -> tensor<8xi64>
// CHECK: %[[LOADED:.*]] = tt.load %[[SOURCE_PTRS]], %[[MASK]], %[[PAD]] evictionPolicy = evict_first
// CHECK: tt.store {{.*}}, %[[LOADED]]
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: bufferization.to_tensor
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: tt.return
module {
  func.func @padded_load(%src: memref<?xi64>, %srcOffset: index,
                         %dstOffset: index, %size: index, %needsInit: i1,
                         %out: memref<8xi64>) {
    %c0 = arith.constant 0 : i64
    %c4 = arith.constant 4 : index
    %srcParent = memref.reinterpret_cast %src to offset: [0], sizes: [8],
      strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    %srcView = memref.subview %srcParent[%srcOffset] [%size] [1]
      : memref<8xi64, strided<[1]>>
        to memref<?xi64, strided<[1], offset: ?>>
    %dst = memref.alloc() : memref<8xi64>
    %dstView = memref.subview %dst[%dstOffset] [%size] [1]
      : memref<8xi64> to memref<?xi64, strided<[1], offset: ?>>
    %leftPad = arith.remui %dstOffset, %c4 : index
    hivm.hir.load
        ins(%srcView : memref<?xi64, strided<[1], offset: ?>>)
        outs(%dstView : memref<?xi64, strided<[1], offset: ?>>)
        pad_mode = <PadValue>
        pad_value = %c0 : i64
        left_padding_num = %leftPad : index
        init_out_buffer = true
        init_condition = %needsInit : i1
        eviction_policy = <EvictFirst>
    %loaded = bufferization.to_tensor %dst restrict writable
      : memref<8xi64>
    hivm.hir.store ins(%loaded : tensor<8xi64>)
                   outs(%out : memref<8xi64>)
    return
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module {
  func.func @padded_load_2d(%src: memref<4x4xi32>, %dst: memref<4x4xi32>,
                            %size: index) {
    %c0 = arith.constant 0 : i32
    %srcView = memref.subview %src[0, 0] [4, %size] [1, 1]
      : memref<4x4xi32> to memref<4x?xi32, strided<[4, 1]>>
    %dstView = memref.subview %dst[0, 0] [4, %size] [1, 1]
      : memref<4x4xi32> to memref<4x?xi32, strided<[4, 1]>>
    // expected-error@+2 {{padded load requires matching static 1D buffers}}
    // expected-error@+1 {{failed to legalize operation 'hivm.hir.load' that was explicitly marked illegal}}
    hivm.hir.load ins(%srcView : memref<4x?xi32, strided<[4, 1]>>)
                  outs(%dstView : memref<4x?xi32, strided<[4, 1]>>)
                  pad_mode = <PadValue> pad_value = %c0 : i32
                  init_out_buffer = true
    %tensor = bufferization.to_tensor %dst restrict writable
      : memref<4x4xi32>
    return
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module {
  func.func @padded_load_without_init(%src: memref<8xi32>,
                                      %dst: memref<8xi32>, %size: index) {
    %c0 = arith.constant 0 : i32
    %srcView = memref.subview %src[0] [%size] [1]
      : memref<8xi32> to memref<?xi32, strided<[1]>>
    %dstView = memref.subview %dst[0] [%size] [1]
      : memref<8xi32> to memref<?xi32, strided<[1]>>
    // expected-error@+2 {{unsupported padded load shape or padding mode}}
    // expected-error@+1 {{failed to legalize operation 'hivm.hir.load' that was explicitly marked illegal}}
    hivm.hir.load ins(%srcView : memref<?xi32, strided<[1]>>)
                  outs(%dstView : memref<?xi32, strided<[1]>>)
                  pad_mode = <PadValue> pad_value = %c0 : i32
                  init_out_buffer = false
    %tensor = bufferization.to_tensor %dst restrict writable
      : memref<8xi32>
    return
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module {
  func.func @padded_load_without_to_tensor(%src: memref<8xi32>,
                                           %dst: memref<8xi32>, %size: index) {
    %c0 = arith.constant 0 : i32
    %srcView = memref.subview %src[0] [%size] [1]
      : memref<8xi32> to memref<?xi32, strided<[1]>>
    %dstView = memref.subview %dst[0] [%size] [1]
      : memref<8xi32> to memref<?xi32, strided<[1]>>
    // expected-error@+2 {{padded load destination has no parent to_tensor user}}
    // expected-error@+1 {{failed to legalize operation 'hivm.hir.load' that was explicitly marked illegal}}
    hivm.hir.load ins(%srcView : memref<?xi32, strided<[1]>>)
                  outs(%dstView : memref<?xi32, strided<[1]>>)
                  pad_mode = <PadValue> pad_value = %c0 : i32
                  init_out_buffer = true
    return
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module {
  func.func @padded_load_non_contiguous(%src: memref<8xi32>,
                                        %dst: memref<8xi32>) {
    %c0 = arith.constant 0 : i32
    %srcView = memref.subview %src[0] [4] [2]
      : memref<8xi32> to memref<4xi32, strided<[2]>>
    %dstView = memref.subview %dst[0] [4] [2]
      : memref<8xi32> to memref<4xi32, strided<[2]>>
    // expected-error@+2 {{invalid padded load descriptors}}
    // expected-error@+1 {{failed to legalize operation 'hivm.hir.load' that was explicitly marked illegal}}
    hivm.hir.load ins(%srcView : memref<4xi32, strided<[2]>>)
                  outs(%dstView : memref<4xi32, strided<[2]>>)
                  pad_mode = <PadValue> pad_value = %c0 : i32
                  init_out_buffer = true
    %tensor = bufferization.to_tensor %dst restrict writable
      : memref<8xi32>
    return
  }
}

// -----

// A dead allocation must disappear rather than leak into function conversion.
// CHECK-LABEL: tt.func @dead_alloc_tensor
// CHECK-NOT: bufferization.alloc_tensor
// CHECK: tt.return
module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @dead_alloc_tensor() {
    %dead = bufferization.alloc_tensor() : tensor<4xi32>
    return
  }
}

// -----

// A supported live allocation keeps the same uninitialized tensor semantics.
// CHECK-LABEL: tt.func @static_alloc_tensor
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK-NOT: bufferization.alloc_tensor
// CHECK: tt.store {{.*}}, %[[EMPTY]]
// CHECK: tt.return
module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @static_alloc_tensor(%out: memref<4xi32>) {
    %alloc = bufferization.alloc_tensor() : tensor<4xi32>
    hivm.hir.store ins(%alloc : tensor<4xi32>) outs(%out : memref<4xi32>)
    return
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @dynamic_alloc_tensor(%size: index) -> tensor<?xi32> {
    // expected-error@+2 {{only static allocations without a copy are supported}}
    // expected-error@+1 {{failed to legalize operation 'bufferization.alloc_tensor' that was explicitly marked illegal}}
    %alloc = bufferization.alloc_tensor(%size) : tensor<?xi32>
    return %alloc : tensor<?xi32>
  }
}

// -----

// expected-error@+1 {{Stage1 failed: HIVM/Bufferization Op conversion failed}}
module attributes {
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func @copy_alloc_tensor(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    // expected-error@+2 {{only static allocations without a copy are supported}}
    // expected-error@+1 {{failed to legalize operation 'bufferization.alloc_tensor' that was explicitly marked illegal}}
    %alloc = bufferization.alloc_tensor() copy(%arg0) : tensor<4xi32>
    return %alloc : tensor<4xi32>
  }
}
