// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: tt.func @simple_indirect_load_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: index, %arg13: index, %arg14: index, %arg15: i32, %arg16: !tt.ptr<f32>, %arg17: !tt.ptr<f32>, %arg18: index, %arg19: index, %arg20: index)
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %5 = tt.addptr %4, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: %6 = tt.load %5 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %7 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %8 = tt.splat %arg16 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %9 = tt.addptr %8, %7 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: tt.store %9, %6 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_load_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %0 : tensor<8xi64>, %arg3 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>, isVolatile = false} -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %1 : tensor<8xf32>)
    return
  }
}

// -----
// CHECK-LABEL: tt.func @simple_indirect_store_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: index, %arg13: index, %arg14: index, %arg15: !tt.ptr<f32>, %arg16: !tt.ptr<f32>, %arg17: index, %arg18: index, %arg19: index, %arg20: i32)
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %5 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// CHECK-NEXT: %7 = tt.load %6 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %8 = tt.splat %arg15 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %9 = tt.addptr %8, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: tt.store %9, %7 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: tt.return

module {
  func.func @simple_indirect_store_kernel_scope_0(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<8xf32>, %arg3: memref<?xf32>, %arg4: i32) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = hivm.hir.local_load ins(%arg2 : memref<8xf32>) -> tensor<8xf32>
    hivm.hir.scatter_store ins(%arg3 : memref<?xf32>, %0 : tensor<8xi64>, %1 : tensor<8xf32>, %arg4 : i32) {cache = 1 : i32, evict = #hivm.evictionpolicy<EvictLast>}
    return
  }
}


// -----
// CHECK-LABEL: tt.func @load_check_strided_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<f32>, %arg6: !tt.ptr<f32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK-NEXT: %1 = tt.reshape %0 : tensor<256xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT: %2 = tt.splat %c0_i32 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %4 = tt.reshape %3 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %5 = tt.broadcast %4 : tensor<16x1xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT: %6 = tt.splat %c64_i32 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %7 = arith.muli %5, %6 : tensor<16x16xi32>
// CHECK-NEXT: %8 = arith.addi %2, %7 : tensor<16x16xi32>
// CHECK-NEXT: %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %10 = tt.reshape %9 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %11 = tt.broadcast %10 : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT: %12 = tt.splat %c4_i32 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %13 = arith.muli %11, %12 : tensor<16x16xi32>
// CHECK-NEXT: %14 = arith.addi %8, %13 : tensor<16x16xi32>
// CHECK-NEXT: %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT: %15 = tt.splat %c32_i32 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %16 = arith.addi %14, %15 : tensor<16x16xi32>
// CHECK-NEXT: %17 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NEXT: %19 = tt.load %18 evictionPolicy = evict_first : tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %20 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT: %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK-NEXT: %22 = tt.reshape %21 : tensor<256xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c0_i32_0 = arith.constant 0 : i32
// CHECK-NEXT: %23 = tt.splat %c0_i32_0 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %24 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %25 = tt.reshape %24 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %26 = tt.broadcast %25 : tensor<16x1xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c64_i32_1 = arith.constant 64 : i32
// CHECK-NEXT: %27 = tt.splat %c64_i32_1 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %28 = arith.muli %26, %27 : tensor<16x16xi32>
// CHECK-NEXT: %29 = arith.addi %23, %28 : tensor<16x16xi32>
// CHECK-NEXT: %30 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %31 = tt.reshape %30 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %32 = tt.broadcast %31 : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %c4_i32_2 = arith.constant 4 : i32
// CHECK-NEXT: %33 = tt.splat %c4_i32_2 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %34 = arith.muli %32, %33 : tensor<16x16xi32>
// CHECK-NEXT: %35 = arith.addi %29, %34 : tensor<16x16xi32>
// CHECK-NEXT: %c32_i32_3 = arith.constant 32 : i32
// CHECK-NEXT: %36 = tt.splat %c32_i32_3 : i32 -> tensor<16x16xi32>
// CHECK-NEXT: %37 = arith.addi %35, %36 : tensor<16x16xi32>
// CHECK-NEXT: %38 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %39 = tt.addptr %38, %37 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NEXT: tt.store %39, %20 : tensor<16x16x!tt.ptr<f32>>
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
// CHECK-LABEL: tt.func @check_store_atomicadd(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw add, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicadd(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <add>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicmax(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw max, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicmax(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <max>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicmin(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw min, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicmin(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <min>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicand(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw and, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicand(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <and>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw or, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicor(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <or>
  return
}

// -----
// CHECK-LABEL: tt.func @check_store_atomicxor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: index, %arg3: index, %arg4: index, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: index, %arg8: index, %arg9: index, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: index, %arg13: index, %arg14: index)
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.splat %arg5 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: tt.store %5, %3 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %8 = tt.addptr %7, %6 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %9 = tt.load %8 : tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %11 = tt.splat %arg10 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
// CHECK-NEXT: %12 = tt.addptr %11, %10 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
// CHECK-NEXT: %13 = tt.atomic_rmw xor, acq_rel, gpu, %12, %9 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
// CHECK-NEXT: tt.return

func.func @check_store_atomicxor(%arg0: memref<16xi32> , %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
  hivm.hir.load ins(%arg1 : memref<16xi32>) outs(%arg0: memref<16xi32>)
  hivm.hir.store ins(%arg0: memref<16xi32>) outs(%arg2 : memref<16xi32>) atomic = <xor>
  return
}