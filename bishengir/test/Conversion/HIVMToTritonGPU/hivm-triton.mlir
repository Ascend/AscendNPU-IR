// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: tt.func @simple_indirect_load_kernel_scope_0(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i64>, %arg6: !tt.ptr<i64>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f32>, %arg11: !tt.ptr<f32>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: !tt.ptr<f32>, %arg17: !tt.ptr<f32>, %arg18: i64, %arg19: i64, %arg20: i64)
// CHECK-NEXT: %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK-NEXT: %3 = tt.load %2 evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK-NEXT: %4 = tensor.empty() : tensor<8xf32>
// CHECK-NEXT: %5 = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %6 = tt.addptr %5, %3 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
// CHECK-NEXT: %7 = tt.load %6 evictionPolicy = evict_last : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: %8 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %9 = tt.splat %arg16 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
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
// CHECK-LABEL: tt.func @load_check_strided_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !tt.ptr<f32>, %arg8: !tt.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !tt.ptr<f32>, %arg13: !tt.ptr<f32>, %arg14: i64, %arg15: i64, %arg16: i64) {
// CHECK-NEXT: %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %1 = tt.reshape %0 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %cst = arith.constant dense<64> : tensor<16x1xi32>
// CHECK-NEXT: %2 = arith.muli %1, %cst : tensor<16x1xi32>
// CHECK-NEXT: %cst_0 = arith.constant dense<32> : tensor<16x1xi32>
// CHECK-NEXT: %3 = arith.addi %2, %cst_0 : tensor<16x1xi32>
// CHECK-NEXT: %4 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NEXT: %5 = tt.addptr %4, %3 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
// CHECK-NEXT: %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %7 = tt.reshape %6 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %cst_1 = arith.constant dense<4> : tensor<1x16xi32>
// CHECK-NEXT: %8 = arith.muli %7, %cst_1 : tensor<1x16xi32>
// CHECK-NEXT: %9 = tt.broadcast %5 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT: %11 = tt.addptr %9, %10 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
// CHECK-NEXT: %12 = tt.load %11 evictionPolicy = evict_first : tensor<16x16x!tt.ptr<f32>>
// CHECK-NEXT: %13 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT: %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %15 = tt.reshape %14 : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT: %cst_2 = arith.constant dense<64> : tensor<16x1xi32>
// CHECK-NEXT: %16 = arith.muli %15, %cst_2 : tensor<16x1xi32>
// CHECK-NEXT: %cst_3 = arith.constant dense<32> : tensor<16x1xi32>
// CHECK-NEXT: %17 = arith.addi %16, %cst_3 : tensor<16x1xi32>
// CHECK-NEXT: %18 = tt.splat %arg12 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
// CHECK-NEXT: %19 = tt.addptr %18, %17 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
// CHECK-NEXT: %20 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT: %21 = tt.reshape %20 : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT: %cst_4 = arith.constant dense<4> : tensor<1x16xi32>
// CHECK-NEXT: %22 = arith.muli %21, %cst_4 : tensor<1x16xi32>
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
// CHECK-LABEL: tt.func @load_check_dynamic_reinterpret_2d
// CHECK: [[ROW_RANGE:%.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK: [[ROW:%.*]] = tt.reshape [[ROW_RANGE]] : tensor<2xi32> -> tensor<2x1xi32>
// CHECK: [[STRIDE:%.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: [[STRIDE_SPLAT:%.*]] = tt.splat [[STRIDE]] : i32 -> tensor<2x1xi32>
// CHECK: [[ROW_STRIDE:%.*]] = arith.muli [[ROW]], [[STRIDE_SPLAT]] : tensor<2x1xi32>
// CHECK: [[BASE_OFFSET:%.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: [[BASE_SPLAT:%.*]] = tt.splat [[BASE_OFFSET]] : i32 -> tensor<2x1xi32>
// CHECK: [[WITH_ROW:%.*]] = arith.addi [[ROW_STRIDE]], [[BASE_SPLAT]] : tensor<2x1xi32>
// CHECK: [[BASE_PTR:%.*]] = tt.splat %{{.*}} : !tt.ptr<i64> -> tensor<2x1x!tt.ptr<i64>>
// CHECK: [[ROW_PTR:%.*]] = tt.addptr [[BASE_PTR]], [[WITH_ROW]] : tensor<2x1x!tt.ptr<i64>>, tensor<2x1xi32>
// CHECK: [[COL_RANGE:%.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK: [[COL:%.*]] = tt.reshape [[COL_RANGE]] : tensor<4xi32> -> tensor<1x4xi32>
// CHECK: [[PTR_BCAST:%.*]] = tt.broadcast [[ROW_PTR]] : tensor<2x1x!tt.ptr<i64>> -> tensor<2x4x!tt.ptr<i64>>
// CHECK: [[COL_BCAST:%.*]] = tt.broadcast [[COL]] : tensor<1x4xi32> -> tensor<2x4xi32>
// CHECK: [[PTRS:%.*]] = tt.addptr [[PTR_BCAST]], [[COL_BCAST]] : tensor<2x4x!tt.ptr<i64>>, tensor<2x4xi32>
// CHECK: tt.load [[PTRS]] evictionPolicy = evict_first : tensor<2x4x!tt.ptr<i64>>

module {
  func.func @load_check_dynamic_reinterpret_2d(%arg0: memref<?xi64>, %arg1: memref<2x4xi64>, %arg2: index, %arg3: index) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg2], sizes: [2, 4], strides: [%arg3, 1]
        : memref<?xi64> to memref<2x4xi64, strided<[?, 1], offset: ?>>
    hivm.hir.load ins(%reinterpret_cast : memref<2x4xi64, strided<[?, 1], offset: ?>>)
                  outs(%arg1 : memref<2x4xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<2x4xi64>
    return
  }
}



// -----
// CHECK-LABEL: tt.func @check_store_atomicadd(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @check_store_atomicmax(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @check_store_atomicmin(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @check_store_atomicand(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @check_store_atomicor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @check_store_atomicxor(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<i32>, %arg6: !tt.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<i32>, %arg11: !tt.ptr<i32>, %arg12: i64, %arg13: i64, %arg14: i64)
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
// CHECK-LABEL: tt.func @restore_get_block_idx
// CHECK-NEXT: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[PID]], %[[C1]] : i32
// CHECK-NEXT: tt.return

module {
  func.func @restore_get_block_idx() {
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.addi %1, %c1_i32 : i32
    return
  }
}
