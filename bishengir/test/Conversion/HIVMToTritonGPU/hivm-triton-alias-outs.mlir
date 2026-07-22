// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file -verify-diagnostics | FileCheck %s

// Regression test: lowering a binary HIVM op must not rewrite an aliased
// `outs` tensor operand to the newly created result. Doing so can create
// self-referential Triton/Arith IR such as `%x = arith.ori %x, ...`.

// CHECK-LABEL: tt.func @alias_outs_to_tensor_operand
// CHECK: %[[C3:.*]] = arith.constant 3 : i64
// CHECK: %[[IDX:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK: %[[BASE:.*]] = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
// CHECK: %[[PTRS:.*]] = tt.addptr %[[BASE]], %[[IDX]] : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTRS]] evictionPolicy = evict_first : tensor<8x!tt.ptr<i64>>
// CHECK: %[[BRC3:.*]] = tt.splat %[[C3]] : i64 -> tensor<8xi64>
// CHECK: %[[OR:.*]] = arith.ori %[[LOAD]], %[[BRC3]] : tensor<8xi64>
// CHECK: %[[GBASE:.*]] = tt.splat %arg10 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK: %[[GPTR:.*]] = tt.addptr %[[GBASE]], %[[OR]] : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
module {
  func.func @alias_outs_to_tensor_operand(%arg0: memref<?xi64>, %arg1: memref<8xi64>, %arg2: memref<?xf32>, %arg3: i32, %arg4: memref<8xf32>) attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c3_i64 = arith.constant 3 : i64
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi64> to memref<8xi64, strided<[1]>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi64, strided<[1]>>) outs(%arg1 : memref<8xi64>) eviction_policy = <EvictFirst>
    %0 = bufferization.to_tensor %arg1 restrict writable : memref<8xi64>
    %1 = tensor.empty() : tensor<8xi64>
    %2 = hivm.hir.vbrc ins(%c3_i64 : i64) outs(%1 : tensor<8xi64>) -> tensor<8xi64>
    %3 = hivm.hir.vor ins(%0, %2 : tensor<8xi64>, tensor<8xi64>) outs(%0 : tensor<8xi64>) -> tensor<8xi64>
    %4 = tensor.empty() : tensor<8xf32>
    %5 = hivm.hir.gather_load ins(%arg2 : memref<?xf32>, %3 : tensor<8xi64>, %arg3 : i32) outs(%4 : tensor<8xf32>) -> tensor<8xf32>
    hivm.hir.local_store ins(%arg4 : memref<8xf32>, %5 : tensor<8xf32>)
    return
  }
}

// -----

// CHECK-LABEL: tt.func @alias_outs_previous_result
// CHECK: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: %[[CNEG1:.*]] = arith.constant -1 : i32
// CHECK: %[[BRC4:.*]] = tt.splat %[[C4]] : i32 -> tensor<8xi32>
// CHECK: %[[BRCNEG1:.*]] = tt.splat %[[CNEG1]] : i32 -> tensor<8xi32>
// CHECK: %[[MUL:.*]] = arith.muli %[[BRC4]], %[[BRCNEG1]] : tensor<8xi32>
// CHECK: tt.return
module {
  func.func @alias_outs_previous_result() attributes {no_inline, outline, vector_function, vf_mode = #hivm.vf_mode<SIMT>} {
    %c4_i32 = arith.constant 4 : i32
    %cneg1_i32 = arith.constant -1 : i32
    %0 = tensor.empty() : tensor<8xi32>
    %1 = hivm.hir.vbrc ins(%c4_i32 : i32) outs(%0 : tensor<8xi32>) -> tensor<8xi32>
    %2 = hivm.hir.vmul ins(%1, %cneg1_i32 : tensor<8xi32>, i32) outs(%1 : tensor<8xi32>) -> tensor<8xi32>
    return
  }
}
