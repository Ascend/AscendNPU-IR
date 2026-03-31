// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(fuse-reduction-into-loop))" %s -split-input-file | FileCheck %s

// Basic case: single 2D accumulator reduced to 1D after the loop.
//
// CHECK-LABEL: func.func @single_reduce
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[INIT1D:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32xf32>)
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT1D]])
// CHECK:         %[[REDUCE:.*]] = linalg.reduce ins({{.*}} : tensor<256x32xf32>) outs(%[[INIT1D]] : tensor<32xf32>) dimensions = [0]
// CHECK:           arith.addf
// CHECK:           linalg.yield
// CHECK:         %[[ADD:.*]] = arith.addf %[[ACC]], %[[REDUCE]]
// CHECK:         scf.yield {{.*}}%[[ADD]]
// CHECK:       arith.truncf %[[FOR]] : tensor<32xf32> to tensor<32xbf16>
// CHECK-NOT:   linalg.reduce
func.func @single_reduce(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>,
                          %N: i32, %stride: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c256_i32 = arith.constant 256 : i32
  %cst = arith.constant 0.000000e+00 : f32

  // 2D zero-filled accumulator
  %empty2d = tensor.empty() : tensor<256x32xf32>
  %init2d = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<256x32xf32>) -> tensor<256x32xf32>

  %c255_i32 = arith.constant 255 : i32
  %nPlus = arith.addi %N, %c255_i32 : i32
  %trips = arith.divsi %nPlus, %c256_i32 : i32

  %result = scf.for %iv = %c0_i32 to %trips step %c1_i32
      iter_args(%acc = %init2d) -> (tensor<256x32xf32>) : i32 {
    // Simulate loading a 256x32 tile
    %alloc = memref.alloc() : memref<256x32xbf16>
    %tile_bf16 = bufferization.to_tensor %alloc restrict writable : memref<256x32xbf16>
    %tile = arith.extf %tile_bf16 : tensor<256x32xbf16> to tensor<256x32xf32>

    // Accumulate in 2D
    %new_acc = arith.addf %acc, %tile : tensor<256x32xf32>
    scf.yield %new_acc : tensor<256x32xf32>
  }

  // Post-loop reduction from 2D to 1D
  %empty1d = tensor.empty() : tensor<32xf32>
  %init1d = linalg.fill ins(%cst : f32) outs(%empty1d : tensor<32xf32>) -> tensor<32xf32>
  %reduced = linalg.reduce ins(%result : tensor<256x32xf32>)
                           outs(%init1d : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }

  %out = arith.truncf %reduced : tensor<32xf32> to tensor<32xbf16>
  %idx = arith.index_cast %stride : i32 to index
  %reinterpret = memref.reinterpret_cast %arg1 to offset: [%idx],
      sizes: [32], strides: [1] : memref<?xbf16> to memref<32xbf16, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %out in writable %reinterpret
      : (tensor<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>) -> ()
  return
}

// -----
// Two accumulators reduced to 1D after the loop (matches the reduceOrig
// pattern from fused_swiglu_bwd_b_kernel).
//
// CHECK-LABEL: func.func @dual_reduce
// CHECK:       %[[INIT1:.*]] = linalg.fill {{.*}} -> tensor<32xf32>
// CHECK:       %[[INIT2:.*]] = linalg.fill {{.*}} -> tensor<32xf32>
// CHECK:       %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[A1:.*]] = %[[INIT1]], %[[A2:.*]] = %[[INIT2]])
// CHECK:         %[[R1:.*]] = linalg.reduce ins({{.*}} : tensor<256x32xf32>) outs({{.*}} : tensor<32xf32>) dimensions = [0]
// CHECK:         %[[S1:.*]] = arith.addf %[[A1]], %[[R1]]
// CHECK:         %[[R2:.*]] = linalg.reduce ins({{.*}} : tensor<256x32xf32>) outs({{.*}} : tensor<32xf32>) dimensions = [0]
// CHECK:         %[[S2:.*]] = arith.addf %[[A2]], %[[R2]]
// CHECK:         scf.yield %[[S1]], %[[S2]]
// CHECK:       arith.truncf %[[FOR]]#0 : tensor<32xf32>
// CHECK:       arith.truncf %[[FOR]]#1 : tensor<32xf32>
// CHECK-NOT:   linalg.reduce
func.func @dual_reduce(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>,
                        %N: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32

  %empty2d = tensor.empty() : tensor<256x32xf32>
  %zero2d = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<256x32xf32>) -> tensor<256x32xf32>
  %ones2d = linalg.fill ins(%cst1 : f32) outs(%empty2d : tensor<256x32xf32>) -> tensor<256x32xf32>

  %result:2 = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%acc0 = %zero2d, %acc1 = %zero2d) -> (tensor<256x32xf32>, tensor<256x32xf32>) : i32 {
    // Simulate computation producing two 2D values.
    %alloc0 = memref.alloc() : memref<256x32xbf16>
    %t0_bf16 = bufferization.to_tensor %alloc0 restrict writable : memref<256x32xbf16>
    %val0 = arith.extf %t0_bf16 : tensor<256x32xbf16> to tensor<256x32xf32>

    // Simulate a sigmoid-like computation producing a second 2D value.
    %neg = arith.subf %zero2d, %val0 : tensor<256x32xf32>
    %ex  = math.exp %neg : tensor<256x32xf32>
    %denom = arith.addf %ex, %ones2d : tensor<256x32xf32>
    %sig = arith.divf %val0, %denom : tensor<256x32xf32>
    %val1 = arith.truncf %sig : tensor<256x32xf32> to tensor<256x32xbf16>
    %val1_f32 = arith.extf %val1 : tensor<256x32xbf16> to tensor<256x32xf32>

    // Accumulate both values in 2D.
    %new0 = arith.addf %acc0, %val0 : tensor<256x32xf32>
    %new1 = arith.addf %acc1, %val1_f32 : tensor<256x32xf32>
    scf.yield %new0, %new1 : tensor<256x32xf32>, tensor<256x32xf32>
  }

  // Post-loop reductions: 2D -> 1D.
  %empty1d = tensor.empty() : tensor<32xf32>
  %zero1d = linalg.fill ins(%cst : f32) outs(%empty1d : tensor<32xf32>) -> tensor<32xf32>

  %red0 = linalg.reduce ins(%result#0 : tensor<256x32xf32>)
                         outs(%zero1d : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }

  %red1 = linalg.reduce ins(%result#1 : tensor<256x32xf32>)
                         outs(%zero1d : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }

  %out0 = arith.truncf %red0 : tensor<32xf32> to tensor<32xbf16>
  %out1 = arith.truncf %red1 : tensor<32xf32> to tensor<32xbf16>

  %c0 = arith.constant 0 : index
  %reinterpret0 = memref.reinterpret_cast %arg0 to offset: [%c0],
      sizes: [32], strides: [1] : memref<?xbf16> to memref<32xbf16, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %out0 in writable %reinterpret0
      : (tensor<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>) -> ()
  %reinterpret1 = memref.reinterpret_cast %arg1 to offset: [%c0],
      sizes: [32], strides: [1] : memref<?xbf16> to memref<32xbf16, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %out1 in writable %reinterpret1
      : (tensor<32xbf16>, memref<32xbf16, strided<[1], offset: ?>>) -> ()
  return
}

// -----
// Negative test: the reduction body uses mulf instead of addf.
// The pass should NOT transform this.
//
// CHECK-LABEL: func.func @no_transform_mulf_reduction
// CHECK:       scf.for
// CHECK:         arith.addf {{.*}} : tensor<128x64xf32>
// CHECK:       linalg.reduce
// CHECK:         arith.mulf
func.func @no_transform_mulf_reduction() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant 0.000000e+00 : f32

  %empty2d = tensor.empty() : tensor<128x64xf32>
  %zero2d = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<128x64xf32>) -> tensor<128x64xf32>

  %result = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%acc = %zero2d) -> (tensor<128x64xf32>) : i32 {
    %alloc = memref.alloc() : memref<128x64xf32>
    %tile = bufferization.to_tensor %alloc restrict writable : memref<128x64xf32>
    %new_acc = arith.addf %acc, %tile : tensor<128x64xf32>
    scf.yield %new_acc : tensor<128x64xf32>
  }

  // Multiplicative reduction -- should NOT be fused.
  %empty1d = tensor.empty() : tensor<64xf32>
  %one_init = arith.constant 1.000000e+00 : f32
  %init1d = linalg.fill ins(%one_init : f32) outs(%empty1d : tensor<64xf32>) -> tensor<64xf32>
  %reduced = linalg.reduce ins(%result : tensor<128x64xf32>)
                           outs(%init1d : tensor<64xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.mulf %in, %init : f32
      linalg.yield %s : f32
    }
  return
}

// -----
// Negative test: iter_arg has multiple uses (used outside the addf too).
// The pass should NOT transform this.
//
// CHECK-LABEL: func.func @no_transform_multi_use_iterarg
// CHECK:       scf.for
// CHECK:         arith.addf {{.*}} : tensor<64x32xf32>
// CHECK:         arith.mulf {{.*}} : tensor<64x32xf32>
// CHECK:       linalg.reduce
func.func @no_transform_multi_use_iterarg() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant 0.000000e+00 : f32

  %empty2d = tensor.empty() : tensor<64x32xf32>
  %zero2d = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<64x32xf32>) -> tensor<64x32xf32>

  %result = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%acc = %zero2d) -> (tensor<64x32xf32>) : i32 {
    %alloc = memref.alloc() : memref<64x32xf32>
    %tile = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
    // iter_arg %acc is used in BOTH addf AND mulf -> not safe to transform.
    %new_acc = arith.addf %acc, %tile : tensor<64x32xf32>
    %side = arith.mulf %acc, %tile : tensor<64x32xf32>
    scf.yield %new_acc : tensor<64x32xf32>
  }

  %empty1d = tensor.empty() : tensor<32xf32>
  %init1d = linalg.fill ins(%cst : f32) outs(%empty1d : tensor<32xf32>) -> tensor<32xf32>
  %reduced = linalg.reduce ins(%result : tensor<64x32xf32>)
                           outs(%init1d : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }
  return
}

// -----
// Negative test: the for result has multiple users (reduce + something else).
// The pass should NOT transform this.
//
// CHECK-LABEL: func.func @no_transform_multi_user_result
// CHECK:       %[[R:.*]] = scf.for
// CHECK:       linalg.reduce ins(%[[R]]
// CHECK:       arith.negf %[[R]]
func.func @no_transform_multi_user_result() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant 0.000000e+00 : f32

  %empty2d = tensor.empty() : tensor<64x32xf32>
  %zero2d = linalg.fill ins(%cst : f32) outs(%empty2d : tensor<64x32xf32>) -> tensor<64x32xf32>

  %result = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32
      iter_args(%acc = %zero2d) -> (tensor<64x32xf32>) : i32 {
    %alloc = memref.alloc() : memref<64x32xf32>
    %tile = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32>
    %new_acc = arith.addf %acc, %tile : tensor<64x32xf32>
    scf.yield %new_acc : tensor<64x32xf32>
  }

  // Result has two users: the reduce AND negf.
  %empty1d = tensor.empty() : tensor<32xf32>
  %init1d = linalg.fill ins(%cst : f32) outs(%empty1d : tensor<32xf32>) -> tensor<32xf32>
  %reduced = linalg.reduce ins(%result : tensor<64x32xf32>)
                           outs(%init1d : tensor<32xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }
  %other = arith.negf %result : tensor<64x32xf32>
  return
}
