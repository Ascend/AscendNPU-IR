// RUN: bishengir-opt %s -hivm-clone-tensor-empty -split-input-file | FileCheck %s
// RUN: bishengir-opt -one-shot-bufferize="allow-return-allocs-from-loops" %s | FileCheck %s --check-prefix=NO-CLONE
// RUN: bishengir-opt -hivm-clone-tensor-empty -one-shot-bufferize="allow-return-allocs-from-loops" %s | FileCheck %s --check-prefix=CLONE

// -----
module {
  func.func @test_clone_tensor_fixpipe(%arg1 : tensor<16x16xf16>,
                                     %arg2 : tensor<16x16xf16>,
                                     %arg3 : tensor<16x16xf16>) -> tensor<16x16xf16> {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %0 = tensor.empty() : tensor<16x16xf16>
    // CHECK: tensor.empty() : tensor<16x16xf16>
    %1 = hivm.hir.copy ins(%arg1 : tensor<16x16xf16>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    // CHECK: tensor.empty() : tensor<16x16xf16>
    %2 = hivm.hir.copy ins(%arg2 : tensor<16x16xf16>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    // CHECK: tensor.empty() : tensor<16x16xf16>
    %4 = hivm.hir.mmadL1 ins(%1, %2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    // CHECK: tensor.empty() : tensor<16x16xf16>
    %5 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<16x16xf16>) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %6 = hivm.hir.copy ins(%5 : tensor<16x16xf16>) outs(%arg3 : tensor<16x16xf16>) -> tensor<16x16xf16>
    return %6 : tensor<16x16xf16>
  }
}

// -----
module {
  func.func @test_clone_tensor_empty_static(%arg1 : tensor<4096xf16>,
                                     %arg2 : tensor<4096xf16>,
                                     %arg3 : tensor<4096xf16>) -> tensor<4096xf16> {
    %0 = tensor.empty() : tensor<4096xf16>
    %2 = tensor.empty() : tensor<4096xf16>
    %6 = tensor.empty() : tensor<4096xf16>
    // CHECK: tensor.empty() : tensor<4096xf16>
    %1 = hivm.hir.copy ins(%arg1 : tensor<4096xf16>) outs(%0 : tensor<4096xf16>) -> tensor<4096xf16>
    // CHECK: tensor.empty() : tensor<4096xf16>
    %3 = hivm.hir.copy ins(%arg2 : tensor<4096xf16>) outs(%2 : tensor<4096xf16>) -> tensor<4096xf16>
    // CHECK: tensor.empty() : tensor<4096xf16>
    %4 = hivm.hir.vmul ins(%1, %3 : tensor<4096xf16>, tensor<4096xf16>)
    outs(%6 : tensor<4096xf16>) -> tensor<4096xf16>
    // CHECK: tensor.empty() : tensor<4096xf16>
    %5 = hivm.hir.vrec ins(%4 : tensor<4096xf16>) outs(%0 : tensor<4096xf16>) -> tensor<4096xf16>
    %7 = hivm.hir.copy ins(%5 : tensor<4096xf16>) outs(%arg3 : tensor<4096xf16>) -> tensor<4096xf16>
    return %5 : tensor<4096xf16>
  }
}

// -----
module {
  func.func @test_clone_tensor_empty_dynamic(%arg0 : index, %arg1 : tensor<?x4096xf16>,
                                     %arg2 : tensor<?x4096xf16>,
                                     %arg3 : tensor<?x4096xf16>) -> tensor<?x4096xf16> {
    %0 = tensor.empty(%arg0) : tensor<?x4096xf16>
    %2 = tensor.empty(%arg0) : tensor<?x4096xf16>
    %6 = tensor.empty(%arg0) : tensor<?x4096xf16>
    // CHECK: tensor.empty(%arg0) : tensor<?x4096xf16>
    %1 = hivm.hir.copy ins(%arg1 : tensor<?x4096xf16>) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    // CHECK: tensor.empty(%arg0) : tensor<?x4096xf16>
    %3 = hivm.hir.copy ins(%arg2 : tensor<?x4096xf16>) outs(%2 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    // CHECK: tensor.empty(%arg0) : tensor<?x4096xf16>
    %4 = hivm.hir.vmul ins(%1, %3 : tensor<?x4096xf16>, tensor<?x4096xf16>)
    outs(%6 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    // CHECK: tensor.empty(%arg0) : tensor<?x4096xf16>
    %5 = hivm.hir.vrec ins(%4 : tensor<?x4096xf16>) outs(%0 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    %7 = hivm.hir.copy ins(%5 : tensor<?x4096xf16>) outs(%arg3 : tensor<?x4096xf16>) -> tensor<?x4096xf16>
    return %5 : tensor<?x4096xf16>
  }
}

// -----
// NO-CLONE-LABEL: test_sink_empty
// CLONE-LABEL: test_sink_empty
func.func @test_sink_empty() -> tensor<16xf32>{
  %c0 = arith.constant 0 : i32
  %ci = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32

  %empty = tensor.empty() : tensor<16xf32>
  %init = hivm.hir.vbrc ins(%ci:f32) outs(%empty:tensor<16xf32>) -> tensor<16xf32>

  %ret = scf.for %i = %c0 to %c1 step %c2 iter_args(%arg = %init) -> tensor<16xf32> : i32 {
    %fi = arith.uitofp %i : i32 to f32
    %res = hivm.hir.vbrc ins(%fi:f32) outs(%empty:tensor<16xf32>) -> tensor<16xf32>
    // NOTE: if this check fails, then the pass is no longer needed before one-shot-bufferize
    // NO-CLONE: memref.copy

    // CLONE-NOT: memref.copy
    scf.yield %res : tensor<16xf32>
  }
  
  return %ret : tensor<16xf32>
}

// -----
// CHECK-LABEL: func.func @rewrite_tensor_before_use
// CHECK: scf.for %[[IV:.*]] = %c0 to %c4096 step %c1
// CHECK:   %[[EMP:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:   %[[INS:.*]] = tensor.insert %c42_i32 into %[[EMP]][%c0] : tensor<1xi32>

func.func @rewrite_tensor_before_use(%arg0: memref<?xi32>) {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4096 = arith.constant 4096 : index
  %7 = tensor.empty() : tensor<1xi32>
  scf.for %arg1 = %c0 to %c4096 step %c1 {
    %inserted = tensor.insert %c42_i32 into %7[%c0] : tensor<1xi32>
    %reinterpret_cast =
      memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      hivm.hir.store ins(%inserted : tensor<1xi32>) outs(%reinterpret_cast : memref<1xi32, strided<[1], offset: ?>>)
    }
  return
}
