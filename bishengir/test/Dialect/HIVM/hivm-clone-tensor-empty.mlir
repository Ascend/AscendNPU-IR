// RUN: bishengir-opt %s -hivm-clone-tensor-empty -split-input-file | FileCheck %s
// RUN: bishengir-opt -one-shot-bufferize="allow-return-allocs-from-loops" %s | FileCheck %s --check-prefix=NO-CLONE
// RUN: bishengir-opt -hivm-clone-tensor-empty -one-shot-bufferize="allow-return-allocs-from-loops" %s | FileCheck %s --check-prefix=CLONE

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
