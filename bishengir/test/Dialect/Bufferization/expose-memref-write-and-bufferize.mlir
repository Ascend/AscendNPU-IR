// RUN: bishengir-opt %s  -hivm-expose-memref-write-to-tensor -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s

func.func @test_to_tensorOp(%arg0: i32, %arg1 : memref<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %alloc = memref.alloc() : memref<256xf16>
    hivm.hir.load ins(%arg1 : memref<256xf16>) outs(%alloc : memref<256xf16>)
    %5 = bufferization.to_tensor %alloc restrict writable : memref<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
  return %3 : tensor<256xf16>
}