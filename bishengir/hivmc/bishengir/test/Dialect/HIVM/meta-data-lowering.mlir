// RUN: bishengir-opt %s -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm="index-bitwidth=0"  -convert-func-to-llvm="index-bitwidth=0" -reconcile-unrealized-casts -o %t_lowered.mlir
// RUN: bishengir-translate %t_lowered.mlir -mlir-to-llvmir -o %t.ll
// RUN: FileCheck %s -v --dump-input always < %t.ll &> %t.verbose

module @test_hacc_attr_metadata_lowering attributes {llvm.data_layout = "e-f32:8:32-f32:8:32-f326:16:32-i64:64-f32:16:32-v16:16-v32:32-n64-S64", llvm.target_triple = "ascend_910-unknown-cce"}  {
  func.func @vadd(%arg0: memref<32x32xf32, 1>, %arg1: memref<32x32xf32, 1>, %arg2: memref<32x32xf32, 1>) attributes {hacc.entry, hacc.always_inline} {
    affine.for %arg4 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg4, %arg3] : memref<32x32xf32, 1>
        %1 = affine.load %arg1[%arg4, %arg3] : memref<32x32xf32, 1>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[%arg4, %arg3] : memref<32x32xf32, 1>
      }
    }
    return
  }
}
// CHECK: ; Function Attrs: alwaysinline
// CHECK: dso_local
// CHECK: !hivm.annotations
