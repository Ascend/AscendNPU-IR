// RUN: bishengir-opt -convert-hfusion-to-hivm %s | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s
// CHECK-LABEL: test_gathermask

func.func @test_gathermask(%src:memref<16xf16>, %mask:memref<16xi8>) {
  %init = memref.alloc() : memref<16xf16>
  // CHECK: vgathermask
  hfusion.gather_mask ins(%src, %mask : memref<16xf16>, memref<16xi8>) outs(%init:memref<16xf16>)
  return
}