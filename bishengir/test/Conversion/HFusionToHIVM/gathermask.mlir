// RUN: bishengir-opt -convert-hfusion-to-hivm %s | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s | FileCheck %s
// CHECK-LABEL:   func.func @test_gathermask(
// CHECK-SAME:                               %[[VAL_0:.*]]: memref<16xf16>,
// CHECK-SAME:                               %[[VAL_1:.*]]: memref<16xi1>) {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<16xf16>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           hivm.hir.vgathermask ins(%[[VAL_0]] : memref<16xf16>) mask(%[[VAL_1]] : memref<16xi1>) outs(%[[VAL_2]], %[[VAL_3]] : memref<16xf16>, memref<1xi32>)
// CHECK:           return
// CHECK:         }

func.func @test_gathermask(%src:memref<16xf16>, %mask:memref<16xi1>) {
  %init_data = memref.alloc() : memref<16xf16>
  %init_size = memref.alloc() : memref<1xi32>
  hfusion.gather_mask ins(%src, %mask : memref<16xf16>, memref<16xi1>) 
                      outs(%init_data, %init_size : memref<16xf16>, memref<1xi32>)
  return
}



