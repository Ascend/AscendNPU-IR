// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 --hivm-alloc-extra-buffer -split-input-file %s | FileCheck %s

func.func @test_vreduce_temp_buffer() {
  %src = memref.alloc() : memref<10x32xf16>
  %dst = memref.alloc() : memref<10x1xf16>

  // CHECK: hivm.hir.vreduce <sum> ins({{.*}}memref<10x32xf16>) outs({{.*}}memref<10x1xf16>) unsigned_src = false reduce_dims = [1]
  // CHECK-NOT: hivm.hir.vreduce <sum>{{.*}}temp_buffer({{.*}})
  hivm.hir.vreduce <sum> ins(%src : memref<10x32xf16>)
                         outs(%dst : memref<10x1xf16>)
                         unsigned_src = false
                         reduce_dims = [1]

  return
}

// -----

func.func @test_vreduce_temp_buffer_2() {
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %src = memref.alloca() : memref<32x16x8x32xf16>
  %dst = memref.alloca() : memref<1x16x8x32xf16>
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      scf.for %arg2 = %c0 to %c8 step %c1 {
        scf.for %arg3 = %c0 to %c32 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            scf.for %arg5 = %c0 to %c16 step %c1 {
                %src2 = memref.collapse_shape %src [[0], [1, 2], [3]] : memref<32x16x8x32xf16> into memref<32x128x32xf16>
                %dst2 = memref.collapse_shape %dst [[0], [1, 2], [3]] : memref<1x16x8x32xf16> into memref<1x128x32xf16>

                // CHECK: memref.alloca() : memref<1x16x8x32xf16>
                // CHECK: hivm.hir.vreduce <sum> ins({{.*}}memref<32x128x32xf16>) outs({{.*}}memref<1x128x32xf16>) unsigned_src = false reduce_dims = [0]
                // CHECK-NOT: hivm.hir.vreduce <sum>{{.*}}temp_buffer({{.*}})
                hivm.hir.vreduce <sum> ins(%src2 : memref<32x128x32xf16>)
                                       outs(%dst2 : memref<1x128x32xf16>)
                                       unsigned_src = false
                                       reduce_dims = [0]
            }
          }
        }
      }
    }
  }
  return
}
