// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend310B4 -hivm-lift-lowest-stride -split-input-file %s | FileCheck %s

module {
  func.func @test_lift_vreduce_ops(
      %src : memref<?x32xf16, strided<[?, 16]>>,
      %dst : memref<?x1xf16, strided<[?, 16]>>,
      %dst1 : memref<?x1xi32, strided<[?, 16]>>,
      %alloc_8 : memref<16384xf32>
      ) {
    // CHECK: hivm.hir.vreduce <sum> ins(%{{.*}} : memref<?x32x1xf16, strided<[?, 16, 1]>>) outs(%{{.*}} : memref<?x1x1xf16, strided<[?, 16, 1]>>) unsigned_src = false reduce_dims = [1]
    hivm.hir.vreduce <sum> ins(%src : memref<?x32xf16, strided<[?, 16]>>) outs(%dst : memref<?x1xf16, strided<[?, 16]>>) unsigned_src = false reduce_dims = [1]
    // CHECK: hivm.hir.vreduce <max_with_index> ins(%{{.*}} : memref<?x32x1xf16, strided<[?, 16, 1]>>) outs(%{{.*}}, %{{.*}} : memref<?x1x1xf16, strided<[?, 16, 1]>>, memref<?x1x1xi32, strided<[?, 16, 1]>>) unsigned_src = false tie_break_left = true reduce_dims = [1]
    // CHECK-NOT: hivm.hir.vreduce <max_with_index>{{.*}}temp_buffer(%{{.*}} : memref<16384xf32>)
    hivm.hir.vreduce <max_with_index>
      ins(%src : memref<?x32xf16, strided<[?, 16]>>)
      outs(%dst, %dst1 : memref<?x1xf16, strided<[?, 16]>>, memref<?x1xi32, strided<[?, 16]>>)
      temp_buffer(%alloc_8 : memref<16384xf32>)
      unsigned_src = false
      tie_break_left = true
      reduce_dims = [1]

    return
  }
}