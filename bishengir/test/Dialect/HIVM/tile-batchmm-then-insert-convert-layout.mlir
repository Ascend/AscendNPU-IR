// RUN: bishengir-opt %s --hivm-tile-batchmm-into-loop --hivm-normalize-matmul \
// RUN:   --hivm-insert-convert-layout \
// RUN:   --hivm-propagate-convert-layout="allow-agnostic-ops=false" \
// RUN:   --canonicalize --cse \
// RUN:   -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @batchmm_gets_fractal_after_tile
// CHECK: scf.for
// CHECK:   hivm.hir.convert_layout
// CHECK:   hivm.hir.convert_layout
// CHECK:   hivm.hir.mmadL1 {{.*}}batch_matmul
// CHECK-SAME: tensor<{{[0-9?]+}}x{{[0-9?]+}}x{{[0-9]+}}x{{[0-9]+}}xf16>
// CHECK-SAME: tensor<{{[0-9?]+}}x{{[0-9?]+}}x{{[0-9]+}}x{{[0-9]+}}xf16>
// CHECK:   hivm.hir.convert_layout
// CHECK:   hivm.hir.fixpipe
// CHECK: }
func.func @batchmm_gets_fractal_after_tile(%dst : memref<2x256x256xf16>) {
  %ma = tensor.empty() : tensor<2x256x128xf16>
  %mb = tensor.empty() : tensor<2x128x256xf16>
  %mc = tensor.empty() : tensor<2x256x256xf32>
  %true = arith.constant true
  %M = arith.constant 256 : index
  %K = arith.constant 128 : index
  %N = arith.constant 256 : index
  %result = hivm.hir.batchMmadL1 {already_set_real_mkn}
    ins(%ma, %mb, %true, %M, %K, %N
        : tensor<2x256x128xf16>, tensor<2x128x256xf16>, i1, index, index, index)
    outs(%mc : tensor<2x256x256xf32>) -> tensor<2x256x256xf32>
  hivm.hir.fixpipe {enable_nz2nd}
    ins(%result : tensor<2x256x256xf32>)
    outs(%dst : memref<2x256x256xf16>)
  return
}