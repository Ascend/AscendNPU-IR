// RUN: bishengir-opt %s -hivm-merge-same-preload-scopes | FileCheck %s
// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work | FileCheck %s

// After SplitMixKernel the module has `_mix_aic` / `_mix_aiv` copies.
// Dest-sink / copy-chain sink on those copies is not isomorphic, so both
// MergeSame (flatten only) and SinkExclusivePreloadWork must leave the
// exclusive dest cluster in the producer.
// CHECK-LABEL: func.func @kernel_mix_aic(
// CHECK:         scope.scope
// CHECK:           bufferization.to_tensor
// CHECK:           hivm.hir.vcast
// CHECK:           scope.return
// CHECK:         } {{{.*}}preload_num = 1
// CHECK:         scope.scope
// CHECK:           hivm.hir.vexp
func.func @kernel_mix_aic(%arg0: memref<32x128xbf16, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %prod = scope.scope : () -> tensor<32x128xf32> {
      %src = bufferization.to_tensor %arg0 restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%src : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// Copy-chain sink is also disabled after SplitMixKernel.
// CHECK-LABEL: func.func @kernel_mix_aiv(
// CHECK:         scope.scope
// CHECK:           hivm.hir.load
// CHECK:           hivm.hir.copy
// CHECK:           scope.return
// CHECK:         } {{{.*}}VECTOR
// CHECK:         scope.scope
// CHECK:           hivm.hir.mmadL1
// CHECK:         } {{{.*}}CUBE
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.load
// CHECK:           scope.return
// CHECK:         } {{{.*}}VECTOR
func.func @kernel_mix_aiv(
    %src: memref<16x16xbf16, #hivm.address_space<gm>>,
    %other: tensor<16x16xbf16>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  scf.for %i = %c0 to %c4 step %c1 {
    %ub = memref.alloc() : memref<16x16xbf16, #hivm.address_space<ub>>
    %cbuf = memref.alloc() : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      hivm.hir.load ins(%src : memref<16x16xbf16, #hivm.address_space<gm>>)
                    outs(%ub : memref<16x16xbf16, #hivm.address_space<ub>>)
      %t = bufferization.to_tensor %ub restrict writable : memref<16x16xbf16, #hivm.address_space<ub>>
      hivm.hir.copy ins(%t : tensor<16x16xbf16>)
                    outs(%cbuf : memref<16x16xbf16, #hivm.address_space<cbuf>>)
                    {"hivm.inserted-copy"}
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%other, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 3 : i32, no_inline}
    %w = bufferization.to_tensor %cbuf restrict writable : memref<16x16xbf16, #hivm.address_space<cbuf>>
    scope.scope : () -> () {
      %acc = tensor.empty() : tensor<16x16xf32>
      %m = hivm.hir.mmadL1 ins(%w, %other, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
