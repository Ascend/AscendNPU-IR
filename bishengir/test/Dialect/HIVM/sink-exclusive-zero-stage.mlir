// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="bypass-shape-registry=true" | FileCheck %s

// Retain the producer's load/cast when all scopes execute in one iteration.
// The mixed-stage case in sink-exclusive-preload-work.mlir still sinks it.
// CHECK-LABEL: func.func @keep_zero_stage_load_before_cube
// CHECK: %[[PROD:.*]] = scope.scope
// CHECK: hivm.hir.load
// CHECK: %[[CAST:.*]] = hivm.hir.vcast
// CHECK: scope.return %[[CAST]]
// CHECK: } {{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK: } {{.*}}CUBE{{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.vcast
// CHECK: hivm.hir.vexp ins(%[[PROD]]
func.func @keep_zero_stage_load_before_cube(
    %src: memref<32x128xbf16, #hivm.address_space<gm>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xbf16, #hivm.address_space<ub>>
    %prod = scope.scope : () -> tensor<32x128xf32> {
      hivm.hir.load ins(%src : memref<32x128xbf16, #hivm.address_space<gm>>)
                    outs(%buf : memref<32x128xbf16, #hivm.address_space<ub>>)
      %q = bufferization.to_tensor %buf restrict writable : memref<32x128xbf16, #hivm.address_space<ub>>
      %empty = tensor.empty() : tensor<32x128xf32>
      %dst = hivm.hir.vcast {enable_overflow = true, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%q : tensor<32x128xbf16>) outs(%empty : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      scope.return %dst : tensor<32x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %out = tensor.empty() : tensor<32x128xf32>
      %e = hivm.hir.vexp ins(%prod : tensor<32x128xf32>) outs(%out : tensor<32x128xf32>) -> tensor<32x128xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
