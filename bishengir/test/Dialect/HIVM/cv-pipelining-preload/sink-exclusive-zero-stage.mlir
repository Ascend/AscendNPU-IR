// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="bypass-shape-registry=true" | FileCheck %s

// Same preload stage, so the load sinks into the consumer after the cube.
// CHECK-LABEL: func.func @sink_zero_stage_load_before_cube
// CHECK: scope.scope
// CHECK-NOT: hivm.hir.load
// CHECK: scope.return
// CHECK: } {{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK: } {{.*}}CUBE{{.*}}hivm.preload_num = 0
// CHECK: scope.scope
// CHECK: hivm.hir.load
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.vexp
// CHECK: } {{.*}}hivm.preload_num = 0
func.func @sink_zero_stage_load_before_cube(
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

// The consumer prepares an L1 copy, so the mask chain lands at the front of
// that scope, ahead of the row broadcast that was already there.
// CHECK-LABEL: func.func @sink_zero_stage_mask_before_result
// CHECK:         scope.scope
// CHECK-NOT:       hivm.hir.vbrc
// CHECK-NOT:       hivm.hir.vsel
// CHECK:           scope.return
// CHECK:         } {{.*}}tcore_type<VECTOR>{{.*}}hivm.preload_num = 0
// CHECK:         scope.scope
// CHECK:         } {{.*}}tcore_type<CUBE>{{.*}}hivm.preload_num = 0
// CHECK:         %[[MASK:.*]] = hivm.hir.vbrc {{.*}}ins(%c2147483647_i32
// CHECK:         %[[CLEAN:.*]] = hivm.hir.vsel
// CHECK:         %[[ROW:.*]] = hivm.hir.vbrc {{.*}}ins(%c0_i32
// CHECK:         hivm.hir.vand ins(%[[ROW]], %[[MASK]]
// CHECK:         hivm.hir.vmax ins(%[[CLEAN]],
func.func @sink_zero_stage_mask_before_result(
    %p: tensor<16xf16>, %dst: memref<16xf16, #hivm.address_space<cbuf>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7fff = arith.constant 2147483647 : i32
  %cneg = arith.constant -2139095040 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<64xf32>
  %init = hivm.hir.vbrc ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<64xf32>) {
    %pair:2 = scope.scope : () -> (tensor<64xi32>, tensor<64xf32>) {
      %ei = tensor.empty() : tensor<64xi32>
      %mask = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          ins(%c7fff : i32) outs(%ei : tensor<64xi32>) -> tensor<64xi32>
      %bits = hivm.hir.bitcast %acc : tensor<64xf32> -> tensor<64xi32>
      %and = hivm.hir.vand ins(%bits, %mask : tensor<64xi32>, tensor<64xi32>)
          outs(%ei : tensor<64xi32>) -> tensor<64xi32>
      %added = hivm.hir.vadd ins(%and, %cneg : tensor<64xi32>, i32)
          outs(%ei : tensor<64xi32>) -> tensor<64xi32>
      %ef = tensor.empty() : tensor<64xf32>
      %cast = hivm.hir.vcast {enable_overflow = false, enable_saturate = false,
                              hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%added : tensor<64xi32>) outs(%ef : tensor<64xf32>) -> tensor<64xf32>
      %pred = tensor.empty() : tensor<64xi1>
      %cmp = hivm.hir.vcmp ins(%cast, %cst : tensor<64xf32>, f32)
          outs(%pred : tensor<64xi1>) compare_mode = <ne> -> tensor<64xi1>
      %clean = hivm.hir.vsel ins(%cmp, %inf, %acc : tensor<64xi1>, f32, tensor<64xf32>)
          outs(%ef : tensor<64xf32>) -> tensor<64xf32>
      scope.return %mask, %clean : tensor<64xi32>, tensor<64xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 0 : i32, no_inline}
    %next = scope.scope : () -> tensor<64xf32> {
      %row = tensor.empty() : tensor<64xi32>
      %rowb = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          ins(%c0_i32 : i32) outs(%row : tensor<64xi32>) -> tensor<64xi32>
      %masked = hivm.hir.vand ins(%rowb, %pair#0 : tensor<64xi32>, tensor<64xi32>)
          outs(%row : tensor<64xi32>) -> tensor<64xi32>
      %ef = tensor.empty() : tensor<64xf32>
      %asf = hivm.hir.vcast {enable_overflow = false, enable_saturate = false,
                             hivm.unsigned_mode = #hivm.unsigned_mode<si2si>}
          ins(%masked : tensor<64xi32>) outs(%ef : tensor<64xf32>) -> tensor<64xf32>
      %out = hivm.hir.vmax ins(%pair#1, %asf : tensor<64xf32>, tensor<64xf32>)
          outs(%ef : tensor<64xf32>) -> tensor<64xf32>
      hivm.hir.copy ins(%p : tensor<16xf16>) outs(%dst : memref<16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %out : tensor<64xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %next : tensor<64xf32>
  }
  return
}
