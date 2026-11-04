// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-mark-disable-load -convert-hivm-to-std | FileCheck %s --check-prefix=NOCACHE
// Padding uses FP32, matching PadValueType<hifloat8_t> and avoiding APFloat.
// NOCACHE-LABEL: func.func @memory
// NOCACHE: call @load_gm_to_ubuf_1d_hifloat8_t_nocache
// CHECK-LABEL: func.func @memory
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: call @load_gm_to_ubuf_1d_hifloat8_t
// CHECK: call @store_ubuf_to_gm_1d_hifloat8_t
func.func @memory(%gm: memref<128xf8HIF, #hivm.address_space<gm>>, %ub: memref<128xf8HIF, #hivm.address_space<ub>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.load ins(%gm : memref<128xf8HIF, #hivm.address_space<gm>>) outs(%ub : memref<128xf8HIF, #hivm.address_space<ub>>)
  hivm.hir.store ins(%ub : memref<128xf8HIF, #hivm.address_space<ub>>) outs(%gm : memref<128xf8HIF, #hivm.address_space<gm>>)
  return
}
// CHECK-LABEL: func.func @scalar_broadcast
// CHECK: arith.bitcast {{.*}} : f8HIF to i8
// CHECK: call @{{.*}}int8_t
func.func @scalar_broadcast(%x: f8HIF, %ub: memref<128xf8HIF, #hivm.address_space<ub>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.vbrc ins(%x : f8HIF) outs(%ub : memref<128xf8HIF, #hivm.address_space<ub>>)
  return
}


// NOCACHE-LABEL: func.func @readonly_load
// NOCACHE: arith.constant 0.000000e+00 : f32
// NOCACHE: call @load_gm_to_ubuf_1d_hifloat8_t(
func.func @readonly_load(%gm: memref<128xf8HIF, #hivm.address_space<gm>>, %ub: memref<128xf8HIF, #hivm.address_space<ub>>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.load ins(%gm : memref<128xf8HIF, #hivm.address_space<gm>>) outs(%ub : memref<128xf8HIF, #hivm.address_space<ub>>)
  return
}
