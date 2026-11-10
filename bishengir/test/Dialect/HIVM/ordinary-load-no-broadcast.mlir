// RUN: bishengir-opt %s | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend960 -convert-hivm-to-std | FileCheck %s --check-prefix=ABI
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -convert-hivm-to-std | FileCheck %s --check-prefix=ABI
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910_9391 -convert-hivm-to-std | FileCheck %s
// ABI-LABEL: func.func @ordinary
// ABI: call @load_gm_to_ubuf_1d_float(
// ABI: call @load_gm_to_ubuf_2d_float(
// ABI: call @load_gm_to_ubuf_3d_float(
// ABI: call @load_gm_to_ubuf_1d_float(
// ABI-NOT: _broadcast
// CHECK-LABEL: func.func @ordinary
// CHECK-COUNT-4: hivm.hir.load
// CHECK-NOT: broadcast_axes
func.func @ordinary(%s1: memref<32xf32, #hivm.address_space<gm>>, %d1: memref<32xf32, #hivm.address_space<ub>>, %s2: memref<2x32xf32, #hivm.address_space<gm>>, %d2: memref<2x32xf32, #hivm.address_space<ub>>, %s3: memref<2x2x32xf32, #hivm.address_space<gm>>, %d3: memref<2x2x32xf32, #hivm.address_space<ub>>, %pad: memref<40xf32, #hivm.address_space<ub>>) {
  %zero = arith.constant 0.0 : f32
  hivm.hir.load ins(%s1 : memref<32xf32, #hivm.address_space<gm>>) outs(%d1 : memref<32xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%s2 : memref<2x32xf32, #hivm.address_space<gm>>) outs(%d2 : memref<2x32xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%s3 : memref<2x2x32xf32, #hivm.address_space<gm>>) outs(%d3 : memref<2x2x32xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%s1 : memref<32xf32, #hivm.address_space<gm>>) outs(%pad : memref<40xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %zero : f32
  return
}
