// REQUIRES: bishengir-hifloat8
// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: func.func @encode_rta
// CHECK: %[[RND:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322hif8.x"({{.*}}, %[[RND]], {{.*}})
func.func @encode_rta(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vtruncf %x, <round>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// HIVM HYBRID is 7, but the CCE intrinsic must receive 6.
// CHECK-LABEL: func.func @encode_hybrid
// CHECK: %[[RND:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322hif8.x"({{.*}}, %[[RND]], {{.*}})
func.func @encode_hybrid(%x: vector<64xf32>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vtruncf %x, <hybrid>, false, <part_even>, %mask : vector<64xf32>, vector<256xf8HIF>, vector<256xi1>
  return %y : vector<256xf8HIF>
}

// CHECK-LABEL: func.func @decode
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.hif82f32.x"
func.func @decode(%x: vector<256xf8HIF>, %mask: vector<256xi1>) -> vector<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vextf %x, <part_even>, %mask : vector<256xf8HIF>, vector<64xf32>, vector<256xi1>
  return %y : vector<64xf32>
}
