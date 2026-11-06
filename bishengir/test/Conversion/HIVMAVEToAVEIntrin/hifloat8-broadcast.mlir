// REQUIRES: bishengir-hifloat8
// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: func.func @broadcast
// CHECK: llvm.bitcast %{{.*}} : f8HIF to i8
// CHECK: "hivm_regbaseintrins.intr.hivm.vbr"
// CHECK: llvm.bitcast %{{.*}} : vector<256xi8> to vector<256xf8HIF>
func.func @broadcast(%x: f8HIF) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.scalar_broadcast %x : f8HIF -> vector<256xf8HIF>
  return %y : vector<256xf8HIF>
}

// CHECK-LABEL: func.func @broadcast_low
// CHECK: llvm.bitcast %{{.*}} : vector<256xf8HIF> to vector<256xi8>
// CHECK: "hivm_regbaseintrins.intr.hivm.vdup.z"
// CHECK: llvm.bitcast %{{.*}} : vector<256xi8> to vector<256xf8HIF>
func.func @broadcast_low(%x: vector<256xf8HIF>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vector_broadcast %x, %mask, true : vector<256xf8HIF>, vector<256xi1> -> vector<256xf8HIF>
  return %y : vector<256xf8HIF>
}

// CHECK-LABEL: func.func @broadcast_high
// CHECK: llvm.bitcast %{{.*}} : vector<256xf8HIF> to vector<256xi8>
// CHECK: "hivm_regbaseintrins.intr.hivm.vdupm.z"
// CHECK: llvm.bitcast %{{.*}} : vector<256xi8> to vector<256xf8HIF>
func.func @broadcast_high(%x: vector<256xf8HIF>, %mask: vector<256xi1>) -> vector<256xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %y = ave.hir.vector_broadcast %x, %mask, false : vector<256xf8HIF>, vector<256xi1> -> vector<256xf8HIF>
  return %y : vector<256xf8HIF>
}
