// RUN: bishengir-opt %s -convert-hivmave-to-ave-intrin | FileCheck %s

// HiFloat8 memory transfers preserve bits through byte register instructions.
// CHECK-LABEL: func.func @copy
// CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v256s8"
// CHECK: llvm.bitcast %{{.*}} : vector<256xi8> to vector<256xf8HIF>
// CHECK: llvm.bitcast %{{.*}} : vector<256xf8HIF> to vector<256xi8>
// CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v256s8"
func.func @copy(%src: memref<256xf8HIF, #hivm.address_space<ub>>, %dst: memref<256xf8HIF, #hivm.address_space<ub>>, %mask: vector<256xi1>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %x = ave.hir.vload <NORM> %src[%c0] : memref<256xf8HIF, #hivm.address_space<ub>> into vector<256xf8HIF>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %mask, %x : memref<256xf8HIF, #hivm.address_space<ub>>, vector<256xi1>, vector<256xf8HIF>
  return
}
