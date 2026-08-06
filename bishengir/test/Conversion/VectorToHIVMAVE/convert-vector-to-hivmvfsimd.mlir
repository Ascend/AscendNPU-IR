// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @transfer_read_was_bool_to_int8_attr
// CHECK: ave.hir.vload <NORM> %arg0[%c0] {was_bool_to_int8 = true} : memref<256xi8, #hivm.address_space<ub>> into vector<256xi8>
func.func @transfer_read_was_bool_to_int8_attr(%arg0: memref<256xi8, #hivm.address_space<ub>>) -> vector<256xi8> attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %0 = vector.transfer_read %arg0[%c0], %c0_i8 {in_bounds = [true], was_bool_to_int8 = true} : memref<256xi8, #hivm.address_space<ub>>, vector<256xi8>
  return %0 : vector<256xi8>
}
