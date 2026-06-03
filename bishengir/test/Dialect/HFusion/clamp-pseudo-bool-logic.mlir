// RUN: bishengir-opt %s -hfusion-legalize-bool=enable-clamp=true | FileCheck %s

// Both operands of the addi are canonical pseudo-bools (was_bool_to_int8), so
// the add + clamp-to-nonzero collapses into vor followed by a "& 1" canonical-
// ization, with no arith.cmpi / arith.extsi / i32 temporaries.
// CHECK-LABEL: func.func @add_two_bools
// CHECK:         %[[OR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>, is_clamped = true} ins
// CHECK:         %[[ONE:.*]] = linalg.fill ins(%c1_i8
// CHECK:         hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>, was_bool_to_int8 = true} ins(%[[OR]], %[[ONE]]
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     arith.extsi
func.func @add_two_bools(%arg0: memref<8x1xi8>, %arg1: memref<8x1xi8>) -> tensor<8x1xi8> {
  %0 = bufferization.to_tensor %arg0 restrict writable {was_bool_to_int8 = true} : memref<8x1xi8>
  %1 = bufferization.to_tensor %arg1 restrict writable {was_bool_to_int8 = true} : memref<8x1xi8>
  %2 = arith.addi %0, %1 : tensor<8x1xi8>
  return %2 : tensor<8x1xi8>
}

// Subtraction of two bools becomes vxor, also canonicalized with "& 1".
// CHECK-LABEL: func.func @sub_two_bools
// CHECK:         %[[XOR:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>, is_clamped = true} ins
// CHECK:         hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>, was_bool_to_int8 = true} ins(%[[XOR]],
// CHECK-NOT:     arith.cmpi
func.func @sub_two_bools(%arg0: memref<8x1xi8>, %arg1: memref<8x1xi8>) -> tensor<8x1xi8> {
  %0 = bufferization.to_tensor %arg0 restrict writable {was_bool_to_int8 = true} : memref<8x1xi8>
  %1 = bufferization.to_tensor %arg1 restrict writable {was_bool_to_int8 = true} : memref<8x1xi8>
  %2 = arith.subi %0, %1 : tensor<8x1xi8>
  return %2 : tensor<8x1xi8>
}

