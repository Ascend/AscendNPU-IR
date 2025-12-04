// RUN: bishengir-opt %s -hivm-lower-to-loops -split-input-file | FileCheck %s

// -----
func.func @mlir_fused_bitwise_not_clone_mul_sub_14(%arg0: memref<1x16384xi64, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<16384x1xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<16384x1xi64, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.get_tiling_struct_size_function = #hacc.get_tiling_struct_size_function<@mlir_fused_bitwise_not_clone_mul_sub_14_get_tiling_struct_size_function>, hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c80256_i64 = arith.constant 80256 : i64
  %c79040_i64 = arith.constant 79040 : i64
  %c155648_i64 = arith.constant 155648 : i64
  %c77824_i64 = arith.constant 77824 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  hivm.hir.set_mask_norm
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x16384xi64, #hivm.address_space<gm>> into memref<16384xi64, #hivm.address_space<gm>>
  %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<16384x1xi8, #hivm.address_space<gm>> into memref<16384xi8, #hivm.address_space<gm>>
  %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<16384x1xi64, #hivm.address_space<gm>> into memref<16384xi64, #hivm.address_space<gm>>
  // CHECK: scf.for
  scf.for %arg3 = %c0 to %c2 step %c1 {
    %0 = affine.apply affine_map<()[s0] -> (s0 * 9728)>()[%arg3]
    %1 = affine.min affine_map<()[s0] -> (s0 * -9728 + 16384, 9728)>()[%arg3]
    %subview = memref.subview %collapse_shape[%0] [%1] [1] : memref<16384xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<77824xi8, #hivm.address_space<ub>>
    %view = memref.view %2[%c0][%1] : memref<77824xi8, #hivm.address_space<ub>> to memref<?xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%subview : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%view : memref<?xi64, #hivm.address_space<ub>>)
    %subview_2 = memref.subview %collapse_shape_0[%0] [%1] [1] : memref<16384xi8, #hivm.address_space<gm>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %3 = hivm.hir.pointer_cast(%c77824_i64) : memref<9728xi8, #hivm.address_space<ub>>
    %view_3 = memref.view %3[%c0][%1] : memref<9728xi8, #hivm.address_space<ub>> to memref<?xi8, #hivm.address_space<ub>>
    hivm.hir.load ins(%subview_2 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%view_3 : memref<?xi8, #hivm.address_space<ub>>)
    %subview_4 = memref.subview %collapse_shape_1[%0] [%1] [1] : memref<16384xi64, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %4 = hivm.hir.pointer_cast(%c155648_i64) : memref<19456xi8, #hivm.address_space<ub>>
    %view_5 = memref.view %4[%c0][%1] : memref<19456xi8, #hivm.address_space<ub>> to memref<?xf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%view_3 : memref<?xi8, #hivm.address_space<ub>>) outs(%view_5 : memref<?xf16, #hivm.address_space<ub>>)
    %5 = hivm.hir.pointer_cast(%c77824_i64) : memref<1216xi8, #hivm.address_space<ub>>
    %view_6 = memref.view %5[%c0][%1] : memref<1216xi8, #hivm.address_space<ub>> to memref<?xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%view_5, %cst : memref<?xf16, #hivm.address_space<ub>>, f16) outs(%view_6 : memref<?xi1, #hivm.address_space<ub>>) compare_mode = <ne>
    %6 = hivm.hir.pointer_cast(%c79040_i64) : memref<1216xi8, #hivm.address_space<ub>>
    %view_7 = memref.view %6[%c0][%1] : memref<1216xi8, #hivm.address_space<ub>> to memref<?xi1, #hivm.address_space<ub>>
    hivm.hir.vnot ins(%view_6 : memref<?xi1, #hivm.address_space<ub>>) outs(%view_7 : memref<?xi1, #hivm.address_space<ub>>)
    %7 = hivm.hir.pointer_cast(%c80256_i64) : memref<19456xi8, #hivm.address_space<ub>>
    %view_8 = memref.view %7[%c0][%1] : memref<19456xi8, #hivm.address_space<ub>> to memref<?xf16, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c77824_i64) : memref<48xf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%view_7 : memref<?xi1, #hivm.address_space<ub>>) outs(%view_8 : memref<?xf16, #hivm.address_space<ub>>) temp_buffer(%8 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
    %9 = hivm.hir.pointer_cast(%c155648_i64) : memref<38912xi8, #hivm.address_space<ub>>
    %view_9 = memref.view %9[%c0][%1] : memref<38912xi8, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%view_8 : memref<?xf16, #hivm.address_space<ub>>) outs(%view_9 : memref<?xf32, #hivm.address_space<ub>>)
    %10 = hivm.hir.pointer_cast(%c77824_i64) : memref<77824xi8, #hivm.address_space<ub>>
    %view_10 = memref.view %10[%c0][%1] : memref<77824xi8, #hivm.address_space<ub>> to memref<?xi64, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%view_9 : memref<?xf32, #hivm.address_space<ub>>) outs(%view_10 : memref<?xi64, #hivm.address_space<ub>>) round_mode = <trunc>
    %11 = hivm.hir.pointer_cast(%c77824_i64) : memref<77824xi8, #hivm.address_space<ub>>
    %view_11 = memref.view %11[%c0][%1] : memref<77824xi8, #hivm.address_space<ub>> to memref<?xi64, #hivm.address_space<ub>>
    // CHECK: scf.for %[[arg4:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK: %[[src0:.*]] = memref.load {{.*}}{{\[}}%[[arg4]]] : memref<?xi64, #hivm.address_space<ub>>
    // CHECK: %[[src1:.*]] = memref.load {{.*}}{{\[}}%[[arg4]]] : memref<?xi64, #hivm.address_space<ub>>
    // CHECK: %[[mul:.*]] = arith.muli %[[src0]], %[[src1]] : i64
    // CHECK: memref.store %[[mul]],  {{.*}}{{\[}}%[[arg4]]] : memref<?xi64, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%view, %view_10 : memref<?xi64, #hivm.address_space<ub>>, memref<?xi64, #hivm.address_space<ub>>) outs(%view_11 : memref<?xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%view_11 : memref<?xi64, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <none>
  } {__tiled_for___2}
  return
}

//===----------------------------------------------------------------------===//
// Test VMulHiOp Decompose MemRef(I32)
//===----------------------------------------------------------------------===//
// -----
func.func @test_vmulhi_op_memref_i32() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 6 : index
  // CHECK: %[[VAL_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: %[[VAL_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  // CHECK: scf.for %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] {
  // CHECK:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   %[[LOW:.*]], %[[HIGH:.*]] = arith.mului_extended {{.*}}, {{.*}} : i32
  // CHECK:   memref.store %[[LOW]], %[[VAL_6]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK:   memref.store %[[HIGH]], %[[VAL_7]]{{\[}}%[[VAL_8]]] : memref<6xi32>
  // CHECK: }

  %lhs = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %rhs = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6xi32>
  hivm.hir.vmulext ins(%lhs, %rhs : memref<6xi32>, memref<6xi32>) outs(%alloc, %alloc_0 : memref<6xi32>, memref<6xi32>)
  return
}

//===----------------------------------------------------------------------===//
// Test VMul Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vmul_b64() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_DST:.*]] = memref.alloc() : memref<64x8xi64>
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
   // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
   // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[dst:.*]] = arith.muli %[[src0]], %[[src1]] : i64
   // CHECK: memref.store %[[dst]], %[[ALLOC_DST]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vmul ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VSub Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vsub_b64() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_DST:.*]] = memref.alloc() : memref<64x8xi64>
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
   // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
   // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[dst:.*]] = arith.subi %[[src0]], %[[src1]] : i64
   // CHECK: memref.store %[[dst]], %[[ALLOC_DST]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vsub ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VAdd Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vadd_b64() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_DST:.*]] = memref.alloc() : memref<64x8xi64>
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
   // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
   // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[dst:.*]] = arith.addi %[[src0]], %[[src1]] : i64
   // CHECK: memref.store %[[dst]], %[[ALLOC_DST]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vadd ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VAbs Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vabs_b64() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[ALLOC_DST:.*]] = memref.alloc() : memref<64x8xi64>
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
   // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
   // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   // CHECK: %[[dst:.*]] = math.absi %[[src0]] : i64
   // CHECK: memref.store %[[dst]], %[[ALLOC_DST]][%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vabs ins(%allocIn0 : memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VCmp Decompose MemRef
//===----------------------------------------------------------------------===//

// -----
func.func @test_vcmp_b32_lt() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<64x8xi32>
  %alloc_0 = memref.alloc() : memref<64x8xi32>
  %alloc_1 = memref.alloc() : memref<64x8xi1>
  %alloc_2 = memref.alloc() : memref<64x8xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
  // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[dst:.*]] = arith.cmpi slt, %[[src0]], %[[src1]] : i32
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]], %[[arg1]]] : memref<64x8xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<64x8xi32>, memref<64x8xi32>) outs(%alloc_2 : memref<64x8xi8>) compare_mode = <lt>
  %alloc_3 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<64x8xi8>) outs(%alloc_3 : memref<64x8xf16>)
  %alloc_4 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<64x8xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<64x8xf16>, memref<64x8xf16>) outs(%alloc_1 : memref<64x8xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b32_gt() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<64x8xi32>
  %alloc_0 = memref.alloc() : memref<64x8xi32>
  %alloc_1 = memref.alloc() : memref<64x8xi1>
  %alloc_2 = memref.alloc() : memref<64x8xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
  // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[dst:.*]] = arith.cmpi sgt, %[[src0]], %[[src1]] : i32
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]], %[[arg1]]] : memref<64x8xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<64x8xi32>, memref<64x8xi32>) outs(%alloc_2 : memref<64x8xi8>) compare_mode = <gt>
  %alloc_3 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<64x8xi8>) outs(%alloc_3 : memref<64x8xf16>)
  %alloc_4 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<64x8xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<64x8xf16>, memref<64x8xf16>) outs(%alloc_1 : memref<64x8xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b32_le() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<64x8xi32>
  %alloc_0 = memref.alloc() : memref<64x8xi32>
  %alloc_1 = memref.alloc() : memref<64x8xi1>
  %alloc_2 = memref.alloc() : memref<64x8xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
  // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[dst:.*]] = arith.cmpi sle, %[[src0]], %[[src1]] : i32
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]], %[[arg1]]] : memref<64x8xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<64x8xi32>, memref<64x8xi32>) outs(%alloc_2 : memref<64x8xi8>) compare_mode = <le>
  %alloc_3 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<64x8xi8>) outs(%alloc_3 : memref<64x8xf16>)
  %alloc_4 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<64x8xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<64x8xf16>, memref<64x8xf16>) outs(%alloc_1 : memref<64x8xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b32_ge() {
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_4:.*]] = arith.constant 64 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<64x8xi32>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<64x8xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<64x8xi32>
  %alloc_0 = memref.alloc() : memref<64x8xi32>
  %alloc_1 = memref.alloc() : memref<64x8xi1>
  %alloc_2 = memref.alloc() : memref<64x8xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_2]]
  // CHECK: scf.for %[[arg1:.*]] = %[[VAL_3]] to %[[VAL_1]] step %[[VAL_2]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]], %[[arg1]]] : memref<64x8xi32>
  // CHECK: %[[dst:.*]] = arith.cmpi sge, %[[src0]], %[[src1]] : i32
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]], %[[arg1]]] : memref<64x8xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<64x8xi32>, memref<64x8xi32>) outs(%alloc_2 : memref<64x8xi8>) compare_mode = <ge>
  %alloc_3 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<64x8xi8>) outs(%alloc_3 : memref<64x8xf16>)
  %alloc_4 = memref.alloc() : memref<64x8xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<64x8xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<64x8xf16>, memref<64x8xf16>) outs(%alloc_1 : memref<64x8xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b32_eq() {
   %allocIn0 = memref.alloc() : memref<64x8xi32>
   %allocIn1 = memref.alloc() : memref<64x8xi32>
   %allocOut = memref.alloc() : memref<64x8xi1>
   // CHECK: hivm.hir.vcmp ins(%[[ARG0:.*]], %[[ARG1:.*]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC:.*]] : memref<64x8xi1>)
   hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                 outs(%allocOut : memref<64x8xi1>)
                 compare_mode = #hivm.compare_mode<eq>
   return
}

// -----
func.func @test_vcmp_b32_ne() {
   %allocIn0 = memref.alloc() : memref<64x8xi32>
   %allocIn1 = memref.alloc() : memref<64x8xi32>
   %allocOut = memref.alloc() : memref<64x8xi1>

   // CHECK: hivm.hir.vcmp ins(%[[ARG0:.*]], %[[ARG1:.*]] : memref<64x8xi32>, memref<64x8xi32>) outs(%[[ALLOC:.*]] : memref<64x8xi1>) compare_mode = <ne>
   hivm.hir.vcmp ins(%allocIn0, %allocIn1 : memref<64x8xi32>, memref<64x8xi32>)
                 outs(%allocOut : memref<64x8xi1>)
                 compare_mode = #hivm.compare_mode<ne>
   return
}

// -----
func.func @test_vcmp_b16_lt() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi slt, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <lt>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b16_gt() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi sgt, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <gt>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b16_le() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi sle, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <le>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b16_ge() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi sge, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <ge>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b16_eq() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi eq, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>)
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b16_ne() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi16>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi16>
  %alloc_0 = memref.alloc() : memref<1024xi16>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi16>
  // CHECK: %[[dst:.*]] = arith.cmpi ne, %[[src0]], %[[src1]] : i16
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi16>, memref<1024xi16>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <ne>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_lt() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi slt, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <lt>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_gt() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi sgt, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <gt>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_le() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi sle, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <le>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_ge() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi sge, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <ge>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_eq() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi eq, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>)
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_ne() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 1024 : index
  // CHECK: %[[ALLOC_SRC0:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<1024xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1024xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<1024xi64>
  %alloc_0 = memref.alloc() : memref<1024xi64>
  %alloc_1 = memref.alloc() : memref<1024xi1>
  %alloc_2 = memref.alloc() : memref<1024xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_0]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC0]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_SRC1]][%[[arg0]]] : memref<1024xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi ne, %[[src0]], %[[src1]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1024xi8>
  hivm.hir.vcmp ins(%alloc, %alloc_0 : memref<1024xi64>, memref<1024xi64>) outs(%alloc_2 : memref<1024xi8>) compare_mode = <ne>
  %alloc_3 = memref.alloc() : memref<1024xf16>
  hivm.hir.vcast ins(%alloc_2 : memref<1024xi8>) outs(%alloc_3 : memref<1024xf16>)
  %alloc_4 = memref.alloc() : memref<1024xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_4 : memref<1024xf16>)
  hivm.hir.vcmp ins(%alloc_3, %alloc_4 : memref<1024xf16>, memref<1024xf16>) outs(%alloc_1 : memref<1024xi1>) compare_mode = <ne>
  return
}

// -----
func.func @test_vcmp_b64_eq_vs() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 2 : i64
  // CHECK: %[[ALLOC_SRC:.*]] = memref.alloc() : memref<1xi64>
  // CHECK: %[[ALLOC_DST1:.*]] = memref.alloc() : memref<1xi8>
  %cst = arith.constant 0.000000e+00 : f16
  %c2_i64 = arith.constant 2 : i64
  %alloc = memref.alloc() : memref<1xi64>
  %alloc_0 = memref.alloc() : memref<1xi1>
  %alloc_1 = memref.alloc() : memref<1xi8>
  // CHECK: scf.for %[[arg0:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_1]]
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC_SRC]][%[[arg0]]] : memref<1xi64>
  // CHECK: %[[dst:.*]] = arith.cmpi eq, %[[src0]], %[[VAL_2]] : i64
  // CHECK: %[[dst1:.*]] = arith.extui %[[dst]] : i1 to i8
  // CHECK: memref.store %[[dst1]], %[[ALLOC_DST1]][%[[arg0]]] : memref<1xi8>
  hivm.hir.vcmp ins(%alloc, %c2_i64 : memref<1xi64>, i64) outs(%alloc_1 : memref<1xi8>)
  %alloc_2 = memref.alloc() : memref<1xf16>
  hivm.hir.vcast ins(%alloc_1 : memref<1xi8>) outs(%alloc_2 : memref<1xf16>)
  %alloc_3 = memref.alloc() : memref<1xf16>
  hivm.hir.vbrc ins(%cst : f16) outs(%alloc_3 : memref<1xf16>)
  hivm.hir.vcmp ins(%alloc_2, %alloc_3 : memref<1xf16>, memref<1xf16>) outs(%alloc_0 : memref<1xi1>) compare_mode = <ne>
  return
}

//===----------------------------------------------------------------------===//
// Test VShl Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vshl_b64() {
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   %cst = arith.constant 2 : i64
   // CHECK-DAG: %[[cst:.*]] = arith.constant 2 : i64
   // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: %[[src0:.*]] = memref.load {{.*}}[%[[arg0]], %[[arg1]]]
   // CHECK: %[[dst:.*]] = arith.shli %[[src0]], %[[cst:.*]] : i64
   // CHECK: memref.store %[[dst]],{{.*}}[%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vshl ins(%allocIn0, %cst : memref<64x8xi64>, i64) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VShr Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vshr_b64() {
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   %cst = arith.constant 2 : i64
   // CHECK-DAG: %[[cst:.*]] = arith.constant 2 : i64
   // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: %[[src0:.*]] = memref.load {{.*}}[%[[arg0]], %[[arg1]]]
   // CHECK: %[[dst:.*]] = arith.shrsi %[[src0]], %[[cst:.*]] : i64
   // CHECK: memref.store %[[dst]],{{.*}}[%[[arg0]], %[[arg1]]] : memref<64x8xi64>
   hivm.hir.vshr ins(%allocIn0, %cst : memref<64x8xi64>, i64) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test Vmin Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vmin_b64() {
   %allocIn0 = memref.alloc() : memref<64xi64>
   %allocIn1 = memref.alloc() : memref<64xi64>
   %allocOut = memref.alloc() : memref<64xi64>
   // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: %[[src0:.*]] = memref.load {{.*}}[%[[arg0]]]
   // CHECK: %[[src1:.*]] = memref.load {{.*}}[%[[arg0]]]
   // CHECK: %[[dst:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
   // CHECK: memref.store %[[dst]],{{.*}}[%[[arg0]]] : memref<64xi64>
   hivm.hir.vmin ins(%allocIn0, %allocIn1 : memref<64xi64>, memref<64xi64>) outs(%allocOut : memref<64xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test Vmax Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vmax_b64() {
   %allocIn0 = memref.alloc() : memref<64xi64>
   %allocIn1 = memref.alloc() : memref<64xi64>
   %allocOut = memref.alloc() : memref<64xi64>
   // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
   // CHECK: %[[src0:.*]] = memref.load {{.*}}[%[[arg0]]]
   // CHECK: %[[src1:.*]] = memref.load {{.*}}[%[[arg0]]]
   // CHECK: %[[dst:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
   // CHECK: memref.store %[[dst]],{{.*}}[%[[arg0]]] : memref<64xi64>
   hivm.hir.vmax ins(%allocIn0, %allocIn1 : memref<64xi64>, memref<64xi64>) outs(%allocOut : memref<64xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VReduce Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_reduce_sum_ar_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<24x1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]] : memref<24x32xi64>
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg1]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
func.func @test_reduce_min_ar_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<24x1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]] : memref<24x32xi64>
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg1]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  // CHECK: %[[dst:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <min> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
func.func @test_reduce_max_ar_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<24x1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]] : memref<24x32xi64>
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg1]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  // CHECK: %[[dst:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<24x1xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<24x1xi64>
  hivm.hir.vreduce <max> ins(%src : memref<24x32xi64>) outs(%dst : memref<24x1xi64>) reduce_dims = [1]
  return
}

// -----
func.func @test_reduce_min_with_index_int64() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi64>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i64
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi64>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi64>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi64>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_with_index_int32() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 2147483647 : i32
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi32>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i32
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i32
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_with_index_int16() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 32767 : i16
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi16>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i16
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i16
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi16>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi16>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi16>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int64() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi64>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i64
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi64>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi64>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi64>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int32() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi32>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i32
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i32
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int16() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -32768 : i16
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi16>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i16
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i16
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi16>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi16>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi16>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_with_index_int64_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi64>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i64
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi64>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi64>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi64>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_with_index_int32_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 2147483647 : i32
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi32>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i32
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i32
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi32>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi32>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_with_index_int16_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 32767 : i16
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi16>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi sgt, %[[src1]], %[[src0]] : i16
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.minsi %[[src0]], %[[src1]] : i16
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi16>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi16>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x2xi16>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int64_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi64>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i64
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi64>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi64>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi64>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi64>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi64>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int32_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi32>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i32
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i32
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi32>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi32>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi32>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi32>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_with_index_int16_with_index_input() {
  // CHECK-DAG: %[[CST_INDEX_INIT:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -32768 : i16
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x2xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x2xi16>
  // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2xi32>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: memref.store %[[CST_INDEX_INIT]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[src2:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst0:.*]] = arith.cmpi slt, %[[src1]], %[[src0]] : i16
  // CHECK: %[[dst1:.*]] = arith.index_cast %[[arg0]] : index to i32
  // CHECK: %[[dst2:.*]] = arith.select %[[dst0]], %[[dst1]], %[[src2]] : i32
  // CHECK: %[[dst3:.*]] = arith.maxsi %[[src0]], %[[src1]] : i16
  // CHECK: memref.store %[[dst3]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x2xi16>
  // CHECK: memref.store %[[dst2]], %[[ALLOC_1]][%[[CST0]], %[[arg1]]] : memref<1x2xi32>
  // CHECK: scf.for %[[arg2:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[idx:.*]] = memref.load %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  // CHECK: %[[idx0:.*]] = arith.index_cast %[[idx]] : i32 to index
  // CHECK: %[[idx1:.*]] = memref.load %[[ALLOC_IDX:.*]][%[[idx0]], %[[arg2]]] : memref<2x2xi32>
  // CHECK: memref.store %[[idx1]], %[[ALLOC_1]][%[[CST0]], %[[arg2]]] : memref<1x2xi32>
  %src = memref.alloc() : memref<2x2xi16>
  %idx = memref.alloc() : memref<2x2xi32>
  %dst1 = memref.alloc() : memref<1x2xi16>
  %dst2 = memref.alloc() : memref<1x2xi32>
  hivm.hir.vreduce <max_with_index_left> ins(%src : memref<2x2xi16>) indices(%idx : memref<2x2xi32>) outs(%dst1, %dst2 : memref<1x2xi16>, memref<1x2xi32>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_sum_ra_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x32xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst:.*]] = arith.addi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x32xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_ra_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x32xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x32xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <min> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_ra_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<24x32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1x32xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: scf.for %[[arg1:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]], %[[arg1]]]
  // CHECK: %[[dst:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]], %[[arg1]]] : memref<1x32xi64>
  %src = memref.alloc() : memref<24x32xi64>
  %dst = memref.alloc() : memref<1x32xi64>
  hivm.hir.vreduce <max> ins(%src : memref<24x32xi64>) outs(%dst : memref<1x32xi64>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_sum_r_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]]]
  // CHECK: %[[dst:.*]] = arith.addi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]]] : memref<1xi64>
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <sum> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_min_r_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]]]
  // CHECK: %[[dst:.*]] = arith.minsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]]] : memref<1xi64>
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <min> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

// -----
func.func @test_reduce_max_r_b64() {
  // CHECK-DAG: %[[CST_VALUE_INIT:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<1xi64>
  // CHECK: scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}}
  // CHECK: %[[src0:.*]] = memref.load %[[ALLOC]][%[[arg0]]]
  // CHECK: %[[isFirst:.*]] = arith.cmpi eq, %[[arg0]], %[[CST0]]
  // CHECK: scf.if %[[isFirst]] {
  // CHECK: memref.store %[[CST_VALUE_INIT]], %[[ALLOC_0]][%[[CST0]]]
  // CHECK: }
  // CHECK: %[[src1:.*]] = memref.load %[[ALLOC_0]][%[[CST0]]]
  // CHECK: %[[dst:.*]] = arith.maxsi %[[src0]], %[[src1]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[CST0]]] : memref<1xi64>
  %src = memref.alloc() : memref<32xi64>
  %dst = memref.alloc() : memref<1xi64>
  hivm.hir.vreduce <max> ins(%src : memref<32xi64>) outs(%dst : memref<1xi64>) reduce_dims = [0]
  return
}

//===----------------------------------------------------------------------===//
// Test VMod Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_vmod_b64() {
  // CHECK-DAG: %[[VAL_0:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[VAL_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[VAL_3:.*]] = arith.constant 64 : index
  // CHECK: %[[VAL_4:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[VAL_5:.*]] = memref.alloc() : memref<64x8xi64>
  // CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<64x8xi64>
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
  // CHECK: scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] {
  // CHECK: scf.for %[[VAL_8:.*]] = %[[VAL_2]] to %[[VAL_0]] step %[[VAL_1]] {
  // CHECK: %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<64x8xi64>
  // CHECK: %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<64x8xi64>
  // CHECK: %[[VAL_11:.*]] = arith.remsi %[[VAL_9]], %[[VAL_10]] : i64
  // CHECK: memref.store %[[VAL_11]], %[[VAL_6]]{{\[}}%[[VAL_7]], %[[VAL_8]]] : memref<64x8xi64>
    hivm.hir.vmod ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

//===----------------------------------------------------------------------===//
// Test VInterleave Decompose MemRef
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL:   func.func @test_decompose_vinterleave_b64(
// CHECK-SAME:              %[[SRC_0:.*]]: memref<16xi64>, %[[SRC_1:.*]]: memref<16xi64>, %[[DST:.*]]: memref<32xi64>) {
// CHECK-DAG:           %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK:             %[[VAL_0:.*]] = memref.load {{.*}}[%[[arg0]]] : memref<16xi64>
// CHECK:             %[[VAL_1:.*]] = memref.load {{.*}}[%[[arg0]]] : memref<16xi64>
// CHECK:             %[[arg1:.*]] = arith.muli %[[arg0]], %[[C2]] : index
// CHECK:             %[[arg2:.*]] = arith.addi %[[arg1]], %[[C1]] : index
// CHECK:             memref.store %[[VAL_0]], %[[DST]][%[[arg1]]] : memref<32xi64>
// CHECK:             memref.store %[[VAL_1]], %[[DST]][%[[arg2]]] : memref<32xi64> 
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vinterleave_b64(%src0: memref<16xi64>, 
                                          %src1: memref<16xi64>,
                                          %dst: memref<32xi64>) {
  hivm.hir.vinterleave ins(%src0, %src1: memref<16xi64>, memref<16xi64>)
                       outs(%dst: memref<32xi64>)
                       interleave_channel_nums = 2

  return
}

//===----------------------------------------------------------------------===//
// Test VDeinterleave Decompose MemRef
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL:   func.func @test_decompose_vdeinterleave_b64(
// CHECK-SAME:              %[[SRC:.*]]: memref<32xi64>, %[[EVEN:.*]]: memref<16xi64>, %[[ODD:.*]]: memref<16xi64>) {
// CHECK-DAG:           %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[arg0:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK:             %[[arg1:.*]] = arith.muli %[[arg0]], %[[C2]] : index
// CHECK:             %[[VAL_0:.*]] = memref.load {{.*}}[%[[arg1]]] : memref<32xi64>
// CHECK:             memref.store %[[VAL_0]], %[[EVEN]][%[[arg0]]] : memref<16xi64>
// CHECK:             %[[arg2:.*]] = arith.addi %[[arg:.*]], %[[C1]] : index
// CHECK:             %[[VAL_1:.*]] = memref.load {{.*}}[%[[arg2]]] : memref<32xi64>
// CHECK:             memref.store %[[VAL_1]], %[[ODD]][%[[arg0]]] : memref<16xi64> 
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vdeinterleave_b64(%src: memref<32xi64>, %dst_even: memref<16xi64>,
                                            %dst_odd: memref<16xi64>) {
  hivm.hir.vdeinterleave ins(%src: memref<32xi64>)
                         outs(%dst_even, %dst_odd: memref<16xi64>, memref<16xi64>)
                         index_mode = <ALL_CHANNELS>

  return
}

// -----
// CHECK-LABEL:   func.func @test_decompose_vdeinterleave_single_f16(
// CHECK-SAME:                                       %[[SRC:.*]]: memref<32xf16>,
// CHECK-SAME:                                       %[[EVEN:.*]]: memref<16xf16>,
// CHECK-SAME:                                       %[[ODD:.*]]: memref<16xf16>) {
// CHECK:           hivm.hir.vdeinterleave ins(%[[SRC]] : memref<32xf16>) outs(%[[EVEN]] : memref<16xf16>) channel_num = 2 index_mode = <CHANNEL_0>
// CHECK:           hivm.hir.vdeinterleave ins(%[[SRC]] : memref<32xf16>) outs(%[[ODD]] : memref<16xf16>) channel_num = 2 index_mode = <CHANNEL_1>
// CHECK:           return
// CHECK:         }
func.func @test_decompose_vdeinterleave_single_f16(%src: memref<32xf16>, %even_dst: memref<16xf16>,
                                                   %odd_dst: memref<16xf16>) {
  hivm.hir.vdeinterleave ins(%src: memref<32xf16>)
                             outs(%even_dst : memref<16xf16>)
                             channel_num = 2
                             index_mode = <CHANNEL_0>

  hivm.hir.vdeinterleave ins(%src: memref<32xf16>)
                             outs(%odd_dst: memref<16xf16>)
                             channel_num = 2
                             index_mode = <CHANNEL_1>

  return
}

// -----

// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpf
// CHECK: arith.index_cast
// CHECK: arith.select
// CHECK: arith.cmpf
// CHECK: arith.select
// CHECK: memref.store
// CHECK: memref.store
func.func @test_decompose_argmin_float(%src: memref<2x5x7xf16, strided<[96, 16, 1], offset: ?>>, %idx: memref<1x5x7xi32>) {
  %dst = memref.alloc() : memref<1x5x7xf16>
  hivm.hir.vreduce <min_with_index_left> ins(%src : memref<2x5x7xf16, strided<[96, 16, 1], offset: ?>>)
                                    outs(%dst, %idx : memref<1x5x7xf16>, memref<1x5x7xi32>) reduce_dims = [0]
  return
}

//===----------------------------------------------------------------------===//
// Test VCumsum Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_cumsum_1d_i16() {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<16xi16>
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]]] : memref<16xi16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]]] : memref<16xi16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<16xi16>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<16xi16>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<16xi16>
  %src = memref.alloc() : memref<16xi16>
  %dst = memref.alloc() : memref<16xi16>
  hivm.hir.vcumsum ins(%src : memref<16xi16>) outs(%dst : memref<16xi16>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumsum_2d_i16() {
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x16xi16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[CST0]]] : memref<2x16xi16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<2x16xi16>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]] : memref<2x16xi16>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]]] : memref<2x16xi16>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]]] : memref<2x16xi16>
  %src = memref.alloc() : memref<2x16xi16>
  %dst = memref.alloc() : memref<2x16xi16>
  hivm.hir.vcumsum ins(%src : memref<2x16xi16>) outs(%dst : memref<2x16xi16>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumsum_3d_f16() {
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST24:.*]] = arith.constant 24 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x24x16xf16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x24x16xf16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[CST24]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<2x24x16xf16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<2x24x16xf16>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<2x24x16xf16>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[pre]]] : memref<2x24x16xf16>
  // CHECK: %[[dst:.*]] = arith.addf %[[input]], %[[cum]] : f16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<2x24x16xf16>
  %src = memref.alloc() : memref<2x24x16xf16>
  %dst = memref.alloc() : memref<2x24x16xf16>
  hivm.hir.vcumsum ins(%src : memref<2x24x16xf16>) outs(%dst : memref<2x24x16xf16>) cum_dims = [2] reverse = false
  return
}

// -----
func.func @test_cumsum_1d_i64() {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<16xi64>
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]]] : memref<16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]]] : memref<16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<16xi64>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<16xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<16xi64>
  %src = memref.alloc() : memref<16xi64>
  %dst = memref.alloc() : memref<16xi64>
  hivm.hir.vcumsum ins(%src : memref<16xi64>) outs(%dst : memref<16xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumsum_2d_i64() {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[CST2]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg1]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[pre]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg1]], %[[arg0]]] : memref<2x16xi64>
  %src = memref.alloc() : memref<2x16xi64>
  %dst = memref.alloc() : memref<2x16xi64>
  hivm.hir.vcumsum ins(%src : memref<2x16xi64>) outs(%dst : memref<2x16xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumsum_3d_i64() {
  // CHECK-DAG: %[[CST24:.*]] = arith.constant 24 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x24x16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x24x16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[CST24]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<2x24x16xi64>
  %src = memref.alloc() : memref<2x24x16xi64>
  %dst = memref.alloc() : memref<2x24x16xi64>
  hivm.hir.vcumsum ins(%src : memref<2x24x16xi64>) outs(%dst : memref<2x24x16xi64>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumsum_1d_dynshape_lastdim(%src : memref<?xi32>, %dst : memref<?xi32>) {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[CST0]]] : memref<?xi32>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0:.*]][%[[CST0]]] : memref<?xi32>
  // CHECK: %[[dyndim:.*]] = memref.dim %[[ALLOC_0]], %[[CST0]] : memref<?xi32>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[dyndim]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<?xi32>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<?xi32>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i32
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<?xi32>
  hivm.hir.vcumsum ins(%src : memref<?xi32>) outs(%dst : memref<?xi32>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumsum_2d_dynshape_highdim(%src : memref<?x?xi64>, %dst : memref<?x?xi64>) {
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST1]] : memref<?x?xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[CST0]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0]], %[[CST0]] : memref<?x?xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg1]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[pre]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg1]], %[[arg0]]] : memref<?x?xi64>
  hivm.hir.vcumsum ins(%src : memref<?x?xi64>) outs(%dst : memref<?x?xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumsum_3d_dynshape_i64_highdim(%src : memref<?x?x?xi64>, %dst : memref<?x?x?xi64>) {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST0]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[dyndim2:.*]] = memref.dim %[[ALLOC_0]], %[[CST2]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[dyndim2]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0]], %[[CST1]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[dst:.*]] = arith.addi %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<?x?x?xi64>
  hivm.hir.vcumsum ins(%src : memref<?x?x?xi64>) outs(%dst : memref<?x?x?xi64>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumsum_3d_synshape_f32_lastdim(%src : memref<?x?x?xf32>, %dst : memref<?x?x?xf32>) {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST0]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0]], %[[CST1]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<?x?x?xf32>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<?x?x?xf32>
  // CHECK: %[[dyndim2:.*]] = memref.dim %[[ALLOC_0]], %[[CST2]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[dyndim2]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<?x?x?xf32>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[pre]]] : memref<?x?x?xf32>
  // CHECK: %[[dst:.*]] = arith.addf %[[input]], %[[cum]] : f32
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<?x?x?xf32>
  hivm.hir.vcumsum ins(%src : memref<?x?x?xf32>) outs(%dst : memref<?x?x?xf32>) cum_dims = [2] reverse = false
  return
}

//===----------------------------------------------------------------------===//
// Test VCumprod Decompose MemRef
//===----------------------------------------------------------------------===//
// -----
func.func @test_cumprod_1d_i16() {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<16xi16>
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]]] : memref<16xi16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]]] : memref<16xi16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<16xi16>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<16xi16>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<16xi16>
  %src = memref.alloc() : memref<16xi16>
  %dst = memref.alloc() : memref<16xi16>
  hivm.hir.vcumprod ins(%src : memref<16xi16>) outs(%dst : memref<16xi16>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumprod_2d_i16() {
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xi16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x16xi16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[CST0]]] : memref<2x16xi16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]]] : memref<2x16xi16>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]]] : memref<2x16xi16>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]]] : memref<2x16xi16>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]]] : memref<2x16xi16>
  %src = memref.alloc() : memref<2x16xi16>
  %dst = memref.alloc() : memref<2x16xi16>
  hivm.hir.vcumprod ins(%src : memref<2x16xi16>) outs(%dst : memref<2x16xi16>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumprod_3d_f16() {
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST24:.*]] = arith.constant 24 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x24x16xf16>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x24x16xf16>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[CST24]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<2x24x16xf16>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<2x24x16xf16>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<2x24x16xf16>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[pre]]] : memref<2x24x16xf16>
  // CHECK: %[[dst:.*]] = arith.mulf %[[input]], %[[cum]] : f16
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<2x24x16xf16>
  %src = memref.alloc() : memref<2x24x16xf16>
  %dst = memref.alloc() : memref<2x24x16xf16>
  hivm.hir.vcumprod ins(%src : memref<2x24x16xf16>) outs(%dst : memref<2x24x16xf16>) cum_dims = [2] reverse = false
  return
}

// -----
func.func @test_cumprod_1d_i64() {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<16xi64>
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]]] : memref<16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]]] : memref<16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<16xi64>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<16xi64>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<16xi64>
  %src = memref.alloc() : memref<16xi64>
  %dst = memref.alloc() : memref<16xi64>
  hivm.hir.vcumprod ins(%src : memref<16xi64>) outs(%dst : memref<16xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumprod_2d_i64() {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[CST0]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[CST2]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg1]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[pre]], %[[arg0]]] : memref<2x16xi64>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg1]], %[[arg0]]] : memref<2x16xi64>
  %src = memref.alloc() : memref<2x16xi64>
  %dst = memref.alloc() : memref<2x16xi64>
  hivm.hir.vcumprod ins(%src : memref<2x16xi64>) outs(%dst : memref<2x16xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumprod_3d_i64() {
  // CHECK-DAG: %[[CST24:.*]] = arith.constant 24 : index
  // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x24x16xi64>
  // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<2x24x16xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[CST2]] step %[[CST1]]
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[CST16]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[CST24]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]], %[[arg1]]] : memref<2x24x16xi64>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<2x24x16xi64>
  %src = memref.alloc() : memref<2x24x16xi64>
  %dst = memref.alloc() : memref<2x24x16xi64>
  hivm.hir.vcumprod ins(%src : memref<2x24x16xi64>) outs(%dst : memref<2x24x16xi64>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumprod_1d_dynshape_lastdim(%src : memref<?xi32>, %dst : memref<?xi32>) {
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[CST0]]] : memref<?xi32>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0:.*]][%[[CST0]]] : memref<?xi32>
  // CHECK: %[[dyndim:.*]] = memref.dim %[[ALLOC_0]], %[[CST0]] : memref<?xi32>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST1]] to %[[dyndim]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]]] : memref<?xi32>
  // CHECK: %[[arg1:.*]] = arith.subi %[[arg0]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg1]]] : memref<?xi32>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i32
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]]] : memref<?xi32>
  hivm.hir.vcumprod ins(%src : memref<?xi32>) outs(%dst : memref<?xi32>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumprod_2d_dynshape_highdim(%src : memref<?x?xi64>, %dst : memref<?x?xi64>) {
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST1]] : memref<?x?xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[CST0]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[CST0]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0]], %[[CST0]] : memref<?x?xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST1]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg1]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg1]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[pre]], %[[arg0]]] : memref<?x?xi64>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg1]], %[[arg0]]] : memref<?x?xi64>
  hivm.hir.vcumprod ins(%src : memref<?x?xi64>) outs(%dst : memref<?x?xi64>) cum_dims = [0] reverse = false
  return
}

// -----
func.func @test_cumprod_3d_dynshape_i64_highdim(%src : memref<?x?x?xi64>, %dst : memref<?x?x?xi64>) {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST0]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[dyndim2:.*]] = memref.dim %[[ALLOC_0]], %[[CST2]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[dyndim2]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[CST0]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0]], %[[CST1]] : memref<?x?x?xi64>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[pre]], %[[arg1]]] : memref<?x?x?xi64>
  // CHECK: %[[dst:.*]] = arith.muli %[[input]], %[[cum]] : i64
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg2]], %[[arg1]]] : memref<?x?x?xi64>
  hivm.hir.vcumprod ins(%src : memref<?x?x?xi64>) outs(%dst : memref<?x?x?xi64>) cum_dims = [1] reverse = false
  return
}

// -----
func.func @test_cumprod_3d_synshape_f32_lastdim(%src : memref<?x?x?xf32>, %dst : memref<?x?x?xf32>) {
  // CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[dyndim0:.*]] = memref.dim %[[ALLOC_0:.*]], %[[CST0]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg0:.*]] = %[[CST0]] to %[[dyndim0]] step %[[CST1]]
  // CHECK: %[[dyndim1:.*]] = memref.dim %[[ALLOC_0]], %[[CST1]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg1:.*]] = %[[CST0]] to %[[dyndim1]] step %[[CST1]]
  // CHECK: %[[input0:.*]] = memref.load %[[ALLOC:.*]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<?x?x?xf32>
  // CHECK: memref.store %[[input0]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[CST0]]] : memref<?x?x?xf32>
  // CHECK: %[[dyndim2:.*]] = memref.dim %[[ALLOC_0]], %[[CST2]] : memref<?x?x?xf32>
  // CHECK: scf.for %[[arg2:.*]] = %[[CST1]] to %[[dyndim2]] step %[[CST1]]
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<?x?x?xf32>
  // CHECK: %[[pre:.*]] = arith.subi %[[arg2]], %[[CST1]] : index
  // CHECK: %[[cum:.*]] = memref.load %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[pre]]] : memref<?x?x?xf32>
  // CHECK: %[[dst:.*]] = arith.mulf %[[input]], %[[cum]] : f32
  // CHECK: memref.store %[[dst]], %[[ALLOC_0]][%[[arg0]], %[[arg1]], %[[arg2]]] : memref<?x?x?xf32>
  hivm.hir.vcumprod ins(%src : memref<?x?x?xf32>) outs(%dst : memref<?x?x?xf32>) cum_dims = [2] reverse = false
  return
}

func.func @test_vshr_vv1() {
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     arith.shrsi
  // CHECK:     memref.store
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   hivm.hir.vshr ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

func.func @test_vshl_vv1() {
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     arith.shli
  // CHECK:     memref.store
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   hivm.hir.vshl ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

func.func @test_vshr_vv_inline_brc_dim1() {
  // CHECK: %[[cst0:.*]] = arith.constant 0 : index
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC:.*]][%[[arg0:.*]], %[[cst0]]] : memref<64x1xi64>
  // CHECK:     arith.shrsi
  // CHECK:     memref.store
   %allocIn0 = memref.alloc() : memref<64x8xi64>
   %allocIn1 = memref.alloc() : memref<64x1xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   hivm.hir.vshr {broadcast = array<i64: 1>} ins(%allocIn0, %allocIn1 : memref<64x8xi64>, memref<64x1xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}

func.func @test_vshr_vv_inline_brc_dim0() {
  // CHECK: %[[cst0:.*]] = arith.constant 0 : index
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK: %[[input:.*]] = memref.load %[[ALLOC:.*]][%[[cst0]], %[[arg1:.*]]] : memref<1x8xi64>
  // CHECK:     arith.shrsi
  // CHECK:     memref.store
   %allocIn0 = memref.alloc() : memref<1x8xi64>
   %allocIn1 = memref.alloc() : memref<64x8xi64>
   %allocOut = memref.alloc() : memref<64x8xi64>
   hivm.hir.vshr {broadcast = array<i64: 0>} ins(%allocIn0, %allocIn1 : memref<1x8xi64>, memref<64x8xi64>) outs(%allocOut : memref<64x8xi64>)
   return
}
