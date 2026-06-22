// RUN: bishengir-opt %s -convert-arith-to-hivmave -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// Test rank-2 [1, N] vector shapes in convert-arith-to-hivmave.
// These arise from hfusion-auto-vectorize-v2 which produces vector<1xN>
// when extracting tensor<1xN> slices. Both the VFCmpOp path (i8/i16/… → i1)
// and the Preg path (i1 → i1 via PregNot/PregOr/PregAnd) must be handled,
// as well as arith.select with rank-2 vectors (VFSelectOp) and
// arith.constant dense<false> of rank-2 i1 vector (PGE creation).

// -----
// CHECK-LABEL: @test_cmpi_ne_i8_1x256
func.func @test_cmpi_ne_i8_1x256(%arg0: vector<1x256xi8>, %arg1: vector<1x256xi8>) -> vector<1x256xi1> attributes {hivm.vector_function} {
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi8> to vector<256xi8>
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi8> to vector<256xi8>
  // CHECK: ave.hir.pge <ALL> : vector<256xi1>
  // CHECK: ave.hir.vcmp <NE>
  // CHECK: vector.shape_cast %{{.*}} : vector<256xi1> to vector<1x256xi1>
  %0 = arith.cmpi ne, %arg0, %arg1 : vector<1x256xi8>
  return %0 : vector<1x256xi1>
}

// -----
// CHECK-LABEL: @test_cmpi_ne_i1_1x256
func.func @test_cmpi_ne_i1_1x256(%arg0: vector<1x256xi1>, %arg1: vector<1x256xi1>) -> vector<1x256xi1> attributes {hivm.vector_function} {
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi1> to vector<256xi1>
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi1> to vector<256xi1>
  // CHECK: ave.hir.pge <ALL> : vector<256xi1>
  // CHECK: ave.hir.vcmp <NE>
  // CHECK: vector.shape_cast %{{.*}} : vector<256xi1> to vector<1x256xi1>
  %0 = arith.cmpi ne, %arg0, %arg1 : vector<1x256xi1>
  return %0 : vector<1x256xi1>
}

// -----
// CHECK-LABEL: @test_cmpi_ult_i1_1x256
func.func @test_cmpi_ult_i1_1x256(%arg0: vector<1x256xi1>, %arg1: vector<1x256xi1>) -> vector<1x256xi1> attributes {hivm.vector_function} {
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi1> to vector<256xi1>
  // CHECK: vector.shape_cast %{{.*}} : vector<1x256xi1> to vector<256xi1>
  // CHECK: ave.hir.pge <ALL> : vector<256xi1>
  // CHECK: ave.hir.preg.not
  // CHECK: ave.hir.preg.and
  // CHECK: vector.shape_cast %{{.*}} : vector<256xi1> to vector<1x256xi1>
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<1x256xi1>
  return %0 : vector<1x256xi1>
}

// -----
// CHECK-LABEL: @test_select_f16_1x128
func.func @test_select_f16_1x128(%cond: vector<1x128xi1>, %a: vector<1x128xf16>, %b: vector<1x128xf16>) -> vector<1x128xf16> attributes {hivm.vector_function} {
  // CHECK: vector.shape_cast %{{.*}} : vector<1x128xi1> to vector<128xi1>
  // CHECK: vector.shape_cast %{{.*}} : vector<1x128xf16> to vector<128xf16>
  // CHECK: vector.shape_cast %{{.*}} : vector<1x128xf16> to vector<128xf16>
  // CHECK: ave.hir.vsel
  // CHECK: vector.shape_cast %{{.*}} : vector<128xf16> to vector<1x128xf16>
  %0 = arith.select %cond, %a, %b : vector<1x128xi1>, vector<1x128xf16>
  return %0 : vector<1x128xf16>
}

// -----
// CHECK-LABEL: @test_constant_dense_false_1x256
func.func @test_constant_dense_false_1x256() -> vector<1x256xi1> attributes {hivm.vector_function} {
  // CHECK: ave.hir.pge <ALLF> : vector<256xi1>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : vector<256xi1> to vector<1x256xi1>
  %cst = arith.constant dense<false> : vector<1x256xi1>
  return %cst : vector<1x256xi1>
}

// -----
// CHECK-LABEL: @test_constant_dense_true_1x256
func.func @test_constant_dense_true_1x256() -> vector<1x256xi1> attributes {hivm.vector_function} {
  // CHECK: ave.hir.pge <ALL> : vector<256xi1>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : vector<256xi1> to vector<1x256xi1>
  %cst = arith.constant dense<true> : vector<1x256xi1>
  return %cst : vector<1x256xi1>
}
