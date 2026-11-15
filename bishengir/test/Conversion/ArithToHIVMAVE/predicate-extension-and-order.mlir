// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize=phase=backend -convert-arith-to-hivmave --verify-each | FileCheck %s --implicit-check-not=arith.extui --implicit-check-not=arith.extsi --implicit-check-not=arith.cmpf

// CHECK-LABEL: @boolean_unsigned_i8
// CHECK: arith.constant 1 : i8
// CHECK: arith.constant 0 : i8
// CHECK: ave.hir.vsel
// CHECK: return
func.func @boolean_unsigned_i8(%predicate: vector<64xi1>) -> vector<64xi8>
    attributes {hivm.vector_function} {
  %result = arith.extui %predicate : vector<64xi1> to vector<64xi8>
  return %result : vector<64xi8>
}

// CHECK-LABEL: @boolean_unsigned_i32
// CHECK: arith.constant 1 : i32
// CHECK: arith.constant 0 : i32
// CHECK: ave.hir.vsel
// CHECK: return
func.func @boolean_unsigned_i32(%predicate: vector<64xi1>) -> vector<64xi32>
    attributes {hivm.vector_function} {
  %result = arith.extui %predicate : vector<64xi1> to vector<64xi32>
  return %result : vector<64xi32>
}

// CHECK-LABEL: @boolean_signed_i32
// CHECK: arith.constant -1 : i32
// CHECK: arith.constant 0 : i32
// CHECK: ave.hir.vsel
// CHECK: return
func.func @boolean_signed_i32(%predicate: vector<64xi1>) -> vector<64xi32>
    attributes {hivm.vector_function} {
  %result = arith.extsi %predicate : vector<64xi1> to vector<64xi32>
  return %result : vector<64xi32>
}

// CHECK-LABEL: @boolean_unsigned_i64
// CHECK: arith.constant 1 : i32
// CHECK: ave.hir.vsel
// CHECK: ave.hir.vextui
// CHECK: return
func.func @boolean_unsigned_i64(%predicate: vector<64xi1>) -> vector<64xi64>
    attributes {hivm.vector_function} {
  %result = arith.extui %predicate : vector<64xi1> to vector<64xi64>
  return %result : vector<64xi64>
}

// CHECK-LABEL: @boolean_signed_i64
// CHECK: arith.constant -1 : i32
// CHECK: ave.hir.vsel
// CHECK: ave.hir.vextsi
// CHECK: return
func.func @boolean_signed_i64(%predicate: vector<64xi1>) -> vector<64xi64>
    attributes {hivm.vector_function} {
  %result = arith.extsi %predicate : vector<64xi1> to vector<64xi64>
  return %result : vector<64xi64>
}

// CHECK-LABEL: @ordered_f32
// CHECK: ave.hir.vcmp <EQ> %arg0, %arg0
// CHECK: ave.hir.vcmp <EQ> %arg1, %arg1
// CHECK: ave.hir.preg.and <b32>
// CHECK-NOT: ave.hir.preg.not
// CHECK: return
func.func @ordered_f32(%lhs: vector<64xf32>, %rhs: vector<64xf32>)
    -> vector<64xi1> attributes {hivm.vector_function} {
  %result = arith.cmpf ord, %lhs, %rhs : vector<64xf32>
  return %result : vector<64xi1>
}

// CHECK-LABEL: @unordered_f16
// CHECK: ave.hir.vcmp <EQ> %arg0, %arg0
// CHECK: ave.hir.vcmp <EQ> %arg1, %arg1
// CHECK: ave.hir.preg.and <b16>
// CHECK: ave.hir.preg.not <b16>
// CHECK: return
func.func @unordered_f16(%lhs: vector<128xf16>, %rhs: vector<128xf16>)
    -> vector<128xi1> attributes {hivm.vector_function} {
  %result = arith.cmpf uno, %lhs, %rhs : vector<128xf16>
  return %result : vector<128xi1>
}
