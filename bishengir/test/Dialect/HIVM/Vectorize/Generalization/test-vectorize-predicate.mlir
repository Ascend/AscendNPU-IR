// RUN: bishengir-opt %s --split-input-file \
// RUN:   --hfusion-pre-vectorization-fusion --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Floating predicates deliberately check ordered/unordered spellings. The
// unordered `ne` result is true when either operand is NaN; the other five
// predicates are ordered and false for NaN operands.

// CHECK-HFUSION-LABEL: func.func @hfusion_cmp_float(
// CHECK-HFUSION: arith.cmpf oeq, {{.*}} : vector<8x8xf32>
// CHECK-HFUSION: arith.cmpf une, {{.*}} : vector<8x8xf32>
// CHECK-HFUSION: arith.cmpf olt, {{.*}} : vector<8x8xf32>
// CHECK-HFUSION: arith.cmpf ole, {{.*}} : vector<8x8xf32>
// CHECK-HFUSION: arith.cmpf ogt, {{.*}} : vector<8x8xf32>
// CHECK-HFUSION: arith.cmpf oge, {{.*}} : vector<8x8xf32>
func.func @hfusion_cmp_float(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> (tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>,
        tensor<8x8xi1>, tensor<8x8xi1>) {
  %empty = tensor.empty() : tensor<8x8xi1>
  %eq = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %ne = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %lt = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %le = hfusion.compare {compare_fn = #hfusion.compare_fn<vle>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %gt = hfusion.compare {compare_fn = #hfusion.compare_fn<vgt>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %ge = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  return %eq, %ne, %lt, %le, %gt, %ge : tensor<8x8xi1>, tensor<8x8xi1>,
      tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_cmp_float(
// CHECK-HIVM: arith.cmpf oeq, {{.*}} : vector<8x8xf32>
// CHECK-HIVM: arith.cmpf une, {{.*}} : vector<8x8xf32>
// CHECK-HIVM: arith.cmpf olt, {{.*}} : vector<8x8xf32>
// CHECK-HIVM: arith.cmpf ole, {{.*}} : vector<8x8xf32>
// CHECK-HIVM: arith.cmpf ogt, {{.*}} : vector<8x8xf32>
// CHECK-HIVM: arith.cmpf oge, {{.*}} : vector<8x8xf32>
func.func @hivm_cmp_float(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> (tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>,
        tensor<8x8xi1>, tensor<8x8xi1>) attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xi1>
  %eq = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <eq> -> tensor<8x8xi1>
  %ne = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <ne> -> tensor<8x8xi1>
  %lt = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <lt> -> tensor<8x8xi1>
  %le = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <le> -> tensor<8x8xi1>
  %gt = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <gt> -> tensor<8x8xi1>
  %ge = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <ge> -> tensor<8x8xi1>
  return %eq, %ne, %lt, %le, %gt, %ge : tensor<8x8xi1>, tensor<8x8xi1>,
      tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>, tensor<8x8xi1>
}

// -----

// Signed and unsigned integer predicates must not collapse to one another.

// CHECK-HFUSION-LABEL: func.func @hfusion_cmp_int(
// CHECK-HFUSION: arith.cmpi slt, {{.*}} : vector<8x8xi32>
// CHECK-HFUSION: arith.cmpi ult, {{.*}} : vector<8x8xi32>
func.func @hfusion_cmp_int(%lhs: tensor<8x8xi32>, %rhs: tensor<8x8xi32>)
    -> (tensor<8x8xi1>, tensor<8x8xi1>) {
  %empty = tensor.empty() : tensor<8x8xi1>
  %signed = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
      ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  %unsigned = hfusion.compare {compare_fn = #hfusion.compare_fn<vult>}
      ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi1>) -> tensor<8x8xi1>
  return %signed, %unsigned : tensor<8x8xi1>, tensor<8x8xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_cmp_int(
// CHECK-HIVM: arith.cmpi slt, {{.*}} : vector<8x8xi32>
// CHECK-HIVM: arith.cmpi ult, {{.*}} : vector<8x8xi32>
func.func @hivm_cmp_int(%lhs: tensor<8x8xi32>, %rhs: tensor<8x8xi32>)
    -> (tensor<8x8xi1>, tensor<8x8xi1>) attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xi1>
  %signed = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <lt> -> tensor<8x8xi1>
  %unsigned = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<8x8xi32>, tensor<8x8xi32>)
      outs(%empty : tensor<8x8xi1>) compare_mode = <lt> is_signed : false
      -> tensor<8x8xi1>
  return %signed, %unsigned : tensor<8x8xi1>, tensor<8x8xi1>
}

// -----

// Boolean equality and inequality.

// CHECK-HFUSION-LABEL: func.func @hfusion_cmp_i1(
// CHECK-HFUSION: arith.cmpi eq, {{.*}} : vector<16x16xi1>
// CHECK-HFUSION: arith.cmpi ne, {{.*}} : vector<16x16xi1>
func.func @hfusion_cmp_i1(%lhs: tensor<16x16xi1>, %rhs: tensor<16x16xi1>)
    -> (tensor<16x16xi1>, tensor<16x16xi1>) {
  %empty = tensor.empty() : tensor<16x16xi1>
  %eq = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
  %ne = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
  return %eq, %ne : tensor<16x16xi1>, tensor<16x16xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_cmp_i1(
// CHECK-HIVM: arith.cmpi eq, {{.*}} : vector<16x16xi1>
// CHECK-HIVM: arith.cmpi ne, {{.*}} : vector<16x16xi1>
func.func @hivm_cmp_i1(%lhs: tensor<16x16xi1>, %rhs: tensor<16x16xi1>)
    -> (tensor<16x16xi1>, tensor<16x16xi1>)
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xi1>
  %eq = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) compare_mode = <eq>
      -> tensor<16x16xi1>
  %ne = hivm.hir.vcmp
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) compare_mode = <ne>
      -> tensor<16x16xi1>
  return %eq, %ne : tensor<16x16xi1>, tensor<16x16xi1>
}
