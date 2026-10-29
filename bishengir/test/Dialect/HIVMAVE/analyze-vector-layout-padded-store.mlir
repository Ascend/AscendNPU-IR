// RUN: bishengir-opt %s -analyze-vector-layout -mlir-print-vector-layout-attr \
// RUN:   | FileCheck %s --implicit-check-not=b8_4vl \
// RUN:       --implicit-check-not="func_dist_type<pk4>"

// A masked store value is built for a whole vector register, so it can be wider
// than the region it stores into: both functions below hold 256 i8 lanes but
// write a 128 element row, and only the first half of the mask is live.
//
// A single VSTS consumes one register, i.e. it covers VL_BITS / laneBitWidth(
// layout) lanes. For PK/B8_2VL that is 2048 / 16 = 128 lanes, exactly the
// destination, so the layout is fine even though the padded value itself spans
// two registers. PK4/B8_4VL would cover only 64 lanes and silently write half of
// the row, which is what must not be selected.

// CHECK-LABEL: func.func @padded_store
// CHECK: ave.hir.masked_store <NORM_B8> {{.*}} {functionType = #ave.func_dist_type<pk>
// CHECK-SAME: vector<256xi8, #ave.vector_layout<{mem = #ave.vec_mem_type<b8_2vl>}>>
func.func @padded_store(%dst: memref<128xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i8
  %zero = arith.constant 0.0 : f16
  %fm = ave.hir.pge <ALL> : vector<128xi1>
  %f = ave.hir.broadcast %zero, %fm : f16, vector<128xi1> -> vector<128xf16>
  %m = ave.hir.vcmp <EQ> %f, %f, %fm : vector<128xf16>, vector<128xi1> -> vector<128xi1>
  %wideMask = builtin.unrealized_conversion_cast %m : vector<128xi1> to vector<256xi1>
  %v = ave.hir.broadcast %one, %wideMask : i8, vector<256xi1> -> vector<256xi8>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %wideMask, %v : memref<128xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}

// A dynamic destination extent cannot be bounded, so no layout may be rejected:
// pruning there would turn a valid VF into "No Solve".

// CHECK-LABEL: func.func @padded_store_dynamic_destination
// CHECK: ave.hir.masked_store
func.func @padded_store_dynamic_destination(%dst: memref<?xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i8
  %zero = arith.constant 0.0 : f16
  %fm = ave.hir.pge <ALL> : vector<128xi1>
  %f = ave.hir.broadcast %zero, %fm : f16, vector<128xi1> -> vector<128xf16>
  %m = ave.hir.vcmp <EQ> %f, %f, %fm : vector<128xf16>, vector<128xi1> -> vector<128xi1>
  %wideMask = builtin.unrealized_conversion_cast %m : vector<128xi1> to vector<256xi1>
  %v = ave.hir.broadcast %one, %wideMask : i8, vector<256xi1> -> vector<256xi8>
  ave.hir.masked_store <NORM_B8> %dst[%c0], %wideMask, %v : memref<?xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  return
}
