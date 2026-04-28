// RUN: bishengir-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @hivm_binary_test
func.func @hivm_binary_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<1> : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vadd %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %0 = ave.hir.vadd %arg0, %arg1, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vsub %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %1 = ave.hir.vsub %arg0, %0, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmul %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %2 = ave.hir.vmul %arg0, %1, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vdiv %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %3 = ave.hir.vdiv %arg0, %2, %mask {cast = #hivm.cast<cast_signed>} : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmax %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %4 = ave.hir.vmax %arg0, %3, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmin %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %5 = ave.hir.vmin %arg0, %4, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vabs_diff %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %6 = ave.hir.vabs_diff %arg0, %5, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vsadd %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %7 = ave.hir.vsadd %arg0, %6, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vssub %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %8 = ave.hir.vssub %arg0, %7, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vrnd %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %9 = ave.hir.vrnd %arg0, %8, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vprelu %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %10 = ave.hir.vprelu %arg0, %9, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vand %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %11 = ave.hir.vand %arg0, %10, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vor %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %12 = ave.hir.vor %arg0, %11, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vxor %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %13 = ave.hir.vxor %arg0, %12, %mask : vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vdivf %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %14 = ave.hir.vdivf %arg0, %13, %mask : vector<64xf32>, vector<256xi1>
  return %14 : vector<64xf32>
}

// -----

// CHECK-LABEL: @hivm_merging_add_test
func.func @hivm_merging_add_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<1> : vector<256xi1>
  %pass_thru = arith.constant dense<0.0> : vector<64xf32>
  // %{{.*}} = ave.hir.vadd
  %0 = ave.hir.vadd %arg0, %arg1, %mask, %pass_thru : vector<64xf32>, vector<256xi1>, vector<64xf32>
  %1 = ave.hir.vsub %arg0, %0, %mask, %pass_thru : vector<64xf32>, vector<256xi1>, vector<64xf32>
  %2 = ave.hir.vmul %arg0, %1, %mask, %pass_thru : vector<64xf32>, vector<256xi1>, vector<64xf32>
  %3 = ave.hir.vdiv %arg0, %2, %mask, %pass_thru {cast = #hivm.cast<cast_signed>} : vector<64xf32>, vector<256xi1>, vector<64xf32>
  return %3 : vector<64xf32>
}

// -----

// abs, neg, not, relu, exp, sqrt, rsqrt vrec, ln, bit1_count, count_leading_sign_bits
// CHECK-LABEL: @hivm_unary_test
func.func @hivm_unary_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<1> : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vsqrt %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %0 = ave.hir.vsqrt %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vrsqrt %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %1 = ave.hir.vrsqrt %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vabs %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %2 = ave.hir.vabs %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vneg %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %3 = ave.hir.vneg %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vnot %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %4 = ave.hir.vnot %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vrelu %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %5 = ave.hir.vrelu %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vexp %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %6 = ave.hir.vexp %arg0, %mask: vector<64xf32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vln %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %7 = ave.hir.vln %arg0, %mask: vector<64xf32>, vector<256xi1>
  return %7 : vector<64xf32>
}

// -----

// mula, add3, adif, sad
// CHECK-LABEL: @hivm_ternary_test
func.func @hivm_ternary_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>, %arg2: vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<1> : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmula %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1>
  %0 = ave.hir.vmula %arg0, %arg1, %arg2, %mask: vector<64xf32>, vector<256xi1>
  return %0 : vector<64xf32>
}

// -----

// adds, muls, lrelus, maxs, mins, shls, shrs, rnds, sadds
// CHECK-LABEL: @hivm_vecscalar_test
func.func @hivm_vecscalar_test(%arg0: vector<64xf32>, %arg1: f32, %arg2: vector<256xi1>) -> vector<64xf32> {
  %true = arith.constant true
  // CHECK: %{{.*}} = ave.hir.vadds %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %0 = ave.hir.vadds %arg0, %arg1, %arg2: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmuls %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %1 = ave.hir.vmuls %arg0, %arg1, %arg2: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vlrelus %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %2 = ave.hir.vlrelus %arg0, %arg1, %arg2: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmaxs %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %3 = ave.hir.vmaxs %arg0, %arg1, %arg2: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vmins %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %4 = ave.hir.vmins %arg0, %arg1, %arg2: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vshls %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %5 = ave.hir.vshls %arg0, %arg1, %arg2, %true: vector<64xf32>, f32, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vshrs %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1>
  %6 = ave.hir.vshrs %arg0, %arg1, %arg2, %true: vector<64xf32>, f32, vector<256xi1>
  return %6 : vector<64xf32>
}

// -----

// preg binary operations:
// CHECK-LABEL: @hivm_preg_operations_test
func.func @hivm_preg_operations_test(%arg0: vector<256xi1>, %arg1: vector<256xi1>) -> vector<256xi1> {
  %mask = arith.constant dense<1> : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.preg.and <b32> %{{.*}}, %{{.*}}, %{{.*}} : vector<256xi1>
  %1 = ave.hir.preg.and <b32> %arg0, %arg1, %mask : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.preg.or <b32> %{{.*}}, %{{.*}}, %{{.*}} : vector<256xi1>
  %2 = ave.hir.preg.or <b32> %arg0, %arg1, %mask : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.preg.xor <b32> %{{.*}}, %{{.*}}, %{{.*}} : vector<256xi1>
  %3 = ave.hir.preg.xor <b32> %arg0, %arg1, %mask : vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.preg.not <b32> %{{.*}}, %{{.*}} : vector<256xi1>
  %4 = ave.hir.preg.not <b32> %arg0, %mask : vector<256xi1>
  return %4 : vector<256xi1>
}

// -----

// CHECK-LABEL: @hivm_gather
func.func @hivm_gather(%arg0: memref<?xf32>, %arg1: index, %arg2: vector<64xindex>, %arg3: vector<256xi1>) {
  %0 = ave.hir.vgather %arg0[%arg1][%arg2], %arg3 : memref<?xf32>, vector<64xindex>, vector<256xi1> into vector<64xf32>
  return
}

// -----

// CHECK-LABEL: @hivm_scatter
func.func @hivm_scatter(%arg0: memref<?xf32>, %arg1: index, %arg2: vector<64xindex>, 
                        %arg3: vector<64xf32>, %mask: vector<256xi1>) {
  ave.hir.vscatter %arg0[%arg1][%arg2], %mask, %arg3 : memref<?xf32>, vector<64xindex>, vector<256xi1>, vector<64xf32>
  return
}

// -----

// EQ, NE, GT, GE, LT, LE
// CHECK-LABEL: @hivm_cmp_test 
func.func @hivm_cmp_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>, %arg2: vector<256xi1>) -> vector<256xi1> {
  // CHECK: %{{.*}} = ave.hir.vcmp <EQ> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %0 = ave.hir.vcmp <EQ> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmp <NE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %1 = ave.hir.vcmp <NE> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmp <GT> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %2 = ave.hir.vcmp <GT> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmp <GE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %3 = ave.hir.vcmp <GE> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmp <LT> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %4 = ave.hir.vcmp <LT> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmp <LE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  %5 = ave.hir.vcmp <LE> %arg0, %arg1, %arg2 : vector<64xf32>, vector<256xi1> -> vector<256xi1>
  return %5 : vector<256xi1>
}

// -----

// EQ, NE, GT, GE, LT, LE
// CHECK-LABEL: @hivm_cmp_scalar_test
func.func @hivm_cmp_scalar_test(%arg0: vector<64xf32>, %arg1: f32, %arg2: vector<256xi1>) -> vector<256xi1> {
  // CHECK: %{{.*}} = ave.hir.vcmps <EQ> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %0 = ave.hir.vcmps <EQ> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmps <NE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %1 = ave.hir.vcmps <NE> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmps <GT> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %2 = ave.hir.vcmps <GT> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmps <GE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %3 = ave.hir.vcmps <GE> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmps <LT> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %4 = ave.hir.vcmps <LT> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vcmps <LE> %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  %5 = ave.hir.vcmps <LE> %arg0, %arg1, %arg2 : vector<64xf32>, f32, vector<256xi1> -> vector<256xi1>
  return %5 : vector<256xi1>
}

// -----

// CHECK-LABEL: @hivm_broadcast_scalar
func.func @hivm_broadcast_scalar(%arg0: f32) -> vector<64xf32> {
  %res = ave.hir.scalar_broadcast %arg0 : f32 -> vector<64xf32>
  return %res : vector<64xf32>
}

// -----

// CHECK-LABEL: @hivm_broadcast
func.func @hivm_broadcast(%arg0: f32, %mask: vector<256xi1>) -> vector<64xf32> {
  %res = ave.hir.broadcast %arg0, %mask : f32, vector<256xi1> -> vector<64xf32>
  return %res : vector<64xf32>
}

// -----

// CHECK-LABEL: @hivm_vec_elemwise_shifts
func.func @hivm_vec_elemwise_shifts(%arg0: vector<64xi32>, %arg1: vector<64xi32>, %mask: vector<256xi1>) -> vector<64xi32> {
  %true = arith.constant true
  %false = arith.constant false
  // CHECK: %{{.*}} = ave.hir.vshl %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi32>, vector<256xi1>
  %0 = ave.hir.vshl %arg0, %arg1, %mask, %false : vector<64xi32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vshr %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi32>, vector<256xi1>
  %1 = ave.hir.vshr %arg0, %0, %mask, %false : vector<64xi32>, vector<256xi1>
  // CHECK: %{{.*}} = ave.hir.vshr %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<64xi32>, vector<256xi1>
  %2 = ave.hir.vshr %arg0, %1, %mask, %true : vector<64xi32>, vector<256xi1>
  return %2 : vector<64xi32>
}