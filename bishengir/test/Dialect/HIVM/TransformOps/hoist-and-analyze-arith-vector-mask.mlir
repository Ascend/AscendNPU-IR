// RUN: bishengir-opt --hoist-constant-masks --cse --arith-vector-mask-analyze -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --hoist-constant-masks --cse --arith-vector-mask-analyze --convert-vector-to-hivmave --convert-arith-to-hivmave -split-input-file %s | FileCheck %s --check-prefix=LOWER
// RUN: bishengir-opt --hoist-constant-masks --cse --arith-vector-mask-analyze --convert-arith-to-hivmave --convert-vector-to-hivmave -split-input-file %s | FileCheck %s --check-prefix=LOWER

// Hoisting makes sibling masks available before the shared arithmetic. CSE
// merges them before analysis assigns indices and writes the direct reference.
// CHECK-LABEL: issue3_identical_masks_in_branches
// CHECK-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CHECK-NEXT: annotation.mark %[[MASK]] {mask_op_idx = 0 : i32} : vector<64xi1>
// CHECK: %[[SHARED:.*]] = arith.addi {{.*}} : vector<64xi32>
// CHECK-NEXT: annotation.mark %[[SHARED]] {reached_mask_ops_idx = 0 : i32} keys = ["reached_mask_value"] values = [%[[MASK]] : vector<64xi1>] : vector<64xi32>
// CHECK-NEXT: scf.if %{{.*}} {
// CHECK-NEXT: vector.transfer_write %[[SHARED]], %arg0[%c0], %[[MASK]]
// CHECK-NEXT: } else {
// CHECK-NEXT: vector.transfer_write %[[SHARED]], %arg1[%c0], %[[MASK]]
// CHECK-NEXT: {{^ *\} *$}}
// CHECK-NEXT: return
// LOWER-LABEL: func.func @issue3_identical_masks_in_branches
// LOWER-NOT: ave.hir.pge
// LOWER: %[[MASK:.*]] = ave.hir.pge <VL32>
// LOWER-NOT: ave.hir.pge
// LOWER: %[[SHARED:.*]] = ave.hir.vadd %{{.*}}, %{{.*}}, %[[MASK]]
// LOWER-NOT: ave.hir.pge
// LOWER: ave.hir.masked_store {{.*}} %arg0[%c0], %[[MASK]], %[[SHARED]]
// LOWER-NOT: ave.hir.pge
// LOWER: ave.hir.masked_store {{.*}} %arg1[%c0], %[[MASK]], %[[SHARED]]
// LOWER-NOT: ave.hir.pge
// LOWER: return
func.func @issue3_identical_masks_in_branches(
    %arg0: memref<32xi32, #hivm.address_space<ub>>,
    %arg1: memref<32xi32, #hivm.address_space<ub>>,
    %cond: i1, %lhs: vector<64xi32>, %rhs: vector<64xi32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %shared = arith.addi %lhs, %rhs : vector<64xi32>
  scf.if %cond {
    %m0 = vector.constant_mask [32] : vector<64xi1>
    vector.transfer_write %shared, %arg0[%c0], %m0 {in_bounds = [true]} : vector<64xi32>, memref<32xi32, #hivm.address_space<ub>>
    scf.yield
  } else {
    %m1 = vector.constant_mask [32] : vector<64xi1>
    vector.transfer_write %shared, %arg1[%c0], %m1 {in_bounds = [true]} : vector<64xi32>, memref<32xi32, #hivm.address_space<ub>>
    scf.yield
  }
  return
}

// -----

// A same-block mask must also precede the arithmetic to avoid cloning it
// during lowering.
// CHECK-LABEL: issue3_identical_masks_same_block
// CHECK-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CHECK-NEXT: annotation.mark %[[MASK]] {mask_op_idx = 0 : i32} : vector<64xi1>
// CHECK: %[[V:.*]] = arith.addi {{.*}} : vector<64xi32>
// CHECK-NEXT: annotation.mark %[[V]] {reached_mask_ops_idx = 0 : i32} keys = ["reached_mask_value"] values = [%[[MASK]] : vector<64xi1>] : vector<64xi32>
// CHECK-NEXT: vector.transfer_write %[[V]], %arg0[%c0], %[[MASK]]
// CHECK-NEXT: vector.transfer_write %[[V]], %arg1[%c0], %[[MASK]]
// CHECK-NEXT: return
// LOWER-LABEL: func.func @issue3_identical_masks_same_block
// LOWER-NOT: ave.hir.pge
// LOWER: %[[MASK:.*]] = ave.hir.pge <VL32>
// LOWER-NOT: ave.hir.pge
// LOWER: %[[SHARED:.*]] = ave.hir.vadd %{{.*}}, %{{.*}}, %[[MASK]]
// LOWER-NOT: ave.hir.pge
// LOWER: ave.hir.masked_store {{.*}} %arg0[%c0], %[[MASK]], %[[SHARED]]
// LOWER-NOT: ave.hir.pge
// LOWER: ave.hir.masked_store {{.*}} %arg1[%c0], %[[MASK]], %[[SHARED]]
// LOWER-NOT: ave.hir.pge
// LOWER: return
func.func @issue3_identical_masks_same_block(
    %arg0: memref<32xi32, #hivm.address_space<ub>>,
    %arg1: memref<32xi32, #hivm.address_space<ub>>,
    %lhs: vector<64xi32>, %rhs: vector<64xi32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %v = arith.addi %lhs, %rhs : vector<64xi32>
  %m0 = vector.constant_mask [32] : vector<64xi1>
  %m1 = vector.constant_mask [32] : vector<64xi1>
  vector.transfer_write %v, %arg0[%c0], %m0 {in_bounds = [true]} : vector<64xi32>, memref<32xi32, #hivm.address_space<ub>>
  vector.transfer_write %v, %arg1[%c0], %m1 {in_bounds = [true]} : vector<64xi32>, memref<32xi32, #hivm.address_space<ub>>
  return
}

// -----

// A dynamic mask whose bound follows the arithmetic cannot be cloned before it.
// CHECK-LABEL: dynamic_mask_late_bound
// CHECK: annotation.mark {{.*}} keys = ["reached_mask_value"]
// LOWER-LABEL: func.func @dynamic_mask_late_bound
// LOWER: %[[ALL:.*]] = ave.hir.pge <ALL>
// LOWER-NEXT: %[[SUM:.*]] = ave.hir.vadd {{.*}}, %[[ALL]]
// LOWER: %[[BOUND:.*]] = arith.addi {{.*}} : index
// LOWER: %[[MASK:.*]], {{.*}} = ave.hir.plt %[[BOUND]]
// LOWER: ave.hir.masked_store {{.*}}, %[[MASK]], %[[SUM]]
func.func @dynamic_mask_late_bound(
    %dst: memref<64xi32, #hivm.address_space<ub>>,
    %lhs: vector<64xi32>, %rhs: vector<64xi32>, %n: index, %step: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %sum = arith.addi %lhs, %rhs : vector<64xi32>
  %bound = arith.addi %n, %step : index
  %mask = vector.create_mask %bound : vector<64xi1>
  vector.maskedstore %dst[%c0], %mask, %sum : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// -----

// Lanes 16-31 are demanded by the store but must come from passthrough.
// CHECK-LABEL: masked_load_narrower_than_store
// CHECK: annotation.mark {{.*}} {reached_mask_ops_idx = {{.*}}}
// LOWER-LABEL: func.func @masked_load_narrower_than_store
// LOWER-SAME: %[[PASS:arg[0-9]+]]: vector<64xi32>
// LOWER: %[[LOAD_MASK:.*]] = ave.hir.pge <VL16>
// LOWER: %[[STORE_MASK:.*]] = ave.hir.pge <VL32>
// LOWER: %[[LOADED:.*]] = ave.hir.vload
// LOWER: %[[SELECTED:.*]] = ave.hir.vsel %[[LOAD_MASK]], %[[LOADED]], %[[PASS]]
// LOWER: ave.hir.masked_store {{.*}}, %[[STORE_MASK]], %[[SELECTED]]
func.func @masked_load_narrower_than_store(
    %src: memref<64xi32, #hivm.address_space<ub>>,
    %dst: memref<64xi32, #hivm.address_space<ub>>, %passthrough: vector<64xi32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %load_mask = vector.constant_mask [16] : vector<64xi1>
  %loaded = vector.maskedload %src[%c0], %load_mask, %passthrough : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
  %store_mask = vector.constant_mask [32] : vector<64xi1>
  vector.maskedstore %dst[%c0], %store_mask, %loaded : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}

// -----

// Positive counterpart to masked_load_narrower_than_store: when the load and
// store use the exact same mask SSA value, inactive passthrough lanes are never
// stored. Keep the existing optimization and omit the select in this case.
// CHECK-LABEL: masked_load_identical_store_mask
// CHECK: annotation.mark {{.*}} {reached_mask_ops_idx = {{.*}}}
// LOWER-LABEL: func.func @masked_load_identical_store_mask
// LOWER: %[[MASK:.*]] = ave.hir.pge <VL32>
// LOWER: %[[LOADED:.*]] = ave.hir.vload
// LOWER-NOT: ave.hir.vsel
// LOWER: ave.hir.masked_store {{.*}}, %[[MASK]], %[[LOADED]]
// LOWER-NOT: ave.hir.vsel
// LOWER: return
func.func @masked_load_identical_store_mask(
    %src: memref<64xi32, #hivm.address_space<ub>>,
    %dst: memref<64xi32, #hivm.address_space<ub>>, %passthrough: vector<64xi32>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %mask = vector.constant_mask [32] : vector<64xi1>
  %loaded = vector.maskedload %src[%c0], %mask, %passthrough : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
  vector.maskedstore %dst[%c0], %mask, %loaded : memref<64xi32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  return
}
