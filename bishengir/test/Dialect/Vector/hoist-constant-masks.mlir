// RUN: bishengir-opt --hoist-constant-masks -split-input-file %s | FileCheck %s --check-prefix=HOIST
// RUN: bishengir-opt --hoist-constant-masks --cse -split-input-file %s | FileCheck %s --check-prefix=CSE
// RUN: bishengir-opt --hoist-constant-masks -split-input-file %s -o %t
// RUN: bishengir-opt --hoist-constant-masks -split-input-file %t | diff %t -

// Hoisting moves both masks; only CSE merges them.
// HOIST-LABEL: func.func @sibling_masks
// HOIST-NEXT: %[[M0:.*]] = vector.constant_mask [32] : vector<64xi1>
// HOIST-NEXT: %[[M1:.*]] = vector.constant_mask [32] : vector<64xi1>
// HOIST-NEXT: %[[RESULT:.*]] = scf.if
// HOIST-NEXT: scf.yield %[[M0]]
// HOIST-NEXT: } else {
// HOIST-NEXT: scf.yield %[[M1]]
// CSE-LABEL: func.func @sibling_masks
// CSE-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CSE-NEXT: %[[RESULT:.*]] = scf.if
// CSE-NEXT: scf.yield %[[MASK]]
// CSE-NEXT: } else {
// CSE-NEXT: scf.yield %[[MASK]]
// CSE-NEXT: }
// CSE-NEXT: return %[[RESULT]]
func.func @sibling_masks(%cond: i1) -> vector<64xi1> {
  %result = scf.if %cond -> vector<64xi1> {
    %m0 = vector.constant_mask [32] : vector<64xi1>
    scf.yield %m0 : vector<64xi1>
  } else {
    %m1 = vector.constant_mask [32] : vector<64xi1>
    scf.yield %m1 : vector<64xi1>
  }
  return %result : vector<64xi1>
}

// -----

// Different contents and types stay distinct, in their original walk order.
// HOIST-LABEL: func.func @different_masks
// HOIST-NEXT: %[[M0:.*]] = vector.constant_mask [32] : vector<64xi1>
// HOIST-NEXT: %[[M1:.*]] = vector.constant_mask [16] : vector<64xi1>
// HOIST-NEXT: %[[M2:.*]] = vector.constant_mask [32] : vector<128xi1>
// HOIST-NEXT: return %[[M0]], %[[M1]], %[[M2]]
// CSE-LABEL: func.func @different_masks
// CSE-NEXT: %[[M0:.*]] = vector.constant_mask [32] : vector<64xi1>
// CSE-NEXT: %[[M1:.*]] = vector.constant_mask [16] : vector<64xi1>
// CSE-NEXT: %[[M2:.*]] = vector.constant_mask [32] : vector<128xi1>
// CSE-NEXT: return %[[M0]], %[[M1]], %[[M2]]
func.func @different_masks() -> (vector<64xi1>, vector<64xi1>, vector<128xi1>) {
  %m0 = vector.constant_mask [32] : vector<64xi1>
  %m1 = vector.constant_mask [16] : vector<64xi1>
  %m2 = vector.constant_mask [32] : vector<128xi1>
  return %m0, %m1, %m2 : vector<64xi1>, vector<64xi1>, vector<128xi1>
}

// -----

// Move a constant mask out of nested control flow while leaving a dynamic mask
// next to the loop induction variable on which it depends.
// HOIST-LABEL: func.func @nested_masks
// HOIST-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// HOIST-NEXT: %c0 = arith.constant 0 : index
// HOIST-NEXT: scf.for %[[IV:.*]] =
// HOIST-NEXT: scf.if
// HOIST-NEXT: vector.maskedstore {{.*}}, %[[MASK]],
// HOIST-NEXT: %[[DYNAMIC:.*]] = vector.create_mask %[[IV]] : vector<64xi1>
// HOIST-NEXT: vector.maskedstore {{.*}}, %[[DYNAMIC]],
// CSE-LABEL: func.func @nested_masks
// CSE-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CSE-NEXT: %c0 = arith.constant 0 : index
// CSE-NEXT: scf.for %[[IV:.*]] =
// CSE-NEXT: scf.if
// CSE-NEXT: vector.maskedstore {{.*}}, %[[MASK]],
// CSE-NEXT: %[[DYNAMIC:.*]] = vector.create_mask %[[IV]] : vector<64xi1>
// CSE-NEXT: vector.maskedstore {{.*}}, %[[DYNAMIC]],
func.func @nested_masks(%dst: memref<64xi32>, %value: vector<64xi32>,
                        %lb: index, %ub: index, %step: index, %cond: i1) {
  %c0 = arith.constant 0 : index
  scf.for %iv = %lb to %ub step %step {
    scf.if %cond {
      %mask = vector.constant_mask [32] : vector<64xi1>
      vector.maskedstore %dst[%c0], %mask, %value : memref<64xi32>, vector<64xi1>, vector<64xi32>
      %dynamic = vector.create_mask %iv : vector<64xi1>
      vector.maskedstore %dst[%c0], %dynamic, %value : memref<64xi32>, vector<64xi1>, vector<64xi32>
    }
  }
  return
}

// -----

// Do not move masks across an IsolatedFromAbove boundary. The outer function
// pass must not collect a mask defined inside the nested module.
// HOIST-LABEL: func.func @isolated
// HOIST-NEXT: builtin.module {
// HOIST-NEXT: func.func @inner()
// HOIST-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// HOIST-NEXT: return %[[MASK]]
// CSE-LABEL: func.func @isolated
// CSE-NEXT: builtin.module {
// CSE-NEXT: func.func @inner()
// CSE-NEXT: %[[MASK:.*]] = vector.constant_mask [32] : vector<64xi1>
// CSE-NEXT: return %[[MASK]]
func.func @isolated() {
  builtin.module {
    func.func @inner() -> vector<64xi1> {
      %mask = vector.constant_mask [32] : vector<64xi1>
      return %mask : vector<64xi1>
    }
  }
  return
}

// -----

// HOIST-LABEL: func.func private @external()
// CSE-LABEL: func.func private @external()
func.func private @external()
