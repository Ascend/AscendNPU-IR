// RUN: bishengir-opt %s \
// RUN:   --hacc-append-device-spec=target=Ascend960DT_968B7 \
// RUN:   --hivm-vectorize=phase=backend --verify-each \
// RUN:   | FileCheck %s --implicit-check-not=arith.extui

// The backend phase legalizes vectorizer-specific forms: boolean predicate
// extension becomes a select against 1/0, and ordered floating predicates
// become the AVE predicate combination.

module {
  func.func @predicate(%p: vector<64xi1>) -> vector<64xi8>
      attributes {hivm.vector_function} {
    %r = arith.extui %p : vector<64xi1> to vector<64xi8>
    return %r : vector<64xi8>
  }

  func.func @ordered(%a: vector<64xf32>, %b: vector<64xf32>) -> vector<64xi1>
      attributes {hivm.vector_function} {
    %r = arith.cmpf ord, %a, %b : vector<64xf32>
    return %r : vector<64xi1>
  }
}

// CHECK-LABEL: func.func @predicate(
// CHECK: %[[YES:.*]] = arith.constant dense<1> : vector<64xi8>
// CHECK: %[[NO:.*]] = arith.constant dense<0> : vector<64xi8>
// CHECK: arith.select %{{.*}}, %[[YES]], %[[NO]] : vector<64xi1>, vector<64xi8>

// CHECK-LABEL: func.func @ordered(
// CHECK: arith.cmpf oeq, %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: arith.cmpf oeq, %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: ave.hir.pge <ALL> : vector<64xi1>
// CHECK: ave.hir.preg.and <b32>
