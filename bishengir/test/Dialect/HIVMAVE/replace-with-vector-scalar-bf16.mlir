// RUN: bishengir-opt %s -split-input-file -ave-replace-with-vector-scalar | FileCheck %s

// CHECK-LABEL: test_fold_broadcast_into_vadds_bf16
func.func @test_fold_broadcast_into_vadds_bf16(%vec : vector<128xbf16>,
                                               %s : bf16) {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %brc = ave.hir.scalar_broadcast %s : bf16 -> vector<128xbf16>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vadds {{.*}} : vector<128xbf16>, bf16, vector<128xi1>
  %0 = ave.hir.vadd %vec, %brc, %mask : vector<128xbf16>, vector<128xi1>
  "test.test"(%0) : (vector<128xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: test_keep_broadcast_for_vmul_bf16
func.func @test_keep_broadcast_for_vmul_bf16(%vec : vector<128xbf16>,
                                                     %s : bf16) {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %brc = ave.hir.scalar_broadcast %s : bf16 -> vector<128xbf16>
  // CHECK: %[[BRC:.*]] = ave.hir.scalar_broadcast
  // CHECK: %[[MUL:.*]] = ave.hir.vmul {{.*}}, %[[BRC]], {{.*}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vmuls
  // CHECK: "test.test"(%[[MUL]])
  %0 = ave.hir.vmul %vec, %brc, %mask : vector<128xbf16>, vector<128xi1>
  "test.test"(%0) : (vector<128xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: test_broadcast_folded_into_vmuls_f16
func.func @test_broadcast_folded_into_vmuls_f16(%vec : vector<128xf16>,
                                                %s : f16) {
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  %brc = ave.hir.scalar_broadcast %s : f16 -> vector<128xf16>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmuls {{.*}} : vector<128xf16>, f16, vector<128xi1>
  %0 = ave.hir.vmul %vec, %brc, %mask : vector<128xf16>, vector<128xi1>
  "test.test"(%0) : (vector<128xf16>) -> ()
  return
}
