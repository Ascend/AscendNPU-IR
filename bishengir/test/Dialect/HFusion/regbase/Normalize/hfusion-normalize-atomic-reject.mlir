// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file -verify-diagnostics

// Checks if hfusion atomic op rejected by hfusion-normalize-ops causes error
//
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_normalize_reject_dyn_root(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>) {
    // expected-error @+1 {{software-atomic store (kind=and) survived hfusion-normalize-ops}}
    hfusion.store {atomic_kind = #hfusion.atomic_kind<and>} ins(%arg1 : memref<?x?xi32>) outs(%arg0 : memref<?x?xi32>)
    return
  }
}
