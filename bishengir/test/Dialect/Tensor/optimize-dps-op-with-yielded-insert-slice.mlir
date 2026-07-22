// RUN: bishengir-opt -allow-unregistered-dialect -optimize-dps-op-with-yielded-insert-slice %s | FileCheck %s
// RUN: bishengir-opt -allow-unregistered-dialect \
// RUN:   -optimize-dps-op-with-yielded-insert-slice \
// RUN:   -one-shot-bufferize=allow-unknown-ops -cse -canonicalize %s | FileCheck %s -check-prefix=CHECK-ONE-SHOT

func.func @optimize_dps_inits(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>)
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    // CHECK: %[[OFFSET:.*]] = "some_calculation"
    %offset = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    %empty = tensor.empty() : tensor<16xf32>
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG:.*]][%[[OFFSET]]] [16] [1] : tensor<64xf32> to tensor<16xf32>
    // CHECK: linalg.add ins({{.*}}) outs(%[[SLICE]] : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: linalg.add
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT-NOT: memref.copy
    // CHECK: tensor.insert_slice {{.*}} into %[[ARG]][%[[OFFSET]]] [16] [1] : tensor<16xf32> into tensor<64xf32>
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}

// -----

func.func @dominance_issue(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>) 
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    %value = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    // CHECK-NOT: tensor.extract_slice
    %empty = tensor.empty() : tensor<16xf32>
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: memref.copy
    %offset = "some_calculation"(%value) : (index) -> (index)
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}

// -----

func.func @init_not_empty(%lb: index, %ub: index, %step: index) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %init = "some_value"() : () -> (tensor<64xf32>)
  %add_src0 = "some_value"() : () -> (tensor<16xf32>)
  %add_src1 = "some_value"() : () -> (tensor<16xf32>) 
  %add_dst = "some_value"() : () -> (tensor<16xf32>) 
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<64xf32>) {
    %value = "some_calculation"(%lb, %ub, %step) : (index, index, index) -> (index)
    // CHECK-NOT: tensor.extract_slice
    %1 = linalg.add ins(%add_src0, %add_src1 : tensor<16xf32>, tensor<16xf32>) outs(%add_dst : tensor<16xf32>) -> tensor<16xf32>
    // CHECK-ONE-SHOT: memref.copy
    %offset = "some_calculation"(%value) : (index) -> (index)
    %inserted_slice = tensor.insert_slice %1 into %arg1[%offset] [16] [1] : tensor<16xf32> into tensor<64xf32>
    scf.yield %inserted_slice : tensor<64xf32>
  }
  return %res : tensor<64xf32>
}

// -----

func.func @load_problem(%lb: index, %ub: index, %step: index) -> tensor<18x111x3xf32> {
  %init = "some_value"() : () -> (tensor<18x111x3xf32>)
  %load_src0 = "some_value"() : () -> (tensor<111x3xf32>)
  %load_dst = tensor.empty() : tensor<111x3xf32>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<18x111x3xf32>) {
    %value = "some_calculation"(%arg0, %lb, %ub, %step) : (index, index, index, index) -> (index)

    // CHECK: tensor.extract_slice
    // CHECK: tensor<18x111x3xf32> to tensor<111x3xf32>
    // CHECK-NOT: tensor<18x111x3xf32> to tensor<1x111x3xf32>
    %16 = hivm.hir.load ins(%load_src0 : tensor<111x3xf32>) outs(%load_dst : tensor<111x3xf32>) {"inserted-load"} core_type = <VECTOR> -> tensor<111x3xf32>
    %inserted_slice = tensor.insert_slice %16 into %arg1[%value, 0, 0] [1, 111, 3] [1, 1, 1] : tensor<111x3xf32> into tensor<18x111x3xf32>
    scf.yield %inserted_slice : tensor<18x111x3xf32>
  }
  return %res : tensor<18x111x3xf32>
}

// -----

// CHECK-LABEL: func.func @hoist_alloc_basic(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @hoist_alloc_basic(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG:.*]] = %{{.*}})
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[BIG_ALLOC]]
    // CHECK: linalg.fill ins({{.*}}) outs(%[[SUBVIEW]]
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG]]
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR]]
  return %res : tensor<128x128xbf16>
}

// -----

// CHECK-LABEL: func.func @hoist_alloc_two_iter_args(
func.func @hoist_alloc_two_iter_args(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC0:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR0:.*]] = bufferization.to_tensor %[[BIG_ALLOC0]]
  // CHECK: %[[BIG_ALLOC1:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR1:.*]] = bufferization.to_tensor %[[BIG_ALLOC1]]
  %res:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %init) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc0 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc0 : memref<32x128xbf16>)
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x128xbf16>
    %inserted0 = tensor.insert_slice %t0 into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    %alloc1 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc1 : memref<32x128xbf16>)
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x128xbf16>
    %inserted1 = tensor.insert_slice %t1 into %arg2[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG0:.*]], %[[ARG1:.*]]
    scf.yield %inserted0, %inserted1 : tensor<128x128xbf16>, tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR0]], %[[BIG_TENSOR1]]
  return %res#0, %res#1 : tensor<128x128xbf16>, tensor<128x128xbf16>
}

// -----

// CHECK-LABEL: func.func @hoist_alloc_with_subview(
func.func @hoist_alloc_with_subview(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    // Load uses a subview of the alloc (not the alloc directly)
    %subview = memref.subview %alloc[0, 0] [32, 128] [1, 1] : memref<32x128xbf16> to memref<32x128xbf16>
    // CHECK: linalg.fill ins({{.*}}) outs(%{{.*}}
    linalg.fill ins(%cst : bf16) outs(%subview : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %{{.*}}
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR]]
  return %res : tensor<128x128xbf16>
}

// -----

// CHECK-LABEL: func.func @hoist_alloc_with_memspace_cast(
func.func @hoist_alloc_with_memspace_cast(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16, 5>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[BIG_ALLOC]]
  // CHECK-SAME: memref<128x128xbf16, 5> to memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[CAST]]
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16, 5>
    %cast = memref.memory_space_cast %alloc : memref<32x128xbf16, 5> to memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%cast : memref<32x128xbf16>)
    %t = bufferization.to_tensor %cast restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %{{.*}}
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR]]
  return %res : tensor<128x128xbf16>
}

// -----

// Verify that the pattern does NOT fire when there is no load user of the alloc.
// CHECK-LABEL: func.func @no_load_user_no_hoist(
func.func @no_load_user_no_hoist(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    // Alloc is only used by to_tensor, not by any DPS op.
    %alloc = memref.alloc() : memref<32x128xbf16>
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %inserted : tensor<128x128xbf16>
  }
  return %res : tensor<128x128xbf16>
}

// -----

// Verify the pattern does NOT fire when the insert_slice dest is not
// the iter_arg. The yield is an identity passthrough to keep the scf.for
// valid for one-shot-bufferize.
// CHECK-LABEL: func.func @wrong_dest_no_hoist(
func.func @wrong_dest_no_hoist(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    // insert_slice into a tensor computed from iter_arg (not directly the iter_arg)
    %extracted = tensor.extract_slice %arg1[%offset, 0] [32, 128] [1, 1] : tensor<128x128xbf16> to tensor<32x128xbf16>
    %inserted = tensor.insert_slice %t into %extracted[0, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<32x128xbf16>
    // The dest of insert_slice is %extracted (not the iter_arg), so HoistAlloc won't fire.
    // Meanwhile, yield is %arg1 (identity), compatible with one-shot-bufferize.
    scf.yield %arg1 : tensor<128x128xbf16>
  }
  return %res : tensor<128x128xbf16>
}

// -----

// Verify mixed iter_args: only tensor ones are hoisted, non-tensor iter_args
// are left unchanged.
// CHECK-LABEL: func.func @mixed_iter_args(
func.func @mixed_iter_args(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, i32) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  %res:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %c0_i32) -> (tensor<128x128xbf16>, i32) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    %next = arith.addi %arg2, %c0_i32 : i32
    // CHECK: scf.yield %[[ARG0:.*]], %{{.*}}
    scf.yield %inserted, %next : tensor<128x128xbf16>, i32
  }
  // CHECK: return %[[BIG_TENSOR]], %{{.*}}
  return %res#0, %res#1 : tensor<128x128xbf16>, i32
}

// -----

// Verify that when the iter_arg init is a scalar vbrc (fill zero),
// the big alloc also gets a vbrc to preserve the initialization semantic.
// CHECK-LABEL: func.func @hoist_alloc_with_vbrc_init(
func.func @hoist_alloc_with_vbrc_init(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %filled = hivm.hir.vbrc ins(%cst : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : bf16) outs(%[[BIG_ALLOC]]
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  // CHECK: scf.for
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %filled) -> (tensor<128x128xbf16>) {
    // CHECK: %[[SV:.*]] = memref.subview %[[BIG_ALLOC]]
    // CHECK: annotation.mark %[[BIG_ALLOC]] {hivm.slice_load} {{.*}}values = [%[[SV]]
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %{{.*}}
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR]]
  return %res : tensor<128x128xbf16>
}

// -----

// Verify the pattern does NOT fire when the alloc has a non-load user
// inside the loop (e.g. func.call). Redirecting it would break the IR.
// CHECK-LABEL: func.func @func_call_user_no_hoist_2(
func.func private @some_kernel_2(%m: memref<32x128xbf16>)
func.func @func_call_user_no_hoist_2(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    // Non-load user: func.call -- redirecting this would break the call.
    func.call @some_kernel_2(%alloc) : (memref<32x128xbf16>) -> ()
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %inserted : tensor<128x128xbf16>
  }
  return %res : tensor<128x128xbf16>
}

// -----

// Verify the pattern does NOT fire when the iter_arg init carries
// external data (e.g. a func.call result). The init is not a fresh
// empty/vbrc/to_tensor buffer, so the pass skips to avoid discarding
// the external data.
// CHECK-LABEL: func.func @init_with_call_result_no_hoist(
// CHECK-NOT: memref.alloc() : memref<128x128xbf16>
func.func private @init_producer() -> tensor<128x128xbf16>
func.func @init_with_call_result_no_hoist(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = call @init_producer() : () -> tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %inserted : tensor<128x128xbf16>
  }
  return %res : tensor<128x128xbf16>
}

// -----

// Two chained for loops: for2's init is for1's result. After greedy
// convergence (for1 hoisted first, for2 then sees a to_tensor init),
// both loops share the same big alloc.
// CHECK-LABEL: func.func @two_chained_fors_share_alloc(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @two_chained_fors_share_alloc(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG1:.*]] = %{{.*}})
  %res1 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG1]]
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG2:.*]] =
  // CHECK-SAME:   %[[BIG_TENSOR]]
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[SV2:.*]] = memref.subview %[[BIG_ALLOC]]
  %res2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %res1) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG2]]
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR]]
  return %res2 : tensor<128x128xbf16>
}
