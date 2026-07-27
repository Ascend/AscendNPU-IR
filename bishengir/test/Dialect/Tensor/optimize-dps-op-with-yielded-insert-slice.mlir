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
    %16 = hivm.hir.load ins(%load_src0 : tensor<111x3xf32>) outs(%load_dst : tensor<111x3xf32>) {"hivm.inserted-load"} core_type = <VECTOR> -> tensor<111x3xf32>
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

// Non-contiguous tensor iter_args at positions 0 and 2, with an i32 at
// position 1. Both tensor args should be independently hoisted.
// CHECK-LABEL: func.func @non_contiguous_tensor_args_pos_0_2(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @non_contiguous_tensor_args_pos_0_2(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, i32, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[BIG_ALLOC0:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR0:.*]] = bufferization.to_tensor %[[BIG_ALLOC0]]
  // CHECK: %[[BIG_ALLOC2:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR2:.*]] = bufferization.to_tensor %[[BIG_ALLOC2]]
  // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG0:.*]] = %{{.*}}, %[[ARG1:.*]] = %{{.*}}, %[[ARG2:.*]] = %{{.*}})
  %res:3 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %c0_i32, %arg3 = %init) -> (tensor<128x128xbf16>, i32, tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    // Tensor at pos 0
    %alloc0 = memref.alloc() : memref<32x128xbf16>
    // CHECK: %[[SV0:.*]] = memref.subview %[[BIG_ALLOC0]]
    linalg.fill ins(%cst : bf16) outs(%alloc0 : memref<32x128xbf16>)
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x128xbf16>
    %inserted0 = tensor.insert_slice %t0 into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // Non-tensor at pos 1 (i32 counter)
    %next = arith.addi %arg2, %c0_i32 : i32
    // Tensor at pos 2
    %alloc2 = memref.alloc() : memref<32x128xbf16>
    // CHECK: %[[SV2:.*]] = memref.subview %[[BIG_ALLOC2]]
    linalg.fill ins(%cst : bf16) outs(%alloc2 : memref<32x128xbf16>)
    %t2 = bufferization.to_tensor %alloc2 restrict writable : memref<32x128xbf16>
    %inserted2 = tensor.insert_slice %t2 into %arg3[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG0]], %{{.*}}, %[[ARG2]]
    scf.yield %inserted0, %next, %inserted2 : tensor<128x128xbf16>, i32, tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR0]], %{{.*}}, %[[BIG_TENSOR2]]
  // CHECK-ONE-SHOT-LABEL: func.func @non_contiguous_tensor_args_pos_0_2(
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res#0, %res#1, %res#2 : tensor<128x128xbf16>, i32, tensor<128x128xbf16>
}

// -----

// Non-contiguous tensor iter_args with mixed init types: pos 0 is empty,
// pos 1 is i32, pos 2 is scalar vbrc. The vbrc init semantics must be
// preserved on the big alloc at pos 2.
// CHECK-LABEL: func.func @non_contiguous_mixed_init(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @non_contiguous_mixed_init(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, i32, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %c0_i32 = arith.constant 0 : i32
  %vbrc_init = hivm.hir.vbrc ins(%cst : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  // CHECK: %[[BIG_ALLOC0:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR0:.*]] = bufferization.to_tensor %[[BIG_ALLOC0]]
  // CHECK: %[[BIG_ALLOC2:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : bf16) outs(%[[BIG_ALLOC2]]
  // CHECK: %[[BIG_TENSOR2:.*]] = bufferization.to_tensor %[[BIG_ALLOC2]]
  %res:3 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %c0_i32, %arg3 = %vbrc_init) -> (tensor<128x128xbf16>, i32, tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    // Tensor pos 0 (empty init)
    %alloc0 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc0 : memref<32x128xbf16>)
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x128xbf16>
    %inserted0 = tensor.insert_slice %t0 into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // Non-tensor pos 1
    %next = arith.addi %arg2, %c0_i32 : i32
    // Tensor pos 2 (vbrc init)
    %alloc2 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc2 : memref<32x128xbf16>)
    %t2 = bufferization.to_tensor %alloc2 restrict writable : memref<32x128xbf16>
    %inserted2 = tensor.insert_slice %t2 into %arg3[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG0:.*]], %{{.*}}, %[[ARG2:.*]]
    scf.yield %inserted0, %next, %inserted2 : tensor<128x128xbf16>, i32, tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR0]], %{{.*}}, %[[BIG_TENSOR2]]
  // CHECK-ONE-SHOT-LABEL: func.func @non_contiguous_mixed_init(
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res#0, %res#1, %res#2 : tensor<128x128xbf16>, i32, tensor<128x128xbf16>
}

// -----

// Negative: the iter_arg init is a function block argument (not
// empty/vbrc/to_tensor). The pass must skip this iter_arg, and since
// it is the only tensor iter_arg, infos is empty → no hoisting.
// CHECK-LABEL: func.func @init_from_block_arg_no_hoist(
func.func @init_from_block_arg_no_hoist(%init_tensor: tensor<128x128xbf16>, %lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init_tensor) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT: memref.copy
  return %res : tensor<128x128xbf16>
}

// -----

// Positive: the hoisted forOp result is consumed by an scf.if. After
// replacement, the scf.if should pass through the big to_tensor result
// without requiring a memref.copy.
// CHECK-LABEL: func.func @result_used_in_scf_if(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[COND:.*]]: i1)
func.func @result_used_in_scf_if(%lb: index, %ub: index, %step: index, %cond: i1) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  %res_for = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    // CHECK: %[[SV:.*]] = memref.subview %[[BIG_ALLOC]]
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG:.*]]
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // The hoisted forOp result is consumed by scf.if — both branches
  // should now pass through %big_tensor (replaced by replaceAllUsesWith).
  %res = scf.if %cond -> tensor<128x128xbf16> {
    scf.yield %res_for : tensor<128x128xbf16>
  } else {
    scf.yield %res_for : tensor<128x128xbf16>
  }
  // CHECK: return %{{.*}}
  // CHECK-ONE-SHOT-LABEL: func.func @result_used_in_scf_if(
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res : tensor<128x128xbf16>
}

// -----

// Negative: the scf.yield operand is a block argument (the iter_arg itself),
// not a direct tensor.insert_slice. The pattern requires the yield value to
// be defined by an insert_slice op whose dest is the iter_arg, so it bails out.
// CHECK-LABEL: func.func @yield_not_from_insert_slice_no_hoist(
func.func @yield_not_from_insert_slice_no_hoist(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    // insert_slice result is computed but discarded — the for yields the
    // iter_arg directly. yieldVal is a BlockArgument, not an InsertSliceOp.
    %updated = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %arg1 : tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res : tensor<128x128xbf16>
}

// -----

// Negative: the DPS store user (linalg.fill) is nested inside an scf.if
// within the for loop. hasBufferStoreUserInLoop only sees users directly
// in the forOp body, so the alloc appears to have no store user → bail out.
// CHECK-LABEL: func.func @dps_user_nested_in_scf_if_no_hoist(
func.func @dps_user_nested_in_scf_if_no_hoist(%lb: index, %ub: index, %step: index, %cond: i1) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    %alloc = memref.alloc() : memref<32x128xbf16>
    // The fill is inside scf.if — its parent op is scf.if, not the forOp.
    // hasBufferStoreUserInLoop skips it, so the pattern bails out.
    scf.if %cond {
      linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    }
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %inserted : tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT: memref.copy
  return %res : tensor<128x128xbf16>
}

// -----

// Positive: single tensor iter_arg at position 1 among four non-tensor
// iter_args. Verifies the pass correctly finds and hoists the sole tensor
// arg regardless of its position.
// CHECK-LABEL: func.func @single_tensor_among_non_tensor_args(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @single_tensor_among_non_tensor_args(%lb: index, %ub: index, %step: index) -> (i32, tensor<128x128xbf16>, i32, f32) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %c0_i32 = arith.constant 0 : i32
  %c0_f32 = arith.constant 0.0 : f32
  // CHECK: %[[BIG_ALLOC:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: %[[BIG_TENSOR:.*]] = bufferization.to_tensor %[[BIG_ALLOC]]
  // CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[A0:.*]] = %{{.*}}, %[[A1:.*]] = %{{.*}}, %[[A2:.*]] = %{{.*}}, %[[A3:.*]] = %{{.*}})
  %res:4 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %c0_i32, %arg2 = %init, %arg3 = %c0_i32, %arg4 = %c0_f32) -> (i32, tensor<128x128xbf16>, i32, f32) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    // Non-tensor pos 0
    %v0 = arith.addi %arg1, %c0_i32 : i32
    // Tensor pos 1 — the only tensor to hoist
    %alloc = memref.alloc() : memref<32x128xbf16>
    // CHECK: %[[SV:.*]] = memref.subview %[[BIG_ALLOC]]
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %inserted = tensor.insert_slice %t into %arg2[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // Non-tensor pos 2, 3
    %v2 = arith.addi %arg3, %c0_i32 : i32
    %v3 = arith.addf %arg4, %c0_f32 : f32
    // CHECK: scf.yield %{{.*}}, %[[A1]], %{{.*}}, %{{.*}}
    scf.yield %v0, %inserted, %v2, %v3 : i32, tensor<128x128xbf16>, i32, f32
  }
  // CHECK: return %{{.*}}, %[[BIG_TENSOR]], %{{.*}}, %{{.*}}
  // CHECK-ONE-SHOT-LABEL: func.func @single_tensor_among_non_tensor_args(
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res#0, %res#1, %res#2, %res#3 : i32, tensor<128x128xbf16>, i32, f32
}

// -----

// Positive: two tensor iter_args with different init types — pos 0 is
// tensor.empty, pos 1 is scalar vbrc. Both are hoisted, with the vbrc
// init replicated on the corresponding big alloc.
// CHECK-LABEL: func.func @all_tensor_mixed_init_empty_and_vbrc(
//  CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
func.func @all_tensor_mixed_init_empty_and_vbrc(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %vbrc_init = hivm.hir.vbrc ins(%cst : bf16) outs(%init : tensor<128x128xbf16>) -> tensor<128x128xbf16>
  // CHECK: %[[BIG_ALLOC0:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK-NOT: hivm.hir.vbrc ins({{.*}}) outs(%[[BIG_ALLOC0]]
  // CHECK: %[[BIG_TENSOR0:.*]] = bufferization.to_tensor %[[BIG_ALLOC0]]
  // CHECK: %[[BIG_ALLOC1:.*]] = memref.alloc() : memref<128x128xbf16>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : bf16) outs(%[[BIG_ALLOC1]]
  // CHECK: %[[BIG_TENSOR1:.*]] = bufferization.to_tensor %[[BIG_ALLOC1]]
  %res:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %vbrc_init) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
    %offset = "some_calculation"(%arg0) : (index) -> (index)
    // Tensor pos 0 (empty init → no vbrc on big alloc)
    %alloc0 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc0 : memref<32x128xbf16>)
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x128xbf16>
    %inserted0 = tensor.insert_slice %t0 into %arg1[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // Tensor pos 1 (vbrc init → vbrc replicated on big alloc)
    %alloc1 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc1 : memref<32x128xbf16>)
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x128xbf16>
    %inserted1 = tensor.insert_slice %t1 into %arg2[%offset, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    // CHECK: scf.yield %[[ARG0:.*]], %[[ARG1:.*]]
    scf.yield %inserted0, %inserted1 : tensor<128x128xbf16>, tensor<128x128xbf16>
  }
  // CHECK: return %[[BIG_TENSOR0]], %[[BIG_TENSOR1]]
  // CHECK-ONE-SHOT-LABEL: func.func @all_tensor_mixed_init_empty_and_vbrc(
  // CHECK-ONE-SHOT-NOT: memref.copy
  return %res#0, %res#1 : tensor<128x128xbf16>, tensor<128x128xbf16>
}

// -----

// Negative: no tensor iter_args at all. The pass should not fire.
// CHECK-LABEL: func.func @no_tensor_iter_args_no_match(
func.func @no_tensor_iter_args_no_match(%lb: index, %ub: index, %step: index) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: memref.alloc()
  // CHECK: scf.for
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %c0_i32) -> (i32) {
    %next = arith.addi %arg1, %c0_i32 : i32
    scf.yield %next : i32
  }
  return %res : i32
}

// -----

// Negative: one alloc feeds two to_tensor / insert_slice pairs that write
// to two different iter_args. The first replaceAllUsesWith would redirect
// the memref to the first subview, starving the second big alloc.
// Detected by both hasUnexpectedUserInLoop (toTensorCount>1) and the
// shared-memcast check (existing.memcast == toTensorOp.getMemref()).
// CHECK-LABEL: func.func @one_alloc_multi_iter_arg_no_hoist(
func.func @one_alloc_multi_iter_arg_no_hoist(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %init) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
    %offset0 = "some_calculation"(%arg0) : (index) -> (index)
    %offset1 = "some_calculation"(%offset0) : (index) -> (index)
    // Single alloc → two to_tensor → two different insert_slices.
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t0 = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %u0 = tensor.insert_slice %t0 into %arg1[%offset0, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    %t1 = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %u1 = tensor.insert_slice %t1 into %arg2[%offset1, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %u0, %u1 : tensor<128x128xbf16>, tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT: memref.copy
  return %res#0, %res#1 : tensor<128x128xbf16>, tensor<128x128xbf16>
}

// -----

// Negative: a single to_tensor (post-CSE) feeds two insert_slices that
// write to different iter_args. The shared-memcast check catches this
// because both infos carry the same toTensorOp.getMemref().
// CHECK-LABEL: func.func @one_to_tensor_multi_insert_slice_no_hoist(
func.func @one_to_tensor_multi_insert_slice_no_hoist(%lb: index, %ub: index, %step: index) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res:2 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init, %arg2 = %init) -> (tensor<128x128xbf16>, tensor<128x128xbf16>) {
    %offset0 = "some_calculation"(%arg0) : (index) -> (index)
    %offset1 = "some_calculation"(%offset0) : (index) -> (index)
    // Single alloc → single to_tensor → two different insert_slices.
    %alloc = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x128xbf16>)
    %t = bufferization.to_tensor %alloc restrict writable : memref<32x128xbf16>
    %u0 = tensor.insert_slice %t into %arg1[%offset0, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    %u1 = tensor.insert_slice %t into %arg2[%offset1, 0] [32, 128] [1, 1] : tensor<32x128xbf16> into tensor<128x128xbf16>
    scf.yield %u0, %u1 : tensor<128x128xbf16>, tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT: memref.copy
  return %res#0, %res#1 : tensor<128x128xbf16>, tensor<128x128xbf16>
}

// -----

// Negative: two chained insert_slices in one iteration. The second
// insert_slice writes into the first insert_slice's result, not into
// the iter_arg directly. Since the yield insert_slice's dest is not
// the iter_arg, the pattern bails out.
// CHECK-LABEL: func.func @two_chained_insert_slices_no_hoist(
func.func @two_chained_insert_slices_no_hoist(%lb: index, %ub: index, %step: index) -> tensor<128x128xbf16> {
  %init = tensor.empty() : tensor<128x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK-NOT: memref.alloc() : memref<128x128xbf16>
  %res = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %init) -> (tensor<128x128xbf16>) {
    %offset1 = "some_calculation"(%arg0) : (index) -> (index)
    %offset2 = "some_calculation"(%offset1) : (index) -> (index)
    // First insert_slice: writes into iter_arg (this one is fine)
    %alloc1 = memref.alloc() : memref<16x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc1 : memref<16x128xbf16>)
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<16x128xbf16>
    %updated = tensor.insert_slice %t1 into %arg1[%offset1, 0] [16, 128] [1, 1] : tensor<16x128xbf16> into tensor<128x128xbf16>
    // Second insert_slice: writes into %updated (not %arg1)
    // The yield uses this one — dest is not the iter_arg → bail out.
    %alloc2 = memref.alloc() : memref<16x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc2 : memref<16x128xbf16>)
    %t2 = bufferization.to_tensor %alloc2 restrict writable : memref<16x128xbf16>
    %updated2 = tensor.insert_slice %t2 into %updated[%offset2, 0] [16, 128] [1, 1] : tensor<16x128xbf16> into tensor<128x128xbf16>
    scf.yield %updated2 : tensor<128x128xbf16>
  }
  // CHECK-ONE-SHOT: memref.copy
  return %res : tensor<128x128xbf16>
}
