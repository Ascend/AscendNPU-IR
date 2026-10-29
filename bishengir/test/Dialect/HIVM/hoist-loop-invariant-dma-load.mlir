// RUN: env BISHENGIR_DISABLE_HOIST_INVARIANT_DMA_LOAD=1 bishengir-opt %s -hivm-hoist-loop-invariant-dma-load -split-input-file | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt %s -hivm-hoist-loop-invariant-dma-load -split-input-file | FileCheck %s

// A masked tl.load materializes as alloc + guarded fill scf.if + hivm.hir.load
// + to_tensor. When every operand is defined outside the inner loop, the whole
// cluster moves before the inner loop.
// DISABLED-LABEL: func.func @invariant_load
// DISABLED: scf.for
// DISABLED: scf.for
// DISABLED: memref.alloc
// DISABLED: hivm.hir.load

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @invariant_load
  // CHECK: %[[SUB:.*]] = memref.subview %{{.*}}[0, 0] [2, 128]
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: scf.if %{{.+}} {
  // CHECK: func.call @fill(%[[ALLOC]])
  // CHECK: %[[DST:.*]] = memref.subview %[[ALLOC]]
  // CHECK: hivm.hir.load ins(%[[SUB]]
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: scf.for
  // CHECK: func.call @compute(%[[T]])
  // CHECK-NOT: hivm.hir.load
  func.func @invariant_load(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      %sub = memref.subview %src[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          func.call @fill(%alloc) : (memref<2x128xf16>) -> ()
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%sub : memref<2x128xf16, strided<[128, 1]>>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @fill(memref<2x128xf16>)
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The dst view size depends on the inner induction variable: the cluster
// stays inside the loop.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @dependent_load
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @dependent_load(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        %jc = arith.index_cast %j : i32 to index
        %one = arith.constant 1 : index
        %size = arith.addi %jc, %one : index
        %dst = memref.subview %alloc[0, 0] [%size, 128] [1, 1] : memref<2x128xf16> to memref<?x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<?x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// An in-loop call may write the loaded GM source: the cluster stays.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @src_written
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @src_written(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          func.call @fill(%alloc) : (memref<2x128xf16>) -> ()
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @write(%src) : (memref<2x128xf16>) -> ()
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @fill(memref<2x128xf16>)
  func.func private @write(memref<2x128xf16>)
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// Another in-loop user of the dst buffer: the cluster stays.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @other_dst_user
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @other_dst_user(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          func.call @fill(%alloc) : (memref<2x128xf16>) -> ()
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @peek(%dst) : (memref<2x128xf16, strided<[128, 1]>>) -> ()
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @fill(memref<2x128xf16>)
  func.func private @peek(memref<2x128xf16, strided<[128, 1]>>)
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The load's pad_value depends on the inner induction variable: the cluster
// stays inside the loop.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @dependent_pad
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @dependent_pad(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        %jv = arith.sitofp %j : i32 to f16
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %jv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The load's pad_value is an in-loop constant: the constant is collected
// into the cluster and hoisted with it.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @inloop_constant_pad
  // CHECK: %[[PV:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: %[[DST:.*]] = memref.subview %[[ALLOC]]
  // CHECK: hivm.hir.load ins(%{{.*}} outs(%[[DST]] : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %[[PV]]
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: scf.for
  // CHECK: func.call @compute(%[[T]])
  // CHECK-NOT: hivm.hir.load
  func.func @inloop_constant_pad(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %pv = arith.constant 0.0 : f16
        %alloc = memref.alloc() : memref<2x128xf16>
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %pv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The guarded fill takes a scalar derived from the inner induction variable:
// the cluster stays inside the loop.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @fill_scalar_dependent
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @fill_scalar_dependent(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %jv = arith.sitofp %j : i32 to f16
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          func.call @fill_pad(%alloc, %jv) : (memref<2x128xf16>, f16) -> ()
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @fill_pad(memref<2x128xf16>, f16)
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The fill guard's condition is computed inside the loop from the cluster's
// own size math (the mask shape before the early passes lift it): the
// condition chain is absorbed into the cluster and the whole thing hoists.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @inloop_fill_condition
  // CHECK: %[[SZ:.*]] = arith.minsi
  // CHECK: %[[MASK:.*]] = arith.cmpi slt
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: scf.if %[[MASK]]
  // CHECK: linalg.fill
  // CHECK: %[[DST:.*]] = memref.subview %[[ALLOC]]
  // CHECK: hivm.hir.load ins(%{{.*}} outs(%[[DST]] : memref<?x128xf16, strided<[128, 1]>>)
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: scf.for
  // CHECK: func.call @compute(%[[T]])
  // CHECK-NOT: hivm.hir.load
  func.func @inloop_fill_condition(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %sz = arith.minsi %c7, %c2 : index
        %mask = arith.cmpi slt, %sz, %c2 : index
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %mask {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<2x128xf16>)
        } {hivm.unlikely_condition}
        %dst = memref.subview %alloc[0, 0] [%sz, 128] [1, 1] : memref<2x128xf16> to memref<?x128xf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<?x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// The dst chain goes through a hivm.hir.bitcast (same buffer, reinterpreted
// element type): the fill is still recognized as a dst user and the whole
// cluster hoists.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @bitcast_dst
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: %[[BC:.*]] = hivm.hir.bitcast %[[ALLOC]]
  // CHECK: scf.if
  // CHECK: func.call @fill(%[[BC]])
  // CHECK: %[[DST:.*]] = memref.subview %[[BC]]
  // CHECK: hivm.hir.load ins(%{{.*}} outs(%[[DST]] : memref<2x128xbf16, strided<[128, 1]>>)
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[BC]]
  // CHECK: scf.for
  // CHECK: func.call @compute_bf(%[[T]])
  // CHECK-NOT: hivm.hir.load
  func.func @bitcast_dst(%src: memref<2x128xbf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : bf16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        %bc = hivm.hir.bitcast %alloc : memref<2x128xf16> -> memref<2x128xbf16>
        scf.if %cond {
          func.call @fill(%bc) : (memref<2x128xbf16>) -> ()
        }
        %dst = memref.subview %bc[0, 0] [2, 128] [1, 1] : memref<2x128xbf16> to memref<2x128xbf16, strided<[128, 1]>>
        hivm.hir.load ins(%src : memref<2x128xbf16>) outs(%dst : memref<2x128xbf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : bf16 eviction_policy = <EvictFirst> core_type = <VECTOR>
        %t = bufferization.to_tensor %bc restrict writable : memref<2x128xbf16>
        func.call @compute_bf(%t) : (tensor<2x128xbf16>) -> ()
      }
    }
    return
  }
  func.func private @fill(memref<2x128xbf16>)
  func.func private @compute_bf(tensor<2x128xbf16>)
}

// -----

// The DMA in memref.copy form with an inline linalg.fill guard (the form the
// pipeline has before the load materializes as hivm.hir.load) hoists the
// same way.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @copy_form
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: scf.if %{{.+}} {
  // CHECK: linalg.fill ins(%{{.*}} : f16) outs(%[[ALLOC]] : memref<2x128xf16>)
  // CHECK: memref.copy %{{.*}}, %{{.*}}
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: scf.for
  // CHECK: func.call @compute(%[[T]])
  // CHECK-NOT: memref.copy
  func.func @copy_form(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<2x128xf16>)
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        memref.copy %src, %dst : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        func.call @compute(%t) : (tensor<2x128xf16>) -> ()
      }
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// Loops whose clusters were hoisted also get their invariant tensor compute
// hoisted (destination-passing cast/add/broadcast). The compute reading the
// variant copy stays inside the loop.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @compute_tail
  // CHECK: scf.for %{{.+}} to %{{.+}} step %{{.+}} : i32 {
  // CHECK: memref.subview %{{.+}}[{{.+}}, 0] [2, 128] [1, 1]
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: memref.copy %{{.*}}, %{{.*}}
  // CHECK: %[[T0:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<2x128xf32>
  // CHECK: hfusion.cast ins(%[[T0]] : tensor<2x128xf16>) outs(%[[E]] : tensor<2x128xf32>)
  // CHECK: %[[SUM:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins({{.*}}, {{.*}} : tensor<2x128xf32>, tensor<2x128xf32>) outs(%[[E]]
  // CHECK: %[[E2:.*]] = tensor.empty() : tensor<1x2x128xf32>
  // CHECK: %[[B:.*]] = linalg.broadcast ins(%[[SUM]] : tensor<2x128xf32>) outs(%[[E2]]
  // CHECK: scf.for %{{.+}} to %{{.+}} step %{{.+}} : i32 {
  // CHECK: memref.copy %{{.*}}, %{{.*}}
  // CHECK: %[[VC:.*]] = hfusion.cast ins({{.*}} : tensor<1x2x128xf16>) outs(%[[E2]] : tensor<1x2x128xf32>)
  // CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[B]], %[[VC]]
  func.func @compute_tail(%src: memref<4x128xf16>, %vsrc: memref<64x2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    scf.for %i = %c0 to %ub step %c1 : i32 {
      %ii = arith.index_cast %i : i32 to index
      %sub = memref.subview %src[%ii, 0] [2, 128] [1, 1] : memref<4x128xf16> to memref<2x128xf16, strided<[128, 1], offset: ?>>
      scf.for %j = %c0 to %ub step %c1 : i32 {
        %alloc = memref.alloc() : memref<2x128xf16>
        scf.if %cond {
          linalg.fill ins(%cst : f16) outs(%alloc : memref<2x128xf16>)
        }
        %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
        memref.copy %sub, %dst : memref<2x128xf16, strided<[128, 1], offset: ?>> to memref<2x128xf16, strided<[128, 1]>>
        %t0 = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
        %e = tensor.empty() : tensor<2x128xf32>
        %cast0 = hfusion.cast ins(%t0 : tensor<2x128xf16>) outs(%e : tensor<2x128xf32>) -> tensor<2x128xf32>
        %cast1 = hfusion.cast ins(%t0 : tensor<2x128xf16>) outs(%e : tensor<2x128xf32>) -> tensor<2x128xf32>
        %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cast0, %cast1 : tensor<2x128xf32>, tensor<2x128xf32>) outs(%e : tensor<2x128xf32>) -> tensor<2x128xf32>
        %e2 = tensor.empty() : tensor<1x2x128xf32>
        %b = linalg.broadcast ins(%sum : tensor<2x128xf32>) outs(%e2 : tensor<1x2x128xf32>) dimensions = [0]
        %ji = arith.index_cast %j : i32 to index
        %vsub = memref.subview %vsrc[%ji, 0, 0] [1, 2, 128] [1, 1, 1] : memref<64x2x128xf16> to memref<1x2x128xf16, strided<[256, 128, 1], offset: ?>>
        %vdst = memref.alloc() : memref<1x2x128xf16>
        memref.copy %vsub, %vdst : memref<1x2x128xf16, strided<[256, 128, 1], offset: ?>> to memref<1x2x128xf16>
        %t1 = bufferization.to_tensor %vdst restrict writable : memref<1x2x128xf16>
        %vc = hfusion.cast ins(%t1 : tensor<1x2x128xf16>) outs(%e2 : tensor<1x2x128xf32>) -> tensor<1x2x128xf32>
        %prod = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%b, %vc : tensor<1x2x128xf32>, tensor<1x2x128xf32>) outs(%e2 : tensor<1x2x128xf32>) -> tensor<1x2x128xf32>
        func.call @use(%prod) : (tensor<1x2x128xf32>) -> ()
      }
    }
    return
  }
  func.func private @use(tensor<1x2x128xf32>)
}

// -----

// Invariant compute without a load cluster: nothing hoists, the compute
// hoist is anchored to load-hoisted loops only.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @compute_no_load
  // CHECK: scf.for
  // CHECK: tensor.empty
  // CHECK: hfusion.cast
  func.func @compute_no_load(%t0: tensor<2x128xf16>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %ub step %c1 : i32 {
      %e = tensor.empty() : tensor<2x128xf32>
      %c = hfusion.cast ins(%t0 : tensor<2x128xf16>) outs(%e : tensor<2x128xf32>) -> tensor<2x128xf32>
      func.call @use(%c) : (tensor<2x128xf32>) -> ()
    }
    return
  }
  func.func private @use(tensor<2x128xf32>)
}

// -----

// The cluster result is the loop's yielded value, flowing into an iter_arg
// whose init comes from another buffer. Hoisting would leave a degenerate
// iter_arg chain that later passes stitch with per-iteration buffer copies,
// so the cluster stays inside the loop.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @yielded_cluster
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @yielded_cluster(%src: memref<2x128xf16>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    %init_alloc = memref.alloc() : memref<2x128xf16>
    %init = bufferization.to_tensor %init_alloc restrict writable : memref<2x128xf16>
    %r = scf.for %j = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<2x128xf16>) : i32 {
      %alloc = memref.alloc() : memref<2x128xf16>
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      scf.yield %t : tensor<2x128xf16>
    }
    func.call @use_t(%r) : (tensor<2x128xf16>) -> ()
    return
  }
  func.func private @use_t(tensor<2x128xf16>)
}

// -----

// A positive compile-time constant trip count: the cluster hoists before
// the loop without a guard.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @constant_trip_hoists
  // CHECK-NOT: scf.if
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x128xf16>
  // CHECK: %[[DST:.*]] = memref.subview %[[ALLOC]]
  // CHECK: hivm.hir.load ins(%{{.*}} outs(%[[DST]]
  // CHECK: %[[T:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: scf.for
  // CHECK: func.call @compute(%[[T]])
  // CHECK-NOT: hivm.hir.load
  func.func @constant_trip_hoists(%src: memref<2x128xf16>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    scf.for %j = %c0 to %c2 step %c1 : i32 {
      %pv = arith.constant 0.0 : f16
      %alloc = memref.alloc() : memref<2x128xf16>
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %pv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      func.call @compute(%t) : (tensor<2x128xf16>) -> ()
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// A compile-time constant zero trip count (lb == ub): the loop never runs,
// so the cluster stays inside it.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @constant_zero_trip
  // CHECK-NOT: scf.if
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @constant_zero_trip(%src: memref<2x128xf16>) {
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %j = %c4 to %c4 step %c1 : i32 {
      %pv = arith.constant 0.0 : f16
      %alloc = memref.alloc() : memref<2x128xf16>
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %pv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      func.call @compute(%t) : (tensor<2x128xf16>) -> ()
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// Constant bounds in the wrong direction (trips < 0): the cluster stays.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @constant_negative_trip
  // CHECK-NOT: scf.if
  // CHECK: scf.for
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  func.func @constant_negative_trip(%src: memref<2x128xf16>) {
    %c8 = arith.constant 8 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %j = %c8 to %c4 step %c1 : i32 {
      %pv = arith.constant 0.0 : f16
      %alloc = memref.alloc() : memref<2x128xf16>
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %pv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      func.call @compute(%t) : (tensor<2x128xf16>) -> ()
    }
    return
  }
  func.func private @compute(tensor<2x128xf16>)
}

// -----

// A runtime trip count: the loop moves into the entry-test guard, the
// cluster sits in front of it, and the else region yields the init args.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @runtime_trip_guarded
  // CHECK: %[[INIT:.*]] = bufferization.to_tensor
  // CHECK: %[[CMP:.*]] = arith.cmpi slt
  // CHECK: %[[R:.*]] = scf.if %[[CMP]]
  // CHECK: memref.alloc
  // CHECK: hivm.hir.load
  // CHECK: %[[T:.*]] = bufferization.to_tensor
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: func.call @mix(%[[T]]
  // CHECK: scf.yield %[[LOOP]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield %[[INIT]]
  // CHECK: {{(func\.)?call}} @use_t(%[[R]])
  // CHECK-NOT: hivm.hir.load
  func.func @runtime_trip_guarded(%src: memref<2x128xf16>, %ub: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init_alloc = memref.alloc() : memref<2x128xf16>
    %init = bufferization.to_tensor %init_alloc restrict writable : memref<2x128xf16>
    %r = scf.for %j = %c0 to %ub step %c1 iter_args(%acc = %init) -> (tensor<2x128xf16>) : i32 {
      %pv = arith.constant 0.0 : f16
      %alloc = memref.alloc() : memref<2x128xf16>
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %pv : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      %u = func.call @mix(%t, %acc) : (tensor<2x128xf16>, tensor<2x128xf16>) -> tensor<2x128xf16>
      scf.yield %u : tensor<2x128xf16>
    }
    func.call @use_t(%r) : (tensor<2x128xf16>) -> ()
    return
  }
  func.func private @mix(tensor<2x128xf16>, tensor<2x128xf16>) -> tensor<2x128xf16>
  func.func private @use_t(tensor<2x128xf16>)
}

// -----

// A mix-split kernel carries hivm.hir.anchor ops partitioning its body into
// the intervals the delayed cross-core GSS matches across the cube / vector /
// backup copies. Hoisting across those anchors desynchronizes the interval
// structure, so the whole function is left alone even though the cluster
// would otherwise hoist.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @anchored_kernel
  // CHECK-NOT: scf.if
  // CHECK: scf.for
  // CHECK: hivm.hir.anchor {id = 4 : i64}
  // CHECK: memref.alloc
  // CHECK: hivm.hir.anchor {id = 5 : i64}
  // CHECK: hivm.hir.load
  // CHECK: hivm.hir.anchor {id = 6 : i64}
  // CHECK: func.call @compute
  func.func @anchored_kernel(%src: memref<2x128xf16>, %ub: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant 0.0 : f16
    hivm.hir.anchor {id = 3 : i64}
    scf.for %j = %c0 to %ub step %c1 : i32 {
      hivm.hir.anchor {id = 4 : i64}
      %alloc = memref.alloc() : memref<2x128xf16>
      scf.if %cond {
        func.call @fill(%alloc) : (memref<2x128xf16>) -> ()
      }
      %dst = memref.subview %alloc[0, 0] [2, 128] [1, 1] : memref<2x128xf16> to memref<2x128xf16, strided<[128, 1]>>
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.load ins(%src : memref<2x128xf16>) outs(%dst : memref<2x128xf16, strided<[128, 1]>>) pad_mode = <PadValue> pad_value = %cst : f16 eviction_policy = <EvictFirst> core_type = <VECTOR>
      hivm.hir.anchor {id = 6 : i64}
      %t = bufferization.to_tensor %alloc restrict writable : memref<2x128xf16>
      func.call @compute(%t) : (tensor<2x128xf16>) -> ()
    }
    return
  }
  func.func private @fill(memref<2x128xf16>)
  func.func private @compute(tensor<2x128xf16>)
}
