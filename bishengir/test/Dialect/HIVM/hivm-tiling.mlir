// RUN: bishengir-opt %s --split-input-file --transform-interpreter --cse --canonicalize-ext --cse | FileCheck %s

// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.vabs

module {
  func.func @static_unary(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.vabs ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.vabs"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.vabs">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vabs">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

module {
  func.func @static_binary(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.vadd"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.vadd">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vadd">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.vabs

module {
  func.func @dyn(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
    %1 = hivm.hir.vabs ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.vabs"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.vabs">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vabs">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.vadd

module {
  func.func @dyn_vadd(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
    %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.vadd"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.vadd">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vadd">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// CHECK-LABEL: func.func @dyn_linalg_brc(
// CHECK: scf.for %[[LOOP1:.*]] = %c0 to %dim step %c1 iter_args(%[[ACC1:.*]] =
// CHECK: scf.for %[[LOOP2:.*]] = %c0 to %c16 step %c2 iter_args(%[[ACC2:.*]] =
// CHECK: extract_slice
// CHECK-SAME: %arg0{{\[}}%[[LOOP1]]] [1] [1]
// CHECK: extract_slice
// CHECK-SAME: %[[ACC2]]{{\[}}%[[LOOP1]], %[[LOOP2]]] [1, 2] [1, 1]
// CHECK: linalg.broadcast
// CHECK: return
// CHECK-LABEL: func.func @dyn_linalg_reduce(
// CHECK: scf.for %[[LOOP1:.*]] = %c0 to %dim step %c1 iter_args(%[[ACC1:.*]] =
// CHECK: scf.for %[[LOOP2:.*]] = %c0 to %dim_0 step %c2 iter_args(%[[ACC2:.*]] =
// CHECK: extract_slice
// CHECK-SAME: %arg0{{\[}}%[[LOOP1]], %[[LOOP2]]] [1, %{{.*}}] [1, 1]
// CHECK: extract_slice
// CHECK-SAME: %[[ACC2]]{{\[}}%[[LOOP1]]] [1] [1]
// CHECK: linalg.reduce
// CHECK: return
module {
  func.func @dyn_linalg_brc(%arg0: tensor<?xf32>) -> tensor<?x16xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = tensor.empty(%dim) : tensor<?x16xf32>
    %1 = linalg.broadcast ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x16xf32>) dimensions = [1]
    return %1 : tensor<?x16xf32>
  }

  func.func @dyn_linalg_reduce(%arg0: tensor<?x?xf32>) -> tensor<?xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.reduce { arith.addf } ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?xf32>) dimensions = [1]
    return %1 : tensor<?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op

      %func1, %func2 = transform.split_handle %0
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      // Transform for vbrc
      %1 = transform.structured.match ops{["linalg.broadcast"]} in %func1 : (!transform.any_op) -> !transform.op<"linalg.broadcast">
      %tiled_vbrc, %loops_vbrc:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"linalg.broadcast">) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      // Transform for vreduce
      %2 = transform.structured.match ops{["linalg.reduce"]} in %func2 : (!transform.any_op) -> !transform.op<"linalg.reduce">
      %tiled_vreduce, %loops_vreduce:2 = transform.structured.tile_using_for %2 tile_sizes [1, 2] : (!transform.op<"linalg.reduce">) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      transform.yield
    }
  }
}

// -----

// CHECK-LABEL: func.func @dyn_vbrc(
// CHECK: scf.for %[[LOOP1:.*]] = %c0 to %dim step %c1 iter_args(%[[ACC1:.*]] =
// CHECK: scf.for %[[LOOP2:.*]] = %c0 to %c16 step %c2 iter_args(%[[ACC2:.*]] =
// CHECK: extract_slice
// CHECK-SAME: %arg0{{\[}}%[[LOOP1]], 0] [1, 1] [1, 1]
// CHECK: extract_slice
// CHECK-SAME: %[[ACC2]]{{\[}}%[[LOOP1]], %[[LOOP2]]] [1, 2] [1, 1]
// CHECK: hivm.hir.vbrc
// CHECK: return
// CHECK-LABEL: func.func @dyn_vreduce(
// CHECK: scf.for %[[LOOP1:.*]] = %c0 to %dim step %c1 iter_args(%[[ACC1:.*]] =
// CHECK: scf.for %[[LOOP2:.*]] = %c0 to %dim_0 step %c2 iter_args(%[[ACC2:.*]] =
// CHECK: extract_slice
// CHECK-SAME: %arg0{{\[}}%[[LOOP1]], %[[LOOP2]]] [1, %{{.*}}] [1, 1]
// CHECK: extract_slice
// CHECK-SAME: %[[ACC2]]{{\[}}%[[LOOP1]], 0] [1, 1] [1, 1]
// CHECK: hivm.hir.vreduce
// CHECK: return
module {
  func.func @dyn_vbrc(%arg0: tensor<?x1xf32>) -> tensor<?x16xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x1xf32>
    %dim_1 = arith.constant 16 : index  // Using a fixed dimension for the broadcast target
    %0 = tensor.empty(%dim) : tensor<?x16xf32>
    %1 = hivm.hir.vbrc ins(%arg0 : tensor<?x1xf32>) outs(%0 : tensor<?x16xf32>) broadcast_dims = [1] -> tensor<?x16xf32>
    return %1 : tensor<?x16xf32>
  }

  func.func @dyn_vreduce(%arg0: tensor<?x?xf32>) -> tensor<?x1xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?x1xf32>
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
    return %1 : tensor<?x1xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op

      %func1, %func2 = transform.split_handle %0
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      // Transform for vbrc
      %1 = transform.structured.match ops{["hivm.hir.vbrc"]} in %func1 : (!transform.any_op) -> !transform.op<"hivm.hir.vbrc">
      %tiled_vbrc, %loops_vbrc:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vbrc">) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      // Transform for vreduce
      %2 = transform.structured.match ops{["hivm.hir.vreduce"]} in %func2 : (!transform.any_op) -> !transform.op<"hivm.hir.vreduce">
      %tiled_vreduce, %loops_vreduce:2 = transform.structured.tile_using_for %2 tile_sizes [1, 2] : (!transform.op<"hivm.hir.vreduce">) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      transform.yield
    }
  }
}

// -----
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.store

module {
  func.func @static_unary(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.store ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.store"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.store">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.store">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.load

module {
  func.func @static_unary(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.load ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.load"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.load">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.load">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.copy

module {
  func.func @static_unary(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> attributes {always_inline, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.copy ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.copy"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.copy">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.copy">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// Test getIterationDomainTileFromResultTile

// CHECK: %[[CONSTANT1:.*]] = arith.constant 1 : index
// CHECK: %[[CONSTANT0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %arg0, %[[CONSTANT0]] : tensor<?x?xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]]) : tensor<?x1xf32>
// CHECK: %[[DIM1:.*]] = tensor.dim %arg0, %[[CONSTANT1]] : tensor<?x?xf32>
// CHECK: scf.for %[[IV:.*]] = %[[CONSTANT0]] to %[[DIM0]]
// CHECK:   %[[TILE_SIZE:.*]] = affine.min #map(%[[IV]])[%[[DIM0]]
// CHECK:   %[[SLICE_INPUT:.*]] = tensor.extract_slice %arg0[%[[IV]], 0] [%[[TILE_SIZE]], %[[DIM1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:   %[[SLICE_INIT:.*]]  = tensor.extract_slice %[[EMPTY]][%[[IV]], 0] [%[[TILE_SIZE]], 1] [1, 1] : tensor<?x1xf32> to tensor<?x1xf32>
// CHECK:   hivm.hir.vreduce <sum> ins(%[[SLICE_INPUT]] : tensor<?x?xf32>) outs(%[[SLICE_INIT]] : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
module {
  func.func @dyn_vreduce(%arg0: tensor<?x?xf32>, %arg1: tensor<?x1xf32>) -> tensor<?x1xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?x1xf32>
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x1xf32>) reduce_dims = [1] -> tensor<?x1xf32>
    %2 = hivm.hir.store ins(%1 : tensor<?x1xf32>) outs(%arg1 : tensor<?x1xf32>) -> tensor<?x1xf32>
    return %2 : tensor<?x1xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.store"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.store">
      %tiled_linalg_op, %loops = transform.structured.tile_using_for %1 tile_sizes [2, 0] : (!transform.op<"hivm.hir.store">) -> (!transform.any_op, !transform.any_op)
      %2 = transform.structured.match ops{["hivm.hir.vreduce"]} in %0 : (!transform.any_op) -> !transform.any_op
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// CHECK-DAG: [[MAP:#map[0-9]*]] = affine_map<(d0)[s0] -> (-d0 + 128, s0)>
// CHECK-DAG: [[MAP1:#map[0-9]*]] = affine_map<(d0) -> (-d0 + 63, 32)>
// CHECK-DAG: [[MAP2:#map[0-9]*]] = affine_map<(d0) -> (-d0 + 512, 128)>

// CHECK: func @mmadL1([[A:%arg[0-9]+]]: tensor{{.*}}, [[B:%arg[0-9]+]]: tensor{{.*}}, [[C:%arg[0-9]+]]: tensor{{.*}}, [[TILE:%arg[0-9]+]]: index
// CHECK:   %[[C512:.*]] = arith.constant 512 : index
// CHECK:   %[[C32:.*]]= arith.constant 32 : index
// CHECK:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:   %[[C_TRUE:.*]] = arith.constant true
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[C128:.*]] = arith.constant 128 : index
// CHECK:   scf.for %[[IV0:.*]] = %[[C0]] to %[[C128]] step [[TILE]]
// CHECK:     scf.for %[[IV1:.*]] = %[[C0]] to %[[C64]] step %[[C32]]
// CHECK:       scf.for %[[IV2:.*]] = %[[C0]] to %[[C512]] step %[[C128]]
// CHECK:         %[[REAL_M:.*]] = affine.min [[MAP]](%[[IV0]])[[[TILE]]]
// CHECK:         %[[SLICE_A:.*]] = tensor.extract_slice [[A]][%[[IV0]], %[[IV2]]] [%[[REAL_M]], 128] [1, 1] : tensor<128x512xf16> to tensor<?x128xf16>
// CHECK:         %[[SLICE_B:.*]] = tensor.extract_slice [[B]][%[[IV2]], %[[IV1]]] [128, 32] [1, 1] : tensor<512x64xf16> to tensor<128x32xf16>
// CHECK:         %[[SLICE_C:.*]] = tensor.extract_slice {{.*}}[%[[IV0]], %[[IV1]]] [%[[REAL_M]], 32] [1, 1] : tensor<128x64xf32> to tensor<?x32xf32>
// CHECK:         %[[REAL_N:.*]] = affine.min [[MAP1]](%[[IV1]])
// CHECK:         %[[REAL_K:.*]] = affine.min [[MAP2]](%[[IV2]])
// CHECK:         hivm.hir.mmadL1 ins(%[[SLICE_A]], %[[SLICE_B]],
// CHECK-SAME:                        %[[C_TRUE]], %[[REAL_M]], %[[REAL_K]], %[[REAL_N]]
// CHECK-SAME:                    outs(%[[SLICE_C]]
module {
  func.func @mmadL1(%a: tensor<128x512xf16>,
                    %b: tensor<512x64xf16>,
                    %c: tensor<128x64xf32>,
                    %tile_m: index) -> tensor<128x64xf32> {
    %c512 = arith.constant 512 : index
    %c128 = arith.constant 128 : index
    %c63 = arith.constant 63 : index
    %true = arith.constant true
    %1 = hivm.hir.mmadL1 ins(%a, %b, %true, %c128, %c512, %c63 : tensor<128x512xf16>, tensor<512x64xf16>, i1, index, index, index) outs(%c : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %1 : tensor<128x64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.func.get_func_argument %0[3]: (!transform.any_op) -> !transform.any_value
      %2 = transform.structured.match ops{["hivm.hir.mmadL1"]} in %0 : (!transform.any_op) -> !transform.any_op
                                                     // [  M,  N,   K]
      transform.structured.tile_using_for %2 tile_sizes [%1, 32, 128] :
        (!transform.any_op, !transform.any_value) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hivm.hir.fixpipe

module {
  func.func @fixpipe(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = hivm.hir.fixpipe ins(%arg0 : tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["hivm.hir.fixpipe"]} in %0 : (!transform.any_op) -> !transform.op<"hivm.hir.fixpipe">
      %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %1 tile_sizes [1, 2] : (!transform.op<"hivm.hir.fixpipe">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}