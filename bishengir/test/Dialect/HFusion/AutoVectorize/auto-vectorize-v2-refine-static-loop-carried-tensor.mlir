// RUN: bishengir-opt %s --hfusion-auto-vectorize-v2 | FileCheck %s

// MCF's whole-function single-iteration-loop cleanup can inline a short tiled
// loop and leave a fully static tensor cast to a dynamic scf.for iter_arg.
// AutoVectorizeV2 must restore the static loop-carried contract when the loop
// only updates that tensor through shape-preserving insert_slice operations.

// CHECK-LABEL: func.func @refine_exact_softmax_tree_reduce_stage
// CHECK-NOT: tensor.cast {{.*}} tensor<1x16x32xf32> to tensor<1x16x?xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1x16x32xf32>)
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 2, 8, 32] : tensor<1x16x32xf32> into tensor<1x2x8x32xf32>
// CHECK-LABEL: func.func @do_not_refine_without_softmax_row_loop
// CHECK: tensor.cast {{.*}} : tensor<1x16x32xf32> to tensor<1x16x?xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1x16x?xf32>)
// CHECK-LABEL: func.func @do_not_refine_other_tree_stage
// CHECK: tensor.cast {{.*}} : tensor<1x3x32xf32> to tensor<1x3x?xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1x3x?xf32>)
// CHECK-LABEL: func.func @do_not_refine_other_tail_width
// CHECK: tensor.cast {{.*}} : tensor<1x16x64xf32> to tensor<1x16x?xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1x16x?xf32>)
// CHECK-LABEL: func.func @do_not_refine_other_static_shape
// CHECK: tensor.cast {{.*}} : tensor<3x7xi16> to tensor<?x?xi16>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<?x?xi16>)
// CHECK-LABEL: func.func @do_not_refine_multiple_iter_args
// CHECK: tensor.cast {{.*}} : tensor<3x7xi16> to tensor<?x?xi16>
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}) -> (tensor<?x?xi16>, tensor<3x7xi16>)
// CHECK-LABEL: func.func @do_not_refine_dynamic_source
// CHECK: tensor.cast {{.*}} : tensor<1x16x?xf32> to tensor<1x?x?xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1x?x?xf32>)

module {
  func.func @refine_exact_softmax_tree_reduce_stage(
      %rows: tensor<19x32xf32>, %init: tensor<1x16x32xf32>,
      %update: tensor<1x1x32xf32>) -> tensor<19x32xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c19 = arith.constant 19 : index
    %result = scf.for %row = %c0 to %c19 step %c1
        iter_args(%rows_arg = %rows) -> tensor<19x32xf32> {
      %cast = tensor.cast %init
          : tensor<1x16x32xf32> to tensor<1x16x?xf32>
      %stage = scf.for %iv = %c0 to %c16 step %c1
          iter_args(%arg = %cast) -> tensor<1x16x?xf32> {
        %slice = tensor.extract_slice %arg[0, %iv, 0] [1, 1, 32]
            [1, 1, 1] : tensor<1x16x?xf32> to tensor<1x1x32xf32>
        %next = tensor.insert_slice %update into %arg[0, %iv, 0] [1, 1, 32]
            [1, 1, 1] : tensor<1x1x32xf32> into tensor<1x16x?xf32>
        scf.yield %next : tensor<1x16x?xf32>
      }
      %dim = tensor.dim %stage, %c2 : tensor<1x16x?xf32>
      %expanded = tensor.expand_shape %stage [[0], [1, 2], [3]]
          output_shape [1, 2, 8, %dim]
          : tensor<1x16x?xf32> into tensor<1x2x8x?xf32>
      %piece = tensor.extract_slice %expanded[0, 0, 0, 0] [1, 1, 1, 32]
          [1, 1, 1, 1] : tensor<1x2x8x?xf32> to tensor<1x1x1x32xf32>
      %flat = tensor.collapse_shape %piece [[0, 1, 2], [3]]
          : tensor<1x1x1x32xf32> into tensor<1x32xf32>
      %next_rows = tensor.insert_slice %flat into %rows_arg[%row, 0] [1, 32]
          [1, 1] : tensor<1x32xf32> into tensor<19x32xf32>
      scf.yield %next_rows : tensor<19x32xf32>
    }
    return %result : tensor<19x32xf32>
  }

  func.func @do_not_refine_without_softmax_row_loop(
      %init: tensor<1x16x32xf32>, %update: tensor<1x1x32xf32>)
      -> tensor<1x16x?xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cast = tensor.cast %init
        : tensor<1x16x32xf32> to tensor<1x16x?xf32>
    %result = scf.for %iv = %c0 to %c16 step %c1
        iter_args(%arg = %cast) -> tensor<1x16x?xf32> {
      %slice = tensor.extract_slice %arg[0, %iv, 0] [1, 1, 32]
          [1, 1, 1] : tensor<1x16x?xf32> to tensor<1x1x32xf32>
      %next = tensor.insert_slice %update into %arg[0, %iv, 0] [1, 1, 32]
          [1, 1, 1] : tensor<1x1x32xf32> into tensor<1x16x?xf32>
      scf.yield %next : tensor<1x16x?xf32>
    }
    return %result : tensor<1x16x?xf32>
  }

  func.func @do_not_refine_other_tree_stage(
      %rows: tensor<19x32xf32>, %init: tensor<1x3x32xf32>,
      %update: tensor<1x1x32xf32>) -> tensor<19x32xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c19 = arith.constant 19 : index
    %result = scf.for %row = %c0 to %c19 step %c1
        iter_args(%rows_arg = %rows) -> tensor<19x32xf32> {
      %cast = tensor.cast %init : tensor<1x3x32xf32> to tensor<1x3x?xf32>
      %stage = scf.for %iv = %c0 to %c3 step %c1
          iter_args(%arg = %cast) -> tensor<1x3x?xf32> {
        %slice = tensor.extract_slice %arg[0, %iv, 0] [1, 1, 32]
            [1, 1, 1] : tensor<1x3x?xf32> to tensor<1x1x32xf32>
        %next = tensor.insert_slice %update into %arg[0, %iv, 0] [1, 1, 32]
            [1, 1, 1] : tensor<1x1x32xf32> into tensor<1x3x?xf32>
        scf.yield %next : tensor<1x3x?xf32>
      }
      %piece = tensor.extract_slice %stage[0, 0, 0] [1, 1, 32]
          [1, 1, 1] : tensor<1x3x?xf32> to tensor<1x1x32xf32>
      %flat = tensor.collapse_shape %piece [[0, 1], [2]]
          : tensor<1x1x32xf32> into tensor<1x32xf32>
      %next_rows = tensor.insert_slice %flat into %rows_arg[%row, 0] [1, 32]
          [1, 1] : tensor<1x32xf32> into tensor<19x32xf32>
      scf.yield %next_rows : tensor<19x32xf32>
    }
    return %result : tensor<19x32xf32>
  }

  func.func @do_not_refine_other_tail_width(
      %rows: tensor<19x32xf32>, %init: tensor<1x16x64xf32>,
      %update: tensor<1x1x64xf32>) -> tensor<19x32xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c19 = arith.constant 19 : index
    %result = scf.for %row = %c0 to %c19 step %c1
        iter_args(%rows_arg = %rows) -> tensor<19x32xf32> {
      %cast = tensor.cast %init
          : tensor<1x16x64xf32> to tensor<1x16x?xf32>
      %stage = scf.for %iv = %c0 to %c16 step %c1
          iter_args(%arg = %cast) -> tensor<1x16x?xf32> {
        %slice = tensor.extract_slice %arg[0, %iv, 0] [1, 1, 64]
            [1, 1, 1] : tensor<1x16x?xf32> to tensor<1x1x64xf32>
        %next = tensor.insert_slice %update into %arg[0, %iv, 0] [1, 1, 64]
            [1, 1, 1] : tensor<1x1x64xf32> into tensor<1x16x?xf32>
        scf.yield %next : tensor<1x16x?xf32>
      }
      %piece = tensor.extract_slice %stage[0, 0, 0] [1, 1, 32]
          [1, 1, 1] : tensor<1x16x?xf32> to tensor<1x1x32xf32>
      %flat = tensor.collapse_shape %piece [[0, 1], [2]]
          : tensor<1x1x32xf32> into tensor<1x32xf32>
      %next_rows = tensor.insert_slice %flat into %rows_arg[%row, 0] [1, 32]
          [1, 1] : tensor<1x32xf32> into tensor<19x32xf32>
      scf.yield %next_rows : tensor<19x32xf32>
    }
    return %result : tensor<19x32xf32>
  }

  func.func @do_not_refine_other_static_shape(
      %init: tensor<3x7xi16>, %update: tensor<1x7xi16>)
      -> tensor<3x7xi16>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cast = tensor.cast %init
        : tensor<3x7xi16> to tensor<?x?xi16>
    %result = scf.for %iv = %c0 to %c3 step %c1
        iter_args(%arg = %cast) -> tensor<?x?xi16> {
      %next = tensor.insert_slice %update into %arg[%iv, 0] [1, 7]
          [1, 1] : tensor<1x7xi16> into tensor<?x?xi16>
      scf.yield %next : tensor<?x?xi16>
    }
    %static = tensor.cast %result : tensor<?x?xi16> to tensor<3x7xi16>
    return %static : tensor<3x7xi16>
  }

  func.func @do_not_refine_multiple_iter_args(
      %init: tensor<3x7xi16>, %other: tensor<3x7xi16>,
      %update: tensor<1x7xi16>) -> (tensor<3x7xi16>, tensor<3x7xi16>)
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cast = tensor.cast %init
        : tensor<3x7xi16> to tensor<?x?xi16>
    %result:2 = scf.for %iv = %c0 to %c3 step %c1
        iter_args(%arg = %cast, %other_arg = %other)
        -> (tensor<?x?xi16>, tensor<3x7xi16>) {
      %next = tensor.insert_slice %update into %arg[%iv, 0] [1, 7]
          [1, 1] : tensor<1x7xi16> into tensor<?x?xi16>
      %other_next = tensor.insert_slice %update into %other_arg[%iv, 0] [1, 7]
          [1, 1] : tensor<1x7xi16> into tensor<3x7xi16>
      scf.yield %next, %other_next : tensor<?x?xi16>, tensor<3x7xi16>
    }
    %static = tensor.cast %result#0 : tensor<?x?xi16> to tensor<3x7xi16>
    return %static, %result#1 : tensor<3x7xi16>, tensor<3x7xi16>
  }

  func.func @do_not_refine_dynamic_source(
      %init: tensor<1x16x?xf32>, %update: tensor<1x1x?xf32>)
      -> tensor<1x?x?xf32>
      attributes {
        hacc.entry,
        hacc.function_kind = #hacc.function_kind<DEVICE>,
        parallel_mode = "simd"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %dim = tensor.dim %init, %c2 : tensor<1x16x?xf32>
    %cast = tensor.cast %init
        : tensor<1x16x?xf32> to tensor<1x?x?xf32>
    %result = scf.for %iv = %c0 to %c16 step %c1
        iter_args(%arg = %cast) -> tensor<1x?x?xf32> {
      %next = tensor.insert_slice %update into %arg[0, %iv, 0] [1, 1, %dim]
          [1, 1, 1] : tensor<1x1x?xf32> into tensor<1x?x?xf32>
      scf.yield %next : tensor<1x?x?xf32>
    }
    return %result : tensor<1x?x?xf32>
  }
}
