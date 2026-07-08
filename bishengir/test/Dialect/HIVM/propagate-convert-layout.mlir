// RUN: bishengir-opt --hivm-propagate-convert-layout="allow-agnostic-ops=true enable-elementwise-propagate=true" --canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @propagate_up_from_for_result(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<16x16xf16>,
// CHECK:           %[[CONVERT_LAYOUT:.*]] = hivm.hir.convert_layout %[[VAL_0]] output_shape [1, 1, 16, 16]
// CHECK-SAME:      (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK:           %[[FOR_RETURN:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %[[CONVERT_LAYOUT]]) -> (tensor<1x1x16x16xf16>) {
// CHECK:             %[[VAL_9:.*]] = hivm.hir.vexp ins(%[[ITER_ARG]] : tensor<1x1x16x16xf16>)
// CHECK:             scf.yield %[[VAL_9]] : tensor<1x1x16x16xf16>
// CHECK:           }
// CHECK:           return %[[FOR_RETURN]] : tensor<1x1x16x16xf16>
func.func @propagate_up_from_for_result(
  %init: tensor<16x16xf16>, %lb: index, %ub: index, %step: index
) -> tensor<1x1x16x16xf16> {
  %r = scf.for %iv = %lb to %ub step %step
      iter_args(%arg = %init) -> (tensor<16x16xf16>) {
    %tmp = tensor.empty() : tensor<16x16xf16>
    %mid = hivm.hir.vexp ins(%arg : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>
    scf.yield %mid : tensor<16x16xf16>
  }

  %r_up = hivm.hir.convert_layout %r output_shape [1, 1, 16, 16]
    {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
     srcLayout = #hivm.data_layout<ND>}
    : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>

  return %r_up : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL:   func.func @propagate_down_from_for_init(
// CHECK:           %[[VAL_4:.*]] = scf.for
// CHECK-SAME:      -> (tensor<1x1x16x16xf16>) {
// CHECK:           %[[VAL_8:.*]] = hivm.hir.vexp
// CHECK-SAME:      -> tensor<1x1x16x16xf16>
// CHECK:           scf.yield %[[VAL_8]] : tensor<1x1x16x16xf16>
// CHECK:           %[[VAL_9:.*]] = hivm.hir.convert_layout %[[VAL_4]]
// CHECK-SAME:      output_shape [16, 16]
// CHECK:           return %[[VAL_9]] : tensor<16x16xf16>

func.func @propagate_down_from_for_init(
  %init_up: tensor<1x1x16x16xf16>, %lb: index, %ub: index, %step: index
) -> tensor<16x16xf16> {
  %init_down = hivm.hir.convert_layout %init_up output_shape [16, 16]
    {dstLayout = #hivm.data_layout<ND>,
     srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
    : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>

  %r = scf.for %iv = %lb to %ub step %step
      iter_args(%arg = %init_down) -> (tensor<16x16xf16>) {
    %tmp = tensor.empty() : tensor<16x16xf16>
    %mid = hivm.hir.vexp ins(%arg : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>
    scf.yield %mid : tensor<16x16xf16>
  }

  return %r : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_up_from_for_body_iter_arg(
// CHECK:      hivm.hir.convert_layout %{{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %[[VAL_5:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1x1x16x16xf16>)
// CHECK:      %[[VAL_9:.*]] = hivm.hir.vexp ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:      scf.yield %[[VAL_9]] : tensor<1x1x16x16xf16>
// CHECK:      %[[VAL_10:.*]] = hivm.hir.convert_layout %[[VAL_5]] output_shape [16, 16]
// CHECK:      return %[[VAL_10]] : tensor<16x16xf16>

func.func @propagate_up_from_for_body_iter_arg(
  %init: tensor<16x16xf16>, %lb: index, %ub: index, %step: index
) -> tensor<16x16xf16> {
  %r = scf.for %iv = %lb to %ub step %step
      iter_args(%arg = %init) -> (tensor<16x16xf16>) {
    %up = hivm.hir.convert_layout %arg output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>

    %tmp0 = tensor.empty() : tensor<1x1x16x16xf16>
    %mid_up = hivm.hir.vexp ins(%up : tensor<1x1x16x16xf16>) outs(%tmp0 : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>

    %down = hivm.hir.convert_layout %mid_up output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>

    scf.yield %down : tensor<16x16xf16>
  }
  return %r : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_down_from_for_yield(
// CHECK:      hivm.hir.convert_layout %{{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %[[VAL_5:.*]] = scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1x1x16x16xf16>)
// CHECK:      tensor.expand_shape %{{.*}} {{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %[[VAL_11:.*]] = hivm.hir.vexp ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:      scf.yield %[[VAL_11]] : tensor<1x1x16x16xf16>
// CHECK:      %[[VAL_12:.*]] = hivm.hir.convert_layout %[[VAL_5]] output_shape [16, 16]
// CHECK:      return %{{.*}} : tensor<16x16xf16>
func.func @propagate_down_from_for_yield(
  %init: tensor<16x16xf16>, %lb: index, %ub: index, %step: index
) -> tensor<16x16xf16> {
  %r = scf.for %iv = %lb to %ub step %step
      iter_args(%arg = %init) -> (tensor<16x16xf16>) {

    %arg2 = tensor.expand_shape %arg [[0, 1, 2], [3]] output_shape [1, 1, 16, 16]
        : tensor<16x16xf16> into tensor<1x1x16x16xf16>

    %tmp = tensor.empty() : tensor<1x1x16x16xf16>
    %mid = hivm.hir.vexp ins(%arg2 : tensor<1x1x16x16xf16>) outs(%tmp : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>

    %down = hivm.hir.convert_layout %mid output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>

    scf.yield %down : tensor<16x16xf16>
  }

  return %r : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_down_from_if_yields(
// CHECK-SAME: %[[C:.*]]: i1
// CHECK:      %[[IFR:.*]] = scf.if %[[C]] -> (tensor<1x1x16x16xf16>)
// CHECK:        %{{.*}} = hivm.hir.vexp ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %{{.*}} : tensor<1x1x16x16xf16>
// CHECK:      } else {
// CHECK:        %{{.*}} = hivm.hir.vabs ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %{{.*}} : tensor<1x1x16x16xf16>
// CHECK:      }
// CHECK:      %[[DOWN:.*]] = hivm.hir.convert_layout %[[IFR]] output_shape [16, 16]
// CHECK:      return %[[DOWN]] : tensor<16x16xf16>
func.func @propagate_down_from_if_yields(
    %cond: i1,
    %a_fr: tensor<1x1x16x16xf16>,
    %b_fr: tensor<1x1x16x16xf16>) -> tensor<16x16xf16> {
  %r = scf.if %cond -> (tensor<16x16xf16>) {
    %a_nd = hivm.hir.convert_layout %a_fr output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>

    %tmp = tensor.empty() : tensor<16x16xf16>
    %vexp_a = hivm.hir.vexp ins(%a_nd : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>

    scf.yield %vexp_a : tensor<16x16xf16>
  } else {
    %b_nd = hivm.hir.convert_layout %b_fr output_shape [16, 16]
      {dstLayout = #hivm.data_layout<ND>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>
    %tmp_b = tensor.empty() : tensor<16x16xf16>
    %vexp_b = hivm.hir.vabs ins(%b_nd : tensor<16x16xf16>) outs(%tmp_b : tensor<16x16xf16>) -> tensor<16x16xf16>

    scf.yield %vexp_b : tensor<16x16xf16>
  }
  return %r : tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_up_from_if_result_with_raw_user(
// CHECK-SAME: %[[C:.*]]: i1
// CHECK:      %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %{{.*}} = hivm.hir.convert_layout %{{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %[[IFR:.*]] = scf.if %[[C]] -> (tensor<1x1x16x16xf16>)
// CHECK:        %{{.*}} = hivm.hir.vexp ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %{{.*}} : tensor<1x1x16x16xf16>
// CHECK:      } else {
// CHECK:        %{{.*}} = hivm.hir.vabs ins(%{{.*}} : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %{{.*}} : tensor<1x1x16x16xf16>
// CHECK:      }
// CHECK:      %{{.*}} = hivm.hir.vexp ins(%[[IFR]] : tensor<1x1x16x16xf16>)
// CHECK:      %[[RAW:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [16, 16]
// CHECK:      return %[[RAW]], %[[IFR]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>
func.func @propagate_up_from_if_result_with_raw_user(
    %cond: i1,
    %a: tensor<16x16xf16>,
    %b: tensor<16x16xf16>) -> (tensor<16x16xf16>, tensor<1x1x16x16xf16>) {
  %tmp = tensor.empty() : tensor<16x16xf16>

  %r = scf.if %cond -> (tensor<16x16xf16>) {
    %modified_a = hivm.hir.vexp ins(%a : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>
    scf.yield %modified_a : tensor<16x16xf16>
  } else {

    %modified_b = hivm.hir.vabs ins(%b : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>
    scf.yield %modified_b : tensor<16x16xf16>
  }

  %r_up = hivm.hir.convert_layout %r output_shape [1, 1, 16, 16]
    {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
     srcLayout = #hivm.data_layout<ND>}
    : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>

  %raw_user = hivm.hir.vexp ins(%r : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>

  return %raw_user, %r_up : tensor<16x16xf16>, tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_up_from_while_result_with_raw_user(
// CHECK-SAME: %{{.*}}: tensor<16x16xf16>, %{{.*}}: i1
// CHECK:      %[[UP:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [1, 1, 16, 16]
// CHECK:      %[[W:.*]] = scf.while (%[[S:.*]] = %[[UP]]) : (tensor<1x1x16x16xf16>) -> {{.*}}
// CHECK:        scf.condition(%{{.*}}) %[[S]] : tensor<1x1x16x16xf16>
// CHECK:      } do {
// CHECK:      ^bb0(%[[A:.*]]: tensor<1x1x16x16xf16>):
// CHECK:        %{{.*}} = hivm.hir.vexp ins(%[[A]] : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %{{.*}} : tensor<1x1x16x16xf16>
// CHECK:      }
// CHECK:      %{{.*}} = hivm.hir.vabs ins(%[[W]] : tensor<1x1x16x16xf16>)
// CHECK:      %[[RAW:.*]] = hivm.hir.convert_layout %{{.*}} output_shape [16, 16]
// CHECK:      return %[[RAW]], %[[W]] : tensor<16x16xf16>, tensor<1x1x16x16xf16>
func.func @propagate_up_from_while_result_with_raw_user(
  %init: tensor<16x16xf16>, %cond: i1
) -> (tensor<16x16xf16>, tensor<1x1x16x16xf16>) {
  %w = scf.while (%s = %init) : (tensor<16x16xf16>) -> (tensor<16x16xf16>) {
    scf.condition(%cond) %s : tensor<16x16xf16>
  } do {
  ^bb0(%arg: tensor<16x16xf16>):
    %tmp = tensor.empty() : tensor<16x16xf16>
    %mid = hivm.hir.vexp ins(%arg : tensor<16x16xf16>) outs(%tmp : tensor<16x16xf16>) -> tensor<16x16xf16>
    scf.yield %mid : tensor<16x16xf16>
  }

  %w_up = hivm.hir.convert_layout %w output_shape [1, 1, 16, 16]
    {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
     srcLayout = #hivm.data_layout<ND>}
    : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>

  %tmp2 = tensor.empty() : tensor<16x16xf16>
  %raw_user = hivm.hir.vabs ins(%w : tensor<16x16xf16>) outs(%tmp2 : tensor<16x16xf16>) -> tensor<16x16xf16>

  return %raw_user, %w_up : tensor<16x16xf16>, tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_down_from_multi_for_results(
// CHECK:      %[[R:.*]]:2 = scf.for
// CHECK-SAME: iter_args(%[[A:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}})
// CHECK:        %[[A_NEXT:.*]] = hivm.hir.vexp ins(%[[A]] : tensor<1x1x16x16xf16>)
// CHECK:        %[[B_NEXT:.*]] = hivm.hir.vabs ins(%[[B]] : tensor<1x1x16x16xf16>)
// CHECK:        scf.yield %[[A_NEXT]], %[[B_NEXT]]
// CHECK-SAME:     : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
// CHECK:      %[[A_DN:.*]] = hivm.hir.convert_layout %[[R]]#0 output_shape [16, 16]
// CHECK:      %[[B_DN:.*]] = hivm.hir.convert_layout %[[R]]#1 output_shape [16, 16]
// CHECK:      return %{{.*}}, %[[A_DN]], %[[B_DN]]
// CHECK-SAME: : tensor<1xf16>, tensor<16x16xf16>, tensor<16x16xf16>
func.func @propagate_down_from_multi_for_results(
  %init_s: tensor<1xf16>,
  %init_a_fr: tensor<1x1x16x16xf16>,
  %init_b_fr: tensor<1x1x16x16xf16>,
  %lb: index, %ub: index, %step: index
) -> (tensor<1xf16>, tensor<16x16xf16>, tensor<16x16xf16>) {
  %r:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%s = %init_s, %a = %init_a_fr, %b = %init_b_fr)
      -> (tensor<1xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) {
    %tmp_a = tensor.empty() : tensor<1x1x16x16xf16>
    %next_a = hivm.hir.vexp ins(%a : tensor<1x1x16x16xf16>) outs(%tmp_a : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
    %tmp_b = tensor.empty() : tensor<1x1x16x16xf16>
    %next_b = hivm.hir.vabs ins(%b : tensor<1x1x16x16xf16>) outs(%tmp_b : tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
    scf.yield %s, %next_a, %next_b
      : tensor<1xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
  }

  %a_nd = hivm.hir.convert_layout %r#1 output_shape [16, 16]
    {dstLayout = #hivm.data_layout<ND>,
     srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
    : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>
  %b_nd = hivm.hir.convert_layout %r#2 output_shape [16, 16]
    {dstLayout = #hivm.data_layout<ND>,
     srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
    : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>

  return %r#0, %a_nd, %b_nd
    : tensor<1xf16>, tensor<16x16xf16>, tensor<16x16xf16>
}

// -----

// CHECK-LABEL: func.func @propagate_down_from_if_yields_mmad_dependency(
// CHECK:      %[[INIT:.*]] = hivm.hir.mmadL1
// CHECK:      %[[IFR:.*]] = scf.if %{{.*}} -> (tensor<4x4x16x16xf32>)
// CHECK:        %[[THEN:.*]] = hivm.hir.mmadL1
// CHECK:        scf.yield %[[THEN]] : tensor<4x4x16x16xf32>
// CHECK:      } else {
// CHECK:        scf.yield %[[INIT]] : tensor<4x4x16x16xf32>
// CHECK:      }
// CHECK:      %[[DOWN:.*]] = hivm.hir.convert_layout %[[IFR]] output_shape [64, 64]
// CHECK:      return %[[DOWN]] : tensor<64x64xf32>
func.func @propagate_down_from_if_yields_mmad_dependency(
    %lhs0: tensor<2x4x16x16xbf16>,
    %rhs0: tensor<4x2x16x16xbf16>,
    %lhs1: tensor<4x4x16x16xbf16>,
    %rhs1: tensor<4x4x16x16xbf16>,
    %acc: tensor<4x4x16x16xf32>,
    %iv: i32) -> tensor<64x64xf32> {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false

  %init_fr = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true, normalized_in_L0C}
      ins(%lhs0, %rhs0, %true, %c64, %c32, %c64 : tensor<2x4x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index)
      outs(%acc : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
  %init_nd = hivm.hir.convert_layout %init_fr output_shape [64, 64]
      {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<4x4x16x16xf32>) -> tensor<64x64xf32>
  %init_fr2 = hivm.hir.convert_layout %init_nd output_shape [4, 4, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
      : (tensor<64x64xf32>) -> tensor<4x4x16x16xf32>

  %cond = arith.cmpi eq, %iv, %c0_i32 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c0_i32, %alloca[] : memref<i32>
  %r = scf.if %cond -> (tensor<64x64xf32>) {
    %loaded = memref.load %alloca[] : memref<i32>
    %then_fr = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, normalized_in_L0C}
        ins(%lhs1, %rhs1, %false, %c64, %c64, %c64 : tensor<4x4x16x16xbf16>, tensor<4x4x16x16xbf16>, i1, index, index, index)
        outs(%init_fr2 : tensor<4x4x16x16xf32>) -> tensor<4x4x16x16xf32>
    %then_nd = hivm.hir.convert_layout %then_fr output_shape [64, 64]
        {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
        : (tensor<4x4x16x16xf32>) -> tensor<64x64xf32>
    %inc = arith.addi %loaded, %c1_i32 : i32
    memref.store %inc, %alloca[] : memref<i32>
    scf.yield %then_nd : tensor<64x64xf32>
  } else {
    scf.yield %init_nd : tensor<64x64xf32>
  }
  return %r : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @move_convert_layout_to_source_Of_ub_to_cbuf_copy(
// CHECK:      %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (tensor<16x16xf32>)  : i32 {
// CHECK:      %[[ALLOC:.*]] = memref.alloc() : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:      %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable :  memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
// CHECK:      %[[CONVERT_LAYOUT:.*]] = hivm.hir.convert_layout %[[ITER_ARG]] output_shape [2, 1, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, not_to_propagate_up = true, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
// CHECK:      hivm.hir.copy ins(%[[CONVERT_LAYOUT]] : tensor<2x1x16x8xf32>) outs(%[[ALLOC]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>) {"inserted-copy"}
// CHECK:      %{{.*}} = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%[[TO_TENSOR]],

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @move_convert_layout_to_source_Of_ub_to_cbuf_copy(
      %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
      %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
      %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32},
      %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32},
      %arg5: i32, %arg6: i32, %arg7: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true

    %0 = arith.muli %arg5, %arg6 : i32
    %1 = arith.muli %0, %arg7 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc : memref<16x16xf32>) eviction_policy = <EvictFirst> core_type = <VECTOR>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    %alloc_1 = memref.alloc() : memref<16x16xf32>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16x16xf32, strided<[16, 1]>>) outs(%alloc_1 : memref<16x16xf32>) eviction_policy = <EvictFirst> core_type = <CUBE>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16xf32>
    %4 = scf.for %arg8 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %2) -> (tensor<16x16xf32>)  : i32 {
      %alloc_3 = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>>
      %memspacecast = memref.memory_space_cast %alloc_3 : memref<16x16xf32, #hivm.address_space<cbuf>> to memref<16x16xf32>
      %5 = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
      hivm.hir.copy ins(%arg9 : tensor<16x16xf32>) outs(%memspacecast : memref<16x16xf32>) {"inserted-copy"}
      %6 = tensor.empty() : tensor<16x16xf32>
      %7 = hivm.hir.convert_layout %5 output_shape [2, 1, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
      %8 = hivm.hir.convert_layout %3 output_shape [2, 1, 16, 8] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
      %9 = hivm.hir.convert_layout %6 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<16x16xf32>) -> tensor<1x1x16x16xf32>
      %10 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%7, %8, %true, %c16, %c16, %c16 : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>, i1, index, index, index) outs(%9 : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %11 = hivm.hir.convert_layout %10 output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<1x1x16x16xf32>) -> tensor<16x16xf32>
      %alloc_4 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      %memspacecast_5 = memref.memory_space_cast %alloc_4 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
      %12 = bufferization.to_tensor %memspacecast_5 restrict writable : memref<16x16xf32>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, do_not_move_out_of_scffor = true} ins(%11 : tensor<16x16xf32>) outs(%alloc_4 : memref<16x16xf32, #hivm.address_space<ub>>)
      scf.yield %12 :tensor<16x16xf32>
    } {fixpipe_for_mmad_result_already_inserted = true}
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1]>>
    return
  }
}

// -----

// CHECK-LABEL: func.func @propagate_up_through_scalar_vbrc(
// CHECK-SAME:      %[[SCALAR:.*]]: bf16
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<8x8x16x16xbf16>
// CHECK:           %[[VBRC:.*]] = hivm.hir.vbrc ins(%[[SCALAR]] : bf16) outs(%[[EMPTY]] : tensor<8x8x16x16xbf16>) -> tensor<8x8x16x16xbf16>
// CHECK:           return %[[VBRC]] : tensor<8x8x16x16xbf16>
func.func @propagate_up_through_scalar_vbrc(%cst: bf16) -> tensor<8x8x16x16xbf16> {
  %empty = tensor.empty() : tensor<128x128xbf16>
  %brc = hivm.hir.vbrc ins(%cst : bf16) outs(%empty : tensor<128x128xbf16>)
      -> tensor<128x128xbf16>
  %fractal = hivm.hir.convert_layout %brc output_shape [8, 8, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<128x128xbf16>) -> tensor<8x8x16x16xbf16>
  return %fractal : tensor<8x8x16x16xbf16>
}

// -----

// CHECK-LABEL: func.func @propagate_up_through_insert_slice(
// CHECK-SAME:      %[[DEST:.*]]: tensor<128x128xbf16>, %[[SRC:.*]]: tensor<32x128xbf16>
// CHECK:           %[[DEST_FR:.*]] = hivm.hir.convert_layout %[[DEST]] output_shape [8, 8, 16, 16]
// CHECK-SAME:      (tensor<128x128xbf16>) -> tensor<8x8x16x16xbf16>
// CHECK:           %[[SRC_FR:.*]] = hivm.hir.convert_layout %[[SRC]] output_shape [8, 2, 16, 16]
// CHECK-SAME:      (tensor<32x128xbf16>) -> tensor<8x2x16x16xbf16>
// CHECK:           %[[INSERTED:.*]] = tensor.insert_slice %[[SRC_FR]] into %[[DEST_FR]][0, 2, 0, 0] [8, 2, 16, 16] [1, 1, 1, 1]
// CHECK:           return %[[INSERTED]] : tensor<8x8x16x16xbf16>
func.func @propagate_up_through_insert_slice(
    %dest: tensor<128x128xbf16>, %source: tensor<32x128xbf16>
) -> tensor<8x8x16x16xbf16> {
  %inserted = tensor.insert_slice %source into %dest[32, 0] [32, 128] [1, 1]
      : tensor<32x128xbf16> into tensor<128x128xbf16>
  %fractal = hivm.hir.convert_layout %inserted output_shape [8, 8, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<128x128xbf16>) -> tensor<8x8x16x16xbf16>
  return %fractal : tensor<8x8x16x16xbf16>
}

// -----

// CHECK-LABEL: func.func @move_convert_layout_before_tensor_copy(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<16x16xf32>, %[[ARG1:.*]]: tensor<16x16xf32>
// CHECK:           %[[CONVERT:.*]] = hivm.hir.convert_layout %[[ARG0]] output_shape [2, 1, 16, 8]
// CHECK-SAME:      not_to_propagate_up = true
// CHECK-SAME:      (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x1x16x8xf32>
// CHECK:           %[[COPY:.*]] = hivm.hir.copy ins(%[[CONVERT]] : tensor<2x1x16x8xf32>) outs(%[[EMPTY]] : tensor<2x1x16x8xf32>) {"inserted-copy"} -> tensor<2x1x16x8xf32>
// CHECK:           %[[RHS:.*]] = hivm.hir.convert_layout %[[ARG1]] output_shape [2, 1, 16, 8]
// CHECK:           hivm.hir.mmadL1
// CHECK-SAME:      ins(%[[COPY]], %[[RHS]]
func.func @move_convert_layout_before_tensor_copy(
    %arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>)
    -> tensor<1x1x16x16xf32> {
  %true = arith.constant true
  %c16 = arith.constant 16 : index
  %copy_dst = tensor.empty() : tensor<16x16xf32>
  %copied = hivm.hir.copy ins(%arg0 : tensor<16x16xf32>)
      outs(%copy_dst : tensor<16x16xf32>) {"inserted-copy"}
      -> tensor<16x16xf32>
  %lhs = hivm.hir.convert_layout %copied output_shape [2, 1, 16, 8]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
  %rhs = hivm.hir.convert_layout %arg1 output_shape [2, 1, 16, 8]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>,
       srcLayout = #hivm.data_layout<ND>}
      : (tensor<16x16xf32>) -> tensor<2x1x16x8xf32>
  %out = tensor.empty() : tensor<1x1x16x16xf32>
  %res = hivm.hir.mmadL1
      ins(%lhs, %rhs, %true, %c16, %c16, %c16
          : tensor<2x1x16x8xf32>, tensor<2x1x16x8xf32>,
            i1, index, index, index)
      outs(%out : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
  return %res : tensor<1x1x16x16xf32>
}
