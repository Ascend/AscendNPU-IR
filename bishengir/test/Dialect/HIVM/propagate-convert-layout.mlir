// RUN: bishengir-opt --hivm-propagate-convert-layout="allow-agnostic-ops=true" --canonicalize %s | FileCheck %s

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
