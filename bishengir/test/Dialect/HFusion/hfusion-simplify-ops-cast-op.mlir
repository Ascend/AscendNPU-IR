// RUN: bishengir-opt %s -hfusion-simplify-ops 2>&1 | FileCheck %s
//
// Regression: When computing iterarg index for a hfusion.cast whose output
// is defined by tensor.empty and input is a non-first iter_arg of an scf.for,
// for the Nth iterarg it yields -N instead of N, which wraps incorrectly when
// cast to unsigned and trips the assert failure, or an out-of-bounds access
// in a no-assert build.
//
// CHECK-LABEL: func.func @looped_cast_second_iterarg
// CHECK: scf.for
// CHECK: hfusion.cast
// CHECK: linalg.matmul
// CHECK: scf.yield

func.func @looped_cast_second_iterarg(
    %init0: tensor<16x16xf32>,
    %init1: tensor<16x16xf32>,
    %w: tensor<16x16xf16>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %e16 = tensor.empty() : tensor<16x16xf16>
  %r:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%a = %init0, %b = %init1)
      -> (tensor<16x16xf32>, tensor<16x16xf32>) : i32 {
    %cb = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                        round_mode = #hfusion.round_mode<rint>}
        ins(%b : tensor<16x16xf32>) outs(%e16 : tensor<16x16xf16>)
        -> tensor<16x16xf16>
    %mm = linalg.matmul {input_precision = "ieee"}
        ins(%cb, %w : tensor<16x16xf16>, tensor<16x16xf16>)
        outs(%a : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mm, %mm : tensor<16x16xf32>, tensor<16x16xf32>
  }
  return %r#0, %r#1 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----
// The loop carries bf16 around an f32 body. Hoisting the cast pair would
// drop the per-iteration rounding to bf16 on the loop back edge and drift
// against the per-iteration-bf16 reference (issue #16), so the casts must
// stay inside the loop even though the iter_arg has a single use.
//
// CHECK-LABEL: func.func @looped_cast_narrow_carried_kept(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xbf16>,
// CHECK-SAME: %[[X:.*]]: tensor<4xf32>) -> tensor<4xbf16> {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<4xbf16>)
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: scf.yield
// CHECK: return %[[R]] : tensor<4xbf16>
func.func @looped_cast_narrow_carried_kept(
    %init: tensor<4xbf16>, %x: tensor<4xf32>) -> tensor<4xbf16> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xbf16>) : i32 {
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %widen = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                           round_mode = #hfusion.round_mode<rint>}
        ins(%acc : tensor<4xbf16>) outs(%empty_f32 : tensor<4xf32>)
        -> tensor<4xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%widen, %x : tensor<4xf32>, tensor<4xf32>)
        outs(%empty_f32 : tensor<4xf32>) -> tensor<4xf32>
    %empty_bf16 = tensor.empty() : tensor<4xbf16>
    %narrow = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                            round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<4xf32>) outs(%empty_bf16 : tensor<4xbf16>)
        -> tensor<4xbf16>
    scf.yield %narrow : tensor<4xbf16>
  }
  return %r : tensor<4xbf16>
}

// -----
// Same loop with fastmath<contract> on the casts (the shape reported in
// issue #16). contract licenses fusing adjacent floating point operations,
// not dropping the back-edge rounding, so the hoist stays blocked.
//
// CHECK-LABEL: func.func @looped_cast_narrow_carried_contract_kept(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xbf16>,
// CHECK-SAME: %[[X:.*]]: tensor<4xf32>) -> tensor<4xbf16> {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<4xbf16>)
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: scf.yield
// CHECK: return %[[R]] : tensor<4xbf16>
func.func @looped_cast_narrow_carried_contract_kept(
    %init: tensor<4xbf16>, %x: tensor<4xf32>) -> tensor<4xbf16> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xbf16>) : i32 {
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %widen = hfusion.cast {arith.fastmath = #arith.fastmath<contract>,
                           cast = #hfusion.type_fn<cast_signed>,
                           enable_overflow = true,
                           round_mode = #hfusion.round_mode<rint>}
        ins(%acc : tensor<4xbf16>) outs(%empty_f32 : tensor<4xf32>)
        -> tensor<4xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%widen, %x : tensor<4xf32>, tensor<4xf32>)
        outs(%empty_f32 : tensor<4xf32>) -> tensor<4xf32>
    %empty_bf16 = tensor.empty() : tensor<4xbf16>
    %narrow = hfusion.cast {arith.fastmath = #arith.fastmath<contract>,
                            cast = #hfusion.type_fn<cast_signed>,
                            enable_overflow = true,
                            round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<4xf32>) outs(%empty_bf16 : tensor<4xbf16>)
        -> tensor<4xbf16>
    scf.yield %narrow : tensor<4xbf16>
  }
  return %r : tensor<4xbf16>
}

// -----
// The loop carries f32 around a bf16 body. Every bf16 value is exactly
// representable in f32, so the back-edge round trip is the identity and the
// cast pair may be hoisted: the entry cast moves before the loop, the exit
// cast behind it, and the loop carries bf16.
//
// CHECK-LABEL: func.func @looped_cast_wider_carried_hoisted(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xf32>,
// CHECK-SAME: %[[X:.*]]: tensor<4xbf16>) -> tensor<4xf32> {
// CHECK: %[[NARROW:.*]] = hfusion.cast {{.*}} ins(%[[INIT]] : tensor<4xf32>) outs({{.*}} : tensor<4xbf16>) -> tensor<4xbf16>
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[NARROW]]) -> (tensor<4xbf16>)
// CHECK: linalg.elemwise_binary
// CHECK: scf.yield
// CHECK: %[[WIDEN:.*]] = hfusion.cast {{.*}} ins(%[[R]] : tensor<4xbf16>) outs({{.*}} : tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[WIDEN]] : tensor<4xf32>
func.func @looped_cast_wider_carried_hoisted(
    %init: tensor<4xf32>, %x: tensor<4xbf16>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xf32>) : i32 {
    %empty_bf16 = tensor.empty() : tensor<4xbf16>
    %narrow = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                            round_mode = #hfusion.round_mode<rint>}
        ins(%acc : tensor<4xf32>) outs(%empty_bf16 : tensor<4xbf16>)
        -> tensor<4xbf16>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%narrow, %x : tensor<4xbf16>, tensor<4xbf16>)
        outs(%empty_bf16 : tensor<4xbf16>) -> tensor<4xbf16>
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %widen = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                           round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<4xbf16>) outs(%empty_f32 : tensor<4xf32>)
        -> tensor<4xf32>
    scf.yield %widen : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----
// f16 carries fewer mantissa bits than f32, so a carried f16 cannot
// represent every f32 body value and the back-edge round trip is lossy;
// keep the casts inside the loop. (The pure exponent-range counterexample,
// f16 carried around a bf16 body, cannot even be expressed here: an
// f16 <-> bf16 hfusion.cast fails cast verification at HEAD.)
//
// CHECK-LABEL: func.func @looped_cast_f16_carried_f32_body_kept(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xf16>,
// CHECK-SAME: %[[X:.*]]: tensor<4xf32>) -> tensor<4xf16> {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<4xf16>)
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: scf.yield
// CHECK: return %[[R]] : tensor<4xf16>
func.func @looped_cast_f16_carried_f32_body_kept(
    %init: tensor<4xf16>, %x: tensor<4xf32>) -> tensor<4xf16> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xf16>) : i32 {
    %empty_f32 = tensor.empty() : tensor<4xf32>
    %widen = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                           round_mode = #hfusion.round_mode<rint>}
        ins(%acc : tensor<4xf16>) outs(%empty_f32 : tensor<4xf32>)
        -> tensor<4xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%widen, %x : tensor<4xf32>, tensor<4xf32>)
        outs(%empty_f32 : tensor<4xf32>) -> tensor<4xf32>
    %empty_f16 = tensor.empty() : tensor<4xf16>
    %narrow = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                            round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<4xf32>) outs(%empty_f16 : tensor<4xf16>)
        -> tensor<4xf16>
    scf.yield %narrow : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}

// -----
// Integer narrowing on the back edge truncates every iteration; a wider body
// type accumulates bits the carried type would have dropped. Keep the casts
// inside the loop.
//
// CHECK-LABEL: func.func @looped_cast_integer_narrow_carried_kept(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xi8>,
// CHECK-SAME: %[[X:.*]]: tensor<4xi32>) -> tensor<4xi8> {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<4xi8>)
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: scf.yield
// CHECK: return %[[R]] : tensor<4xi8>
func.func @looped_cast_integer_narrow_carried_kept(
    %init: tensor<4xi8>, %x: tensor<4xi32>) -> tensor<4xi8> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xi8>) : i32 {
    %empty_i32 = tensor.empty() : tensor<4xi32>
    %widen = hfusion.cast ins(%acc : tensor<4xi8>)
        outs(%empty_i32 : tensor<4xi32>) -> tensor<4xi32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%widen, %x : tensor<4xi32>, tensor<4xi32>)
        outs(%empty_i32 : tensor<4xi32>) -> tensor<4xi32>
    %empty_i8 = tensor.empty() : tensor<4xi8>
    %narrow = hfusion.cast ins(%sum : tensor<4xi32>)
        outs(%empty_i8 : tensor<4xi8>) -> tensor<4xi8>
    scf.yield %narrow : tensor<4xi8>
  }
  return %r : tensor<4xi8>
}

// -----
// Integer widening round trip is exact, so the redundant i8 -> i32 -> i8
// pair around the loop body may be hoisted.
//
// CHECK-LABEL: func.func @looped_cast_integer_wider_carried_hoisted(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xi32>,
// CHECK-SAME: %[[X:.*]]: tensor<4xi8>) -> tensor<4xi32> {
// CHECK: %[[NARROW:.*]] = hfusion.cast{{.*}} ins(%[[INIT]] : tensor<4xi32>) outs({{.*}} : tensor<4xi8>) -> tensor<4xi8>
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[NARROW]]) -> (tensor<4xi8>)
// CHECK: linalg.elemwise_binary
// CHECK: scf.yield
// CHECK: %[[WIDEN:.*]] = hfusion.cast{{.*}} ins(%[[R]] : tensor<4xi8>) outs({{.*}} : tensor<4xi32>) -> tensor<4xi32>
// CHECK: return %[[WIDEN]] : tensor<4xi32>
func.func @looped_cast_integer_wider_carried_hoisted(
    %init: tensor<4xi32>, %x: tensor<4xi8>) -> tensor<4xi32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xi32>) : i32 {
    %empty_i8 = tensor.empty() : tensor<4xi8>
    %narrow = hfusion.cast ins(%acc : tensor<4xi32>)
        outs(%empty_i8 : tensor<4xi8>) -> tensor<4xi8>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%narrow, %x : tensor<4xi8>, tensor<4xi8>)
        outs(%empty_i8 : tensor<4xi8>) -> tensor<4xi8>
    %empty_i32 = tensor.empty() : tensor<4xi32>
    %widen = hfusion.cast ins(%sum : tensor<4xi8>)
        outs(%empty_i32 : tensor<4xi32>) -> tensor<4xi32>
    scf.yield %widen : tensor<4xi32>
  }
  return %r : tensor<4xi32>
}

// -----
// A cast with identical input and output type represents an intended change
// in value (rint rounding here); hoisting it out would round only once after
// the loop instead of on every iteration.
//
// CHECK-LABEL: func.func @looped_cast_same_type_kept(
// CHECK-SAME: %[[INIT:.*]]: tensor<4xf32>,
// CHECK-SAME: %[[X:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<4xf32>)
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: scf.yield
// CHECK: return %[[R]] : tensor<4xf32>
func.func @looped_cast_same_type_kept(
    %init: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%acc = %init) -> (tensor<4xf32>) : i32 {
    %empty = tensor.empty() : tensor<4xf32>
    %rin = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
        ins(%acc : tensor<4xf32>) outs(%empty : tensor<4xf32>)
        -> tensor<4xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%rin, %x : tensor<4xf32>, tensor<4xf32>)
        outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %empty2 = tensor.empty() : tensor<4xf32>
    %rout = hfusion.cast {round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<4xf32>) outs(%empty2 : tensor<4xf32>)
        -> tensor<4xf32>
    scf.yield %rout : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----
// Scalar iter_arg form: the loop inserts the carried scalar into a tensor
// before casting, so the cast input is not the iter_arg itself and nothing
// may be hoisted (regression coverage for the pre-tensor-iter_arg form of
// issue #16).
//
// CHECK-LABEL: func.func @looped_cast_scalar_iterarg_kept(
// CHECK-SAME: %[[INIT:.*]]: bf16,
// CHECK-SAME: %[[T:.*]]: tensor<1xbf16>) -> bf16 {
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (bf16)
// CHECK: tensor.insert
// CHECK: hfusion.cast
// CHECK: linalg.elemwise_binary
// CHECK: hfusion.cast
// CHECK: tensor.extract
// CHECK: scf.yield
// CHECK: return %[[R]] : bf16
func.func @looped_cast_scalar_iterarg_kept(
    %init: bf16, %t: tensor<1xbf16>) -> bf16 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c0_idx = arith.constant 0 : index
  %empty_f32 = tensor.empty() : tensor<1xf32>
  %base_f32 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                            round_mode = #hfusion.round_mode<rint>}
      ins(%t : tensor<1xbf16>) outs(%empty_f32 : tensor<1xf32>)
      -> tensor<1xf32>
  %r = scf.for %iv = %c0 to %c3 step %c1
      iter_args(%acc = %init) -> (bf16) : i32 {
    %inserted = tensor.insert %acc into %t[%c0_idx] : tensor<1xbf16>
    %empty_in = tensor.empty() : tensor<1xf32>
    %widen = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                           round_mode = #hfusion.round_mode<rint>}
        ins(%inserted : tensor<1xbf16>) outs(%empty_in : tensor<1xf32>)
        -> tensor<1xf32>
    %sum = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%widen, %base_f32 : tensor<1xf32>, tensor<1xf32>)
        outs(%empty_in : tensor<1xf32>) -> tensor<1xf32>
    %empty_bf16 = tensor.empty() : tensor<1xbf16>
    %narrow = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                            round_mode = #hfusion.round_mode<rint>}
        ins(%sum : tensor<1xf32>) outs(%empty_bf16 : tensor<1xbf16>)
        -> tensor<1xbf16>
    %extracted = tensor.extract %narrow[%c0_idx] : tensor<1xbf16>
    scf.yield %extracted : bf16
  }
  return %r : bf16
}


// -----
//
// Regression (#961): `f32 -> f8E4M3FN -> f32` is not the identity. The fp8 hop
// saturates to +-448 and rounds the significand to 3 bits, so folding the pair
// away deletes the quantization. getPrecisionRank used to return -1 for fp8,
// which made hasPrecisionDownUpPattern skip both hops and let CastOpPattern
// remove the whole chain.
//
// CHECK-LABEL: func.func @keep_fp8_roundtrip
// CHECK: hfusion.cast
// CHECK-SAME: tensor<16x32xf32>) outs({{.*}} : tensor<16x32xf8E4M3FN>)
// CHECK: hfusion.cast
// CHECK-SAME: tensor<16x32xf8E4M3FN>) outs({{.*}} : tensor<16x32xf32>)

func.func @keep_fp8_roundtrip(%src: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %f8 = tensor.empty() : tensor<16x32xf8E4M3FN>
  %f32 = tensor.empty() : tensor<16x32xf32>
  %q = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                     round_mode = #hfusion.round_mode<rint>}
      ins(%src : tensor<16x32xf32>) outs(%f8 : tensor<16x32xf8E4M3FN>)
      -> tensor<16x32xf8E4M3FN>
  %dq = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                      round_mode = #hfusion.round_mode<rint>}
      ins(%q : tensor<16x32xf8E4M3FN>) outs(%f32 : tensor<16x32xf32>)
      -> tensor<16x32xf32>
  return %dq : tensor<16x32xf32>
}

// -----
//
// Counter-case: a single `f32 -> f16` cast is not a round trip and must keep
// being simplified as before, i.e. this test only guards against over-blocking
// in getCastChainBlockReason.
//
// CHECK-LABEL: func.func @single_downcast_is_not_a_chain
// CHECK: hfusion.cast

func.func @single_downcast_is_not_a_chain(%src: tensor<16x32xf32>) -> tensor<16x32xf16> {
  %f16 = tensor.empty() : tensor<16x32xf16>
  %r = hfusion.cast {cast = #hfusion.type_fn<cast_signed>,
                     round_mode = #hfusion.round_mode<rint>}
      ins(%src : tensor<16x32xf32>) outs(%f16 : tensor<16x32xf16>)
      -> tensor<16x32xf16>
  return %r : tensor<16x32xf16>
}
