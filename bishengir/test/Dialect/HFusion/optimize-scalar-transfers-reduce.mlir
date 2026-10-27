// RUN: bishengir-opt %s --allow-unregistered-dialect --hfusion-optimize-scalar-transfers --symbol-dce --cse --split-input-file | FileCheck %s

// Tests for lifting the singleton dimension of a zero-rank `linalg.reduce`
// result into the reduce itself. When the lifted extract reads a zero-rank
// tensor produced by a `linalg.reduce` that reduces all of its input
// dimensions, the pass rewrites the reduce so that it produces
// `tensor<1 x elem>` directly instead of inserting a
// `tensor.expand_shape` on the reduce result:
//
//   %r = linalg.reduce ins(%a : tensor<NxT>) outs(%i : tensor<T>)
//        dimensions = [0]
//   %e = tensor.extract %r[] : tensor<T>
//
// becomes
//
//   %a' = tensor.expand_shape %a [[0, 1]] : tensor<NxT> into tensor<1xNxT>
//   %i' = tensor.expand_shape %i [] : tensor<T> into tensor<1xT>
//   %r' = linalg.reduce ins(%a' : tensor<1xNxT>) outs(%i' : tensor<1xT>)
//        dimensions = [1]
//   %e' = tensor.extract %r'[%c0] : tensor<1xT>  (then lifted to identity)

// Positive: the reduce of a one-dimensional f32 tensor is lifted to a
// singleton-producing reduce.
// CHECK-LABEL: func.func @reduce_lift_f32(
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 2048] : tensor<2048xf32> into tensor<1x2048xf32>
// CHECK: %[[INEXP:.*]] = tensor.expand_shape {{.*}} [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x2048xf32>) outs(%[[INEXP]] : tensor<1xf32>) dimensions = [1]
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return %[[REDUCE]] : tensor<1xf32>
module {
  func.func @reduce_lift_f32(%arg0: tensor<2048xf32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2048xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %empty1 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %extracted into %empty1[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}

// -----

// Positive: f16 element type, the scalar chain after the extract is lifted
// together with the reduce.
// CHECK-LABEL: func.func @reduce_lift_f16(
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 1024] : tensor<1024xf16> into tensor<1x1024xf16>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x1024xf16>) outs({{.*}} : tensor<1xf16>) dimensions = [1]
// CHECK: %[[ADD:.*]] = arith.addf %[[REDUCE]], %[[REDUCE]] : tensor<1xf16>
// CHECK-NOT: tensor.extract
// CHECK: return %[[ADD]] : tensor<1xf16>
module {
  func.func @reduce_lift_f16(%arg0: tensor<1024xf16>) -> tensor<1xf16> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f16>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xf16>) outs(%empty0 : tensor<f16>) dimensions = [0]
      (%in: f16, %init: f16) {
        %s = arith.addf %in, %init : f16
        linalg.yield %s : f16
      }
    %extracted = tensor.extract %reduced[] : tensor<f16>
    %add = arith.addf %extracted, %extracted : f16
    %empty1 = tensor.empty() : tensor<1xf16>
    %inserted = tensor.insert %add into %empty1[%c0] : tensor<1xf16>
    return %inserted : tensor<1xf16>
  }
}

// -----

// Positive: 1D reduce normalized besides its numel
// CHECK-LABEL: func.func @reduce_lift_small_shape(
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 256] : tensor<256xf16> into tensor<1x256xf16>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x256xf16>) outs({{.*}} : tensor<1xf16>) dimensions = [1]
// CHECK: %[[ADD:.*]] = arith.addf %[[REDUCE]], %[[REDUCE]] : tensor<1xf16>
// CHECK-NOT: tensor.extract
// CHECK: return %[[ADD]] : tensor<1xf16>
module {
  func.func @reduce_lift_small_shape(%arg0: tensor<256xf16>) -> tensor<1xf16> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f16>
    %reduced = linalg.reduce ins(%arg0 : tensor<256xf16>) outs(%empty0 : tensor<f16>) dimensions = [0]
      (%in: f16, %init: f16) {
        %s = arith.addf %in, %init : f16
        linalg.yield %s : f16
      }
    %extracted = tensor.extract %reduced[] : tensor<f16>
    %add = arith.addf %extracted, %extracted : f16
    %empty1 = tensor.empty() : tensor<1xf16>
    %inserted = tensor.insert %add into %empty1[%c0] : tensor<1xf16>
    return %inserted : tensor<1xf16>
  }
}

// -----

// Positive: bfloat16 element type.
// CHECK-LABEL: func.func @reduce_lift_bf16(
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 1024] : tensor<1024xbf16> into tensor<1x1024xbf16>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x1024xbf16>) outs({{.*}} : tensor<1xbf16>) dimensions = [1]
// CHECK-NOT: tensor.extract
// CHECK: return %[[REDUCE]] : tensor<1xbf16>
module {
  func.func @reduce_lift_bf16(%arg0: tensor<1024xbf16>) -> tensor<1xbf16> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<bf16>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xbf16>) outs(%empty0 : tensor<bf16>) dimensions = [0]
      (%in: bf16, %init: bf16) {
        %m = arith.maxnumf %in, %init : bf16
        linalg.yield %m : bf16
      }
    %extracted = tensor.extract %reduced[] : tensor<bf16>
    %empty1 = tensor.empty() : tensor<1xbf16>
    %inserted = tensor.insert %extracted into %empty1[%c0] : tensor<1xbf16>
    return %inserted : tensor<1xbf16>
  }
}

// -----

// Positive: a longer scalar chain after the extract is lifted together with
// the reduce; the non-intermediate constant operand is adapted into a
// singleton tensor.
// CHECK-LABEL: func.func @reduce_lift_chain(
// CHECK: %[[CST:.*]] = arith.constant 3.000000e+00 : f32
// CHECK: %[[CSTT:.*]] = linalg.fill ins(%[[CST]] : f32) outs({{.*}} : tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[EXP:.*]] = tensor.expand_shape %arg0 {{\[\[}}0, 1]] output_shape [1, 2048] : tensor<2048xf32> into tensor<1x2048xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x2048xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
// CHECK: %[[ADD:.*]] = arith.addf %[[REDUCE]], %[[CSTT]] : tensor<1xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[ADD]] : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK: return %[[MUL]] : tensor<1xf32>
module {
  func.func @reduce_lift_chain(%arg0: tensor<2048xf32>) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.0 : f32
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2048xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %add = arith.addf %extracted, %cst : f32
    %mul = arith.mulf %add, %add : f32
    %empty1 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %mul into %empty1[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}

// -----

// Positive: the reduce over a loop-carried tensor is rewritten in place
// inside the loop body.
// CHECK-LABEL: func.func @reduce_lift_in_loop(
// CHECK: %[[FOR:.*]]:2 = scf.for {{.*}} iter_args({{.*}}, {{.*}}) -> (tensor<2048xf32>, tensor<1xf32>) {
// CHECK: %[[EXP:.*]] = tensor.expand_shape {{.*}} {{\[\[}}0, 1]] output_shape [1, 2048] : tensor<2048xf32> into tensor<1x2048xf32>
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%[[EXP]] : tensor<1x2048xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
// CHECK: %[[ADD:.*]] = arith.addf %[[REDUCE]], %[[REDUCE]] : tensor<1xf32>
// CHECK-NOT: tensor.extract
// CHECK: scf.yield {{.*}}, %[[ADD]] : tensor<2048xf32>, tensor<1xf32>
// CHECK: }
// CHECK: return %[[FOR]]#1 : tensor<1xf32>
module {
  func.func @reduce_lift_in_loop(%arg: tensor<2048xf32>, %lb: index, %ub: index, %step: index) -> tensor<1xf32> attributes { mix_mode = "aiv" } {
    %cst = arith.constant 2.0 : f32
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %init0 = tensor.empty() : tensor<f32>
    %result:2 = scf.for %i = %lb to %ub step %step iter_args(%buf = %arg, %acc = %init) -> (tensor<2048xf32>, tensor<1xf32>) {
      %reduced = linalg.reduce ins(%buf : tensor<2048xf32>) outs(%init0 : tensor<f32>) dimensions = [0]
        (%in: f32, %initr: f32) {
          %m = arith.maxnumf %in, %initr : f32
          linalg.yield %m : f32
        }
      %extracted = tensor.extract %reduced[] : tensor<f32>
      %add = arith.addf %extracted, %extracted : f32
      %empty1 = tensor.empty() : tensor<1xf32>
      %inserted = tensor.insert %add into %empty1[%c0] : tensor<1xf32>
      scf.yield %buf, %inserted : tensor<2048xf32>, tensor<1xf32>
    }
    return %result#1 : tensor<1xf32>
  }
}

// -----

// Negative: index semantics keep the reduce result type unchanged.
// CHECK-LABEL: func.func @reduce_no_lift_index_semantics(
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: linalg.reduce ins(%arg0 : tensor<4xi32>) outs({{.*}} : tensor<i32>) dimensions = [0]
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.insert
// CHECK: return {{.*}} : tensor<1xi32>
module {
  func.func @reduce_no_lift_index_semantics(%arg0: tensor<4xi32>) -> tensor<1xi32> attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %empty0 = tensor.empty() : tensor<i32>
    %init = linalg.fill ins(%zero : i32) outs(%empty0 : tensor<i32>) -> tensor<i32>
    %reduced = linalg.reduce ins(%arg0 : tensor<4xi32>) outs(%init : tensor<i32>) dimensions = [0]
      (%in: i32, %acc: i32) {
        %idx = linalg.index 0 : index
        %coord = arith.index_cast %idx : index to i32
        %sum = arith.addi %acc, %coord : i32
        linalg.yield %sum : i32
      }
    %extracted = tensor.extract %reduced[] : tensor<i32>
    %empty1 = tensor.empty() : tensor<1xi32>
    %inserted = tensor.insert %extracted into %empty1[%c0] : tensor<1xi32>
    return %inserted : tensor<1xi32>
  }
}

// -----

// Negative: the reduce result has two users; the reduce is left untouched
// and the lifted extract falls back to an expand_shape of the reduce result.
// CHECK-LABEL: func.func @reduce_no_lift_multi_use(
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs({{.*}} : tensor<f32>) dimensions = [0]
// CHECK: %[[EXP:.*]] = tensor.expand_shape %[[REDUCE]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[E2:.*]] = tensor.extract %[[REDUCE]][] : tensor<f32>
// CHECK: %[[ADD:.*]] = arith.addf %[[EXP]], %[[EXP]] : tensor<1xf32>
// CHECK: %[[B:.*]] = arith.addf %[[E2]], %[[E2]] : f32
// CHECK: return %[[ADD]], %[[B]] : tensor<1xf32>, f32
module {
  func.func @reduce_no_lift_multi_use(%arg0: tensor<1024xf32>) -> (tensor<1xf32>, f32) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted1 = tensor.extract %reduced[] : tensor<f32>
    %extracted2 = tensor.extract %reduced[] : tensor<f32>
    %add = arith.addf %extracted1, %extracted1 : f32
    %empty1 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %add into %empty1[%c0] : tensor<1xf32>
    %side = arith.addf %extracted2, %extracted2 : f32
    return %inserted, %side : tensor<1xf32>, f32
  }
}

// -----

// Negative: the reduce result is returned directly, so it keeps its
// zero-rank type; the lifted extract falls back to an expand_shape of the
// reduce result.
// CHECK-LABEL: func.func @reduce_no_lift_result_returned(
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs({{.*}} : tensor<f32>) dimensions = [0]
// CHECK: %[[EXP:.*]] = tensor.expand_shape %[[REDUCE]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[EXP]], %[[EXP]] : tensor<1xf32>
// CHECK: return %[[ADD]], %[[REDUCE]] : tensor<1xf32>, tensor<f32>
module {
  func.func @reduce_no_lift_result_returned(%arg0: tensor<1024xf32>) -> (tensor<1xf32>, tensor<f32>) attributes { mix_mode = "aiv" } {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %add = arith.addf %extracted, %extracted : f32
    %empty1 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %add into %empty1[%c0] : tensor<1xf32>
    return %inserted, %reduced : tensor<1xf32>, tensor<f32>
  }
}

// -----

// Negative: the function is not an AIV function, so the pass is skipped
// entirely and the reduce stays zero-rank.
// CHECK-LABEL: func.func @reduce_no_lift_not_aiv(
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%arg0 : tensor<2048xf32>) outs({{.*}} : tensor<f32>) dimensions = [0]
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[REDUCE]][] : tensor<f32>
// CHECK: tensor.insert
module {
  func.func @reduce_no_lift_not_aiv(%arg0: tensor<2048xf32>) -> tensor<1xf32> {
    %c0 = arith.constant 0 : index
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<2048xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %empty1 = tensor.empty() : tensor<1xf32>
    %inserted = tensor.insert %extracted into %empty1[%c0] : tensor<1xf32>
    return %inserted : tensor<1xf32>
  }
}

// -----

// Negative: the extracted scalar is only consumed in a scalar context, so
// the chain is not an intermediate transfer and nothing is rewritten.
// CHECK-LABEL: func.func @reduce_no_lift_scalar_only(
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[REDUCE:.*]] = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs({{.*}} : tensor<f32>) dimensions = [0]
// CHECK: %[[EXTRACT:.*]] = tensor.extract %[[REDUCE]][] : tensor<f32>
// CHECK: %[[ADD:.*]] = arith.addf %[[EXTRACT]], %[[EXTRACT]] : f32
// CHECK: return %[[ADD]] : f32
module {
  func.func @reduce_no_lift_scalar_only(%arg0: tensor<1024xf32>) -> f32 attributes { mix_mode = "aiv" } {
    %empty0 = tensor.empty() : tensor<f32>
    %reduced = linalg.reduce ins(%arg0 : tensor<1024xf32>) outs(%empty0 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.maxnumf %in, %init : f32
        linalg.yield %m : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %add = arith.addf %extracted, %extracted : f32
    return %add : f32
  }
}
