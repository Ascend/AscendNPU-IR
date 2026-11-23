// RUN: bishengir-opt %s --hfusion-flatten-reduction-regions --canonicalize --cse --split-input-file | FileCheck %s
// RUN: bishengir-opt %s --hfusion-flatten-reduction-regions --canonicalize --cse --hfusion-flatten-reduction-regions --canonicalize --cse --split-input-file | FileCheck %s

// Share a cast across two reductions. The square is already fused into its
// reduction by PreVectorizationFusion and must not become a tensor temporary.
// Collapse the shared cast once and keep both reductions reading its result.
// CHECK-LABEL: func.func @shared_cast(
// CHECK: %[[X:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<6x80x8xf16> into tensor<6x640xf16>
// CHECK: %[[CAST:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: ins(%[[X]] : tensor<6x640xf16>)
// CHECK: arith.extf
// CHECK: } -> tensor<6x640xf32>
// CHECK-NOT: arith.extf
// CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[CAST]] : tensor<6x640xf32>) outs(%{{.*}} : tensor<6xf32>)
// CHECK-NOT: arith.extf
// CHECK: arith.addf
// CHECK-NOT: arith.extf
// CHECK: %[[SQUARE_SUM:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[CAST]] : tensor<6x640xf32>) outs(%{{.*}} : tensor<6xf32>)
// CHECK-NOT: arith.extf
// CHECK: arith.mulf
// CHECK-NOT: arith.extf
// CHECK: arith.addf
// CHECK-NOT: arith.extf
// CHECK-NOT: tensor.expand_shape
// CHECK: return %[[SUM]], %[[SQUARE_SUM]] : tensor<6xf32>, tensor<6xf32>
func.func @shared_cast(%x: tensor<6x80x8xf16>, %init: tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
  %empty = tensor.empty() : tensor<6x80x8xf32>
  %cast = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<6x80x8xf16>) outs(%empty : tensor<6x80x8xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      linalg.yield %c : f32
  } -> tensor<6x80x8xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%cast : tensor<6x80x8xf32>) outs(%init : tensor<6xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<6xf32>
  %square_sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%cast : tensor<6x80x8xf32>) outs(%init : tensor<6xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %square = arith.mulf %v, %v : f32
      %s = arith.addf %square, %acc : f32
      linalg.yield %s : f32
  } -> tensor<6xf32>
  return %sum, %square_sum : tensor<6xf32>, tensor<6xf32>
}

// -----

// Keep the shared cast separate when the collapsed reduction group is followed
// by a parallel dimension as well.
// CHECK-LABEL: func.func @shared_cast_middle_reduction(
// CHECK: %[[CAST:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: arith.extf
// CHECK: } -> tensor<2x15x7xf32>
// CHECK: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[CAST]] : tensor<2x15x7xf32>)
// CHECK: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[CAST]] : tensor<2x15x7xf32>)
func.func @shared_cast_middle_reduction(%x: tensor<2x3x5x7xf16>, %init: tensor<2x7xf32>) -> (tensor<2x7xf32>, tensor<2x7xf32>) {
  %empty = tensor.empty() : tensor<2x3x5x7xf32>
  %cast = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b,m,k,n)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5x7xf16>) outs(%empty : tensor<2x3x5x7xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      linalg.yield %c : f32
  } -> tensor<2x3x5x7xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b,n)>], iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%cast : tensor<2x3x5x7xf32>) outs(%init : tensor<2x7xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2x7xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b,n)>], iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%cast : tensor<2x3x5x7xf32>) outs(%init : tensor<2x7xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %m = arith.maximumf %v, %acc : f32
      linalg.yield %m : f32
  } -> tensor<2x7xf32>
  return %sum, %max : tensor<2x7xf32>, tensor<2x7xf32>
}

// -----

// A shared producer with multiple payload operations also stays separate.
// CHECK-LABEL: func.func @shared_multi_op_producer(
// CHECK: %[[SQUARE:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: } -> tensor<2x15xf32>
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[SQUARE]] : tensor<2x15xf32>)
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[SQUARE]] : tensor<2x15xf32>)
func.func @shared_multi_op_producer(%x: tensor<2x3x5xf16>, %init: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %square = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf16>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      %s = arith.mulf %c, %c : f32
      linalg.yield %s : f32
  } -> tensor<2x3x5xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%square : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%square : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %m = arith.maximumf %v, %acc : f32
      linalg.yield %m : f32
  } -> tensor<2xf32>
  return %sum, %max : tensor<2xf32>, tensor<2xf32>
}

// -----

// Do not duplicate a producer with an outside user. Its ancestors also stay
// three-dimensional; only the reduction consumes a collapsed boundary view.
// CHECK-LABEL: func.func @outside_user(
// CHECK-COUNT-2: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: tensor.collapse_shape %[[PRODUCER:.*]] {{\[\[}}0], [1, 2]] : tensor<2x5x7xf32> into tensor<2x35xf32>
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: tensor<2x35xf32>
// CHECK: return %[[PRODUCER]], %{{.*}} : tensor<2x5x7xf32>, tensor<2xf32>
func.func @outside_user(%x: tensor<2x5x7xf32>, %init: tensor<2xf32>) -> (tensor<2x5x7xf32>, tensor<2xf32>) {
  %empty = tensor.empty() : tensor<2x5x7xf32>
  %neg = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x5x7xf32>) outs(%empty : tensor<2x5x7xf32>) {
    ^bb0(%v: f32, %out: f32):
      %n = arith.negf %v : f32
      linalg.yield %n : f32
  } -> tensor<2x5x7xf32>
  %abs = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%neg : tensor<2x5x7xf32>) outs(%empty : tensor<2x5x7xf32>) {
    ^bb0(%v: f32, %out: f32):
      %a = math.absf %v : f32
      linalg.yield %a : f32
  } -> tensor<2x5x7xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%abs : tensor<2x5x7xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %abs, %sum : tensor<2x5x7xf32>, tensor<2xf32>
}

// -----

// Preserve every parallel dimension and support a reduction in the middle,
// multiple inputs/results, and nonzero, distinct initialization tensors.
// CHECK-LABEL: func.func @variadic_middle(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3]] : tensor<2x3x5x7xi32> into tensor<2x15x7xi32>
// CHECK: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: outs(%{{.*}}, %{{.*}} : tensor<2x7xi32>, tensor<2x7xi32>)
// CHECK: arith.addi
// CHECK: arith.maxsi
// CHECK: return %{{.*}}#0, %{{.*}}#1 : tensor<2x7xi32>, tensor<2x7xi32>
func.func @variadic_middle(%x: tensor<2x3x5x7xi32>, %y: tensor<2x3x5x7xi32>, %sum_init: tensor<2x7xi32>, %max_init: tensor<2x7xi32>) -> (tensor<2x7xi32>, tensor<2x7xi32>) {
  %r:2 = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b,n)>, affine_map<(b,m,k,n)->(b,n)>], iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%x, %y : tensor<2x3x5x7xi32>, tensor<2x3x5x7xi32>) outs(%sum_init, %max_init : tensor<2x7xi32>, tensor<2x7xi32>) {
    ^bb0(%vx: i32, %vy: i32, %s: i32, %m: i32):
      %sum = arith.addi %vx, %s : i32
      %max = arith.maxsi %vy, %m : i32
      linalg.yield %sum, %max : i32, i32
  } -> (tensor<2x7xi32>, tensor<2x7xi32>)
  return %r#0, %r#1 : tensor<2x7xi32>, tensor<2x7xi32>
}

// -----

// CHECK-LABEL: func.func @dynamic_extent(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<?x?x8xf32> into tensor<?x?xf32>
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: tensor<?x?xf32>
func.func @dynamic_extent(%x: tensor<?x?x8xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<?x?x8xf32>) outs(%init : tensor<?xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// Two dynamic reduction dimensions must not be collapsed together.
// CHECK-LABEL: func.func @two_dynamic_extents(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @two_dynamic_extents(%x: tensor<2x?x?xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x?x?xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// Parallel dimensions, including unit extents, are never merged into a
// reduction group just because the storage is contiguous.
// CHECK-LABEL: func.func @separated_reductions(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["reduction", "parallel", "reduction"]
func.func @separated_reductions(%x: tensor<3x1x5xf32>, %init: tensor<1xf32>) -> tensor<1xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(m,b,k)->(m,b,k)>, affine_map<(m,b,k)->(b)>], iterator_types = ["reduction", "parallel", "reduction"]}
      ins(%x : tensor<3x1x5xf32>) outs(%init : tensor<1xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<1xf32>
  return %r : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @index_dependent(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK: linalg.index 1
func.func @index_dependent(%x: tensor<2x3x5xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xi32>) outs(%init : tensor<2xi32>) {
    ^bb0(%v: i32, %acc: i32):
      %i = linalg.index 1 : index
      %j = arith.index_cast %i : index to i32
      %a = arith.addi %v, %j : i32
      %s = arith.addi %a, %acc : i32
      linalg.yield %s : i32
  } -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// -----

// A broadcast already fused into a reduction needs quotient/remainder indexing
// after flattening. Preserve that generic instead of forcing identity maps.
// CHECK-LABEL: func.func @projected_input(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @projected_input(%x: tensor<2x3x5xf32>, %scale: tensor<3xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(m)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x, %scale : tensor<2x3x5xf32>, tensor<3xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %scale_v: f32, %acc: f32):
      %p = arith.mulf %v, %scale_v : f32
      %s = arith.addf %p, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// A conflicting sibling reduction keeps the common producer outside both
// regions. Its type and its other users must remain valid.
// CHECK-LABEL: func.func @conflicting_users(
// CHECK: %[[CAST:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: tensor.collapse_shape %[[CAST]] {{\[\[}}0], [1, 2]]
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: iterator_types = ["reduction", "parallel", "reduction"]
func.func @conflicting_users(%x: tensor<2x3x5xf16>, %init_b: tensor<2xf32>, %init_m: tensor<3xf32>) -> (tensor<2xf32>, tensor<3xf32>) {
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %cast = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf16>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      linalg.yield %c : f32
  } -> tensor<2x3x5xf32>
  %b = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%cast : tensor<2x3x5xf32>) outs(%init_b : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  %m = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(m)>], iterator_types = ["reduction", "parallel", "reduction"]}
      ins(%cast : tensor<2x3x5xf32>) outs(%init_m : tensor<3xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<3xf32>
  return %b, %m : tensor<2xf32>, tensor<3xf32>
}

// -----

// CHECK-LABEL: func.func @subbyte_mask(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @subbyte_mask(%x: tensor<2x3x5xi1>, %init: tensor<2xi32>) -> tensor<2xi32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xi1>) outs(%init : tensor<2xi32>) {
    ^bb0(%v: i1, %acc: i32):
      %c = arith.extui %v : i1 to i32
      %s = arith.addi %c, %acc : i32
      linalg.yield %s : i32
  } -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// -----

// The pass only rewrites tensor regions; it must not force an illegal collapse
// of a non-contiguous GM-style view.
// CHECK-LABEL: func.func @strided_buffer(
// CHECK-NOT: memref.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @strided_buffer(%x: memref<2x3x5xf32, strided<[5,10,1]>>, %init: memref<2xf32>) {
  linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : memref<2x3x5xf32, strided<[5,10,1]>>) outs(%init : memref<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  }
  return
}

// -----

// Region discovery is based on iteration semantics, not cast/square/add names.
// CHECK-LABEL: func.func @pointwise_chain(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<2x3x5xf32> into tensor<2x15xf32>
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: arith.negf
// CHECK: } -> tensor<2x15xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: math.absf
// CHECK: } -> tensor<2x15xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.maximumf
func.func @pointwise_chain(%x: tensor<2x3x5xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %neg = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf32>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f32, %out: f32):
      %n = arith.negf %v : f32
      linalg.yield %n : f32
  } -> tensor<2x3x5xf32>
  %abs = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%neg : tensor<2x3x5xf32>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f32, %out: f32):
      %a = math.absf %v : f32
      linalg.yield %a : f32
  } -> tensor<2x3x5xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%abs : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %m = arith.maximumf %v, %acc : f32
      linalg.yield %m : f32
  } -> tensor<2xf32>
  return %max : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @payload_read(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK: memref.load
func.func @payload_read(%x: tensor<2x3x5xf32>, %extra: memref<f32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %e = memref.load %extra[] : memref<f32>
      %a = arith.addf %v, %e : f32
      %s = arith.addf %a, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// Same-shape binary inputs, scalar operands, and multi-level pointwise chains
// join the region. Scalar operands keep their empty indexing maps.
// CHECK-LABEL: func.func @binary_scalar_chain(
// CHECK-DAG: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<4x16x32xf32> into tensor<4x512xf32>
// CHECK-DAG: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2]] : tensor<4x16x32xf32> into tensor<4x512xf32>
// CHECK: %[[ADD:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x512xf32>, tensor<4x512xf32>)
// CHECK: arith.addf
// CHECK: %[[SCALE:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: ins(%[[ADD]], %{{.*}} : tensor<4x512xf32>, f32)
// CHECK: arith.mulf
// CHECK-NOT: tensor.expand_shape
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%[[SCALE]] : tensor<4x512xf32>) outs(%{{.*}} : tensor<4xf32>)
func.func @binary_scalar_chain(%x: tensor<4x16x32xf32>, %y: tensor<4x16x32xf32>, %scale: f32, %init: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4x16x32xf32>
  %add = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x, %y : tensor<4x16x32xf32>, tensor<4x16x32xf32>) outs(%empty : tensor<4x16x32xf32>) {
    ^bb0(%vx: f32, %vy: f32, %out: f32):
      %a = arith.addf %vx, %vy : f32
      linalg.yield %a : f32
  } -> tensor<4x16x32xf32>
  %scaled = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->()>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%add, %scale : tensor<4x16x32xf32>, f32) outs(%empty : tensor<4x16x32xf32>) {
    ^bb0(%v: f32, %s: f32, %out: f32):
      %m = arith.mulf %v, %s : f32
      linalg.yield %m : f32
  } -> tensor<4x16x32xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%scaled : tensor<4x16x32xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<4xf32>
  return %sum : tensor<4xf32>
}

// -----

// Each run of adjacent reduction dimensions is a separate group. The kept
// dimension between them is never crossed.
// CHECK-LABEL: func.func @multi_segment(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1, 2], [3], [4, 5]] : tensor<2x3x5x7x11x13xf32> into tensor<2x15x7x143xf32>
// CHECK: iterator_types = ["parallel", "reduction", "parallel", "reduction"]
// CHECK-SAME: outs(%{{.*}} : tensor<2x7xf32>)
func.func @multi_segment(%x: tensor<2x3x5x7x11x13xf32>, %init: tensor<2x7xf32>) -> tensor<2x7xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k,n,p,q)->(b,m,k,n,p,q)>, affine_map<(b,m,k,n,p,q)->(b,n)>], iterator_types = ["parallel", "reduction", "reduction", "parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5x7x11x13xf32>) outs(%init : tensor<2x7xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2x7xf32>
  return %r : tensor<2x7xf32>
}

// -----

// A zero extent is not merged, while an independent legal group still is.
// CHECK-LABEL: func.func @zero_extent(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] : tensor<2x0x5x3xf32> into tensor<2x0x15xf32>
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @zero_extent(%x: tensor<2x0x5x3xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b)>], iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
      ins(%x : tensor<2x0x5x3xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// A static product that overflows int64 starts a new group.
// CHECK-LABEL: func.func @extent_overflow(
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] : tensor<2x4294967296x4294967296x3xf32> into tensor<2x4294967296x12884901888xf32>
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @extent_overflow(%x: tensor<2x4294967296x4294967296x3xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k,n)->(b,m,k,n)>, affine_map<(b,m,k,n)->(b)>], iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
      ins(%x : tensor<2x4294967296x4294967296x3xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// A producer reached only through a DPS init is not part of the region, even
// when the payload reads the init value.
// CHECK-LABEL: func.func @init_not_edge(
// CHECK: %[[INIT:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: arith.negf
// CHECK: tensor.collapse_shape %[[INIT]] {{\[\[}}0], [1, 2]]
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK: math.absf
// CHECK: iterator_types = ["parallel", "reduction"]
func.func @init_not_edge(%x: tensor<2x3x5xf32>, %y: tensor<2x3x5xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %neg = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%y : tensor<2x3x5xf32>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f32, %out: f32):
      %n = arith.negf %v : f32
      linalg.yield %n : f32
  } -> tensor<2x3x5xf32>
  %abs = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf32>) outs(%neg : tensor<2x3x5xf32>) {
    ^bb0(%v: f32, %out: f32):
      %a = math.absf %v : f32
      %s = arith.addf %a, %out : f32
      linalg.yield %s : f32
  } -> tensor<2x3x5xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%abs : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %sum : tensor<2xf32>
}

// -----

// A user in a nested block is outside the region. The cast keeps its shape
// for that user, and the reduction collapses only its own input.
// CHECK-LABEL: func.func @nested_user(
// CHECK: %[[CAST:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: tensor.collapse_shape %[[CAST]] {{\[\[}}0], [1, 2]]
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: scf.if
// CHECK: tensor.extract %[[CAST]]
func.func @nested_user(%x: tensor<2x3x5xf16>, %init: tensor<2xf32>, %cond: i1) -> (tensor<2xf32>, f32) {
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %cast = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf16>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      linalg.yield %c : f32
  } -> tensor<2x3x5xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%cast : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  %e = scf.if %cond -> (f32) {
    %v = tensor.extract %cast[%c0, %c0, %c0] : tensor<2x3x5xf32>
    scf.yield %v : f32
  } else {
    %zero = arith.constant 0.0 : f32
    scf.yield %zero : f32
  }
  return %sum, %e : tensor<2xf32>, f32
}

// -----

// Producers outside a loop are not joined with a reduction inside its body.
// CHECK-LABEL: func.func @loop_boundary(
// CHECK: %[[CAST:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK: scf.for
// CHECK: tensor.collapse_shape %[[CAST]] {{\[\[}}0], [1, 2]]
// CHECK: iterator_types = ["parallel", "reduction"]
func.func @loop_boundary(%x: tensor<2x3x5xf16>, %init: tensor<2xf32>, %n: index) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2x3x5xf32>
  %cast = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<2x3x5xf16>) outs(%empty : tensor<2x3x5xf32>) {
    ^bb0(%v: f16, %out: f32):
      %c = arith.extf %v : f16 to f32
      linalg.yield %c : f32
  } -> tensor<2x3x5xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (tensor<2xf32>) {
    %s = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%cast : tensor<2x3x5xf32>) outs(%acc : tensor<2xf32>) {
      ^bb0(%v: f32, %a: f32):
        %t = arith.addf %v, %a : f32
        linalg.yield %t : f32
    } -> tensor<2xf32>
    scf.yield %s : tensor<2xf32>
  }
  return %r : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @encoded_tensor(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @encoded_tensor(%x: tensor<2x3x5xf32, "layout">, %init: tensor<2xf32>) -> tensor<2xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xf32, "layout">) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----

// Host functions and functions already marked as pure AIV are protected.
// CHECK-LABEL: func.func @host_func(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @host_func(%x: tensor<2x3x5xf32>, %init: tensor<2xf32>) -> tensor<2xf32>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: func.func @aiv_func(
// CHECK-NOT: tensor.collapse_shape
// CHECK: iterator_types = ["parallel", "reduction", "reduction"]
func.func @aiv_func(%x: tensor<2x3x5xf32>, %init: tensor<2xf32>) -> tensor<2xf32>
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %r = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b)>], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%x : tensor<2x3x5xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %v, %acc : f32
      linalg.yield %s : f32
  } -> tensor<2xf32>
  return %r : tensor<2xf32>
}
