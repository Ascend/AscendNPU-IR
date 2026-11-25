// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s
// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion -canonicalize -hfusion-auto-vectorize-v2 \
// RUN:   | FileCheck %s --check-prefix=V2 --implicit-check-not=scf.for \
// RUN:     --implicit-check-not='arith.{{(subi|shli|bitcast|cmpi|andi|select|negf)}}'

// Keep multiplier computation over 256 scales before broadcasting to 256x32.
// The broadcast itself should still fuse into the quantization consumer.
// CHECK-DAG: #[[$ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$FULL:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$BRC:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$FULL3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MIDDLE:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$LAST:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// V2-DAG: #[[$V2_BRC:.*]] = affine_map<(d0) -> (d0, 0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @preserve_multiplier_before_broadcast(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<256x32xbf16>, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<256xi32>
// CHECK: %[[MULTIPLIER:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME: ins(%[[SCALE]] : tensor<256xi32>)
// CHECK: arith.subi
// CHECK-NEXT: arith.shli
// CHECK-NEXT: arith.bitcast
// CHECK-NEXT: arith.cmpi
// CHECK-NEXT: arith.cmpi
// CHECK-NEXT: arith.andi
// CHECK-NEXT: %[[VALUE:.*]] = arith.select
// CHECK-NEXT: linalg.yield %[[VALUE]] : f32
// CHECK-NEXT: } -> tensor<256xf32>
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$FULL]], #[[$BRC]], #[[$FULL]]]
// CHECK-SAME: ins(%[[INPUT]], %[[MULTIPLIER]] : tensor<256x32xbf16>, tensor<256xf32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: bf16, %[[MUL:.*]]: f32, %{{.*}}: f8E4M3FN):
// CHECK-NEXT: %[[WIDE:.*]] = arith.extf %[[IN]] : bf16 to f32
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[WIDE]], %[[MUL]] : f32
// CHECK-NEXT: %[[QUANT:.*]] = arith.truncf %[[PRODUCT]] : f32 to f8E4M3FN
// CHECK-NEXT: linalg.yield %[[QUANT]] : f8E4M3FN
// CHECK-NEXT: } -> tensor<256x32xf8E4M3FN>
// CHECK-NEXT: return %[[RESULT]] : tensor<256x32xf8E4M3FN>

// Compute 64 distinct multipliers per iteration, then broadcast the saved f32.
// The implicit negative checks forbid additional loops or producer arithmetic.
// V2-LABEL: func.func @preserve_multiplier_before_broadcast(
// V2-DAG: %[[C0:.*]] = arith.constant 0 : index
// V2-DAG: %[[C1:.*]] = arith.constant 1 : index
// V2-DAG: %[[C32:.*]] = arith.constant 32 : index
// V2-DAG: %[[C64:.*]] = arith.constant 64 : index
// V2-DAG: %[[C256:.*]] = arith.constant 256 : index
// V2: %[[MULTIPLIERS:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C256]] step %[[C64]]
// V2-SAME: -> (tensor<256xf32>) {
// V2-NOT: scf.yield
// V2: %[[EXPONENT:.*]] = arith.subi %{{.*}}, %{{.*}} : vector<64xi32>
// V2: %[[BITS:.*]] = arith.shli %[[EXPONENT]], %{{.*}} : vector<64xi32>
// V2-NEXT: %[[VALUE:.*]] = arith.bitcast %[[BITS]] : vector<64xi32> to vector<64xf32>
// V2: %[[POSITIVE:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : vector<64xi32>
// V2: %[[FINITE:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : vector<64xi32>
// V2-NEXT: %[[VALID:.*]] = arith.andi %[[POSITIVE]], %[[FINITE]] : vector<64xi1>
// V2: %[[SELECTED:.*]] = arith.select %[[VALID]], %[[VALUE]], %{{.*}} : vector<64xi1>, vector<64xf32>
// V2: vector.transfer_write %[[SELECTED]],
// V2: scf.yield %{{.*}} : tensor<256xf32>
// V2-NEXT: }
// V2: scf.for %[[GROUP:.*]] = %[[C0]] to %[[C256]] step %[[C1]]
// V2: %[[SLICE:.*]] = tensor.extract_slice %[[MULTIPLIERS]][%[[GROUP]]] [1] [1]
// V2-NEXT: %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C32]] step %[[C64]]
// V2: %[[BROADCAST:.*]] = vector.transfer_read %[[SLICE]][
// V2-SAME: permutation_map = #[[$V2_BRC]]} : tensor<1xf32>, vector<1x64xf32>
// V2: %[[WIDE:.*]] = arith.extf %{{.*}} : vector<1x64xbf16> to vector<1x64xf32>
// V2-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[WIDE]], %[[BROADCAST]] : vector<1x64xf32>
// V2-NEXT: arith.truncf %[[PRODUCT]] : vector<1x64xf32> to vector<1x64xf8E4M3FN>
// V2: return
func.func @preserve_multiplier_before_broadcast(%input: tensor<256x32xbf16>, %scale: tensor<256xi32>) -> tensor<256x32xf8E4M3FN>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
  %c0 = arith.constant 0 : i32
  %c23 = arith.constant 23 : i32
  %c254 = arith.constant 254 : i32
  %c255 = arith.constant 255 : i32
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<256xf32>
  %multiplier = linalg.generic {
    indexing_maps = [affine_map<(g) -> (g)>, affine_map<(g) -> (g)>],
    iterator_types = ["parallel"]
  } ins(%scale : tensor<256xi32>) outs(%empty : tensor<256xf32>) {
  ^bb0(%s: i32, %out: f32):
    %exponent = arith.subi %c254, %s : i32
    %bits = arith.shli %exponent, %c23 : i32
    %value = arith.bitcast %bits : i32 to f32
    %positive = arith.cmpi sgt, %s, %c0 : i32
    %finite = arith.cmpi slt, %s, %c255 : i32
    %valid = arith.andi %positive, %finite : i1
    %selected = arith.select %valid, %value, %zero : f32
    linalg.yield %selected : f32
  } -> tensor<256xf32>
  %brc_init = tensor.empty() : tensor<256x32xf32>
  %broadcast = linalg.broadcast ins(%multiplier : tensor<256xf32>) outs(%brc_init : tensor<256x32xf32>) dimensions = [1]
  %result_init = tensor.empty() : tensor<256x32xf8E4M3FN>
  %result = linalg.generic {
    indexing_maps = [affine_map<(g, j) -> (g, j)>, affine_map<(g, j) -> (g, j)>, affine_map<(g, j) -> (g, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %broadcast : tensor<256x32xbf16>, tensor<256x32xf32>) outs(%result_init : tensor<256x32xf8E4M3FN>) {
  ^bb0(%in: bf16, %mul: f32, %out: f8E4M3FN):
    %wide = arith.extf %in : bf16 to f32
    %product = arith.mulf %wide, %mul : f32
    %quant = arith.truncf %product : f32 to f8E4M3FN
    linalg.yield %quant : f8E4M3FN
  } -> tensor<256x32xf8E4M3FN>
  return %result : tensor<256x32xf8E4M3FN>
}

// Rank-2 producers should fuse across a middle-axis broadcast.
// CHECK-LABEL: func.func @fuse_broadcast_middle_axis(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x32x128xf32>, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4x128xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$FULL3]], #[[$MIDDLE]], #[[$FULL3]]]
// CHECK-SAME: ins(%[[INPUT]], %[[SCALE]] : tensor<4x32x128xf32>, tensor<4x128xf32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[VALUE:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[NEG:.*]] = arith.negf %[[VALUE]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[IN]], %[[NEG]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK-NEXT: } -> tensor<4x32x128xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x32x128xf32>

// Vectorize the fused body: 4 * 32 * (128 / 64) vector negations.
// V2-LABEL: func.func @fuse_broadcast_middle_axis(
// V2-DAG: %[[C0:.*]] = arith.constant 0 : index
// V2-DAG: %[[C1:.*]] = arith.constant 1 : index
// V2-DAG: %[[C4:.*]] = arith.constant 4 : index
// V2-DAG: %[[C32:.*]] = arith.constant 32 : index
// V2-DAG: %[[C64:.*]] = arith.constant 64 : index
// V2-DAG: %[[C128:.*]] = arith.constant 128 : index
// V2: scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// V2-NEXT: %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C32]] step %[[C1]]
// V2-NEXT: %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C64]]
// V2-NOT: scf.yield
// V2: %[[NEG:.*]] = arith.negf %{{.*}} : vector<1x1x64xf32>
// V2-NEXT: arith.addf %{{.*}}, %[[NEG]] : vector<1x1x64xf32>
// V2: return
func.func @fuse_broadcast_middle_axis(%input: tensor<4x32x128xf32>, %scale: tensor<4x128xf32>) -> tensor<4x32x128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
  %producer_init = tensor.empty() : tensor<4x128xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%scale : tensor<4x128xf32>) outs(%producer_init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %negated = arith.negf %in : f32
    linalg.yield %negated : f32
  } -> tensor<4x128xf32>
  %consumer_init = tensor.empty() : tensor<4x32x128xf32>
  %broadcast = linalg.broadcast ins(%producer : tensor<4x128xf32>) outs(%consumer_init : tensor<4x32x128xf32>) dimensions = [1]
  %result = linalg.add ins(%input, %broadcast : tensor<4x32x128xf32>, tensor<4x32x128xf32>) outs(%consumer_init : tensor<4x32x128xf32>) -> tensor<4x32x128xf32>
  return %result : tensor<4x32x128xf32>
}

// Rank-2 producers should fuse across a last-axis broadcast.
// CHECK-LABEL: func.func @fuse_broadcast_last_axis(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<4x128x32xf32>, %[[SCALE:[a-zA-Z0-9_]+]]: tensor<4x128xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$FULL3]], #[[$LAST]], #[[$FULL3]]]
// CHECK-SAME: ins(%[[INPUT]], %[[SCALE]] : tensor<4x128x32xf32>, tensor<4x128xf32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %[[VALUE:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[NEG:.*]] = arith.negf %[[VALUE]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[IN]], %[[NEG]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK-NEXT: } -> tensor<4x128x32xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x128x32xf32>

// Vectorize the fused body: 4 * 128 vector negations. The innermost
// 0..32 step 64 loop executes once.
// V2-LABEL: func.func @fuse_broadcast_last_axis(
// V2-DAG: %[[C0:.*]] = arith.constant 0 : index
// V2-DAG: %[[C1:.*]] = arith.constant 1 : index
// V2-DAG: %[[C4:.*]] = arith.constant 4 : index
// V2-DAG: %[[C32:.*]] = arith.constant 32 : index
// V2-DAG: %[[C64:.*]] = arith.constant 64 : index
// V2-DAG: %[[C128:.*]] = arith.constant 128 : index
// V2: scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// V2-NEXT: %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C1]]
// V2: scf.for %{{.*}} = %[[C0]] to %[[C32]] step %[[C64]]
// V2-NOT: scf.yield
// V2: %[[NEG:.*]] = arith.negf %{{.*}} : vector<1x1x64xf32>
// V2-NEXT: arith.addf %{{.*}}, %[[NEG]] : vector<1x1x64xf32>
// V2: return
func.func @fuse_broadcast_last_axis(%input: tensor<4x128x32xf32>, %scale: tensor<4x128xf32>) -> tensor<4x128x32xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
  %producer_init = tensor.empty() : tensor<4x128xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%scale : tensor<4x128xf32>) outs(%producer_init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %negated = arith.negf %in : f32
    linalg.yield %negated : f32
  } -> tensor<4x128xf32>
  %consumer_init = tensor.empty() : tensor<4x128x32xf32>
  %broadcast = linalg.broadcast ins(%producer : tensor<4x128xf32>) outs(%consumer_init : tensor<4x128x32xf32>) dimensions = [2]
  %result = linalg.add ins(%input, %broadcast : tensor<4x128x32xf32>, tensor<4x128x32xf32>) outs(%consumer_init : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>
  return %result : tensor<4x128x32xf32>
}

// Without broadcast, producer and consumer must still fuse into one generic.
// CHECK-LABEL: func.func @fuse_same_domain(
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$FULL]], #[[$FULL]], #[[$FULL]]]
// CHECK: %[[NEG:.*]] = arith.negf
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %{{.*}}, %[[NEG]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK-NEXT: } -> tensor<4x128xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4x128xf32>

// V2-LABEL: func.func @fuse_same_domain(
// V2-DAG: %[[C0:.*]] = arith.constant 0 : index
// V2-DAG: %[[C1:.*]] = arith.constant 1 : index
// V2-DAG: %[[C4:.*]] = arith.constant 4 : index
// V2-DAG: %[[C64:.*]] = arith.constant 64 : index
// V2-DAG: %[[C128:.*]] = arith.constant 128 : index
// V2: scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// V2-NEXT: %{{.*}} = scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C64]]
// V2-NOT: scf.yield
// V2: %[[NEG:.*]] = arith.negf %{{.*}} : vector<1x64xf32>
// V2-NEXT: arith.addf %{{.*}}, %[[NEG]] : vector<1x64xf32>
// V2: return
func.func @fuse_same_domain(%input: tensor<4x128xf32>, %scale: tensor<4x128xf32>) -> tensor<4x128xf32>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, parallel_mode = "simd"} {
  %producer_init = tensor.empty() : tensor<4x128xf32>
  %producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%scale : tensor<4x128xf32>) outs(%producer_init : tensor<4x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %negated = arith.negf %in : f32
    linalg.yield %negated : f32
  } -> tensor<4x128xf32>
  %consumer_init = tensor.empty() : tensor<4x128xf32>
  %result = linalg.add ins(%input, %producer : tensor<4x128xf32>, tensor<4x128xf32>) outs(%consumer_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}
}
