// RUN: bishengir-opt %s -hfusion-fold-extract-insert-pair -hfusion-pre-vectorization-fusion -canonicalize | FileCheck %s --check-prefix=FUSION --implicit-check-not=tensor.insert_slice --implicit-check-not=linalg.elemwise_binary
// RUN: bishengir-opt %s -hfusion-fold-extract-insert-pair -hfusion-pre-vectorization-fusion -one-shot-bufferize="bufferize-function-boundaries" -canonicalize | FileCheck %s --check-prefix=BUFFER --implicit-check-not=memref.copy --implicit-check-not=linalg.fill
// RUN: bishengir-opt %s -hacc-append-device-spec="target=Ascend950PR_9579" -hfusion-fold-extract-insert-pair -hfusion-pre-vectorization-fusion -hfusion-auto-vectorize-v2 -outline-vector-function -convert-hfusion-to-vector -lift-arith-indexcast -canonicalize | FileCheck %s --check-prefix=VECTOR --implicit-check-not=linalg.generic --implicit-check-not='vector<1x64xindex>'

// Existing fusion absorbs the select into the consumer.
// FUSION-LABEL: func.func @prefix
// FUSION: linalg.generic
// FUSION: %[[SEL:.*]] = arith.select
// FUSION-NEXT: %[[SUM:.*]] = arith.addf %{{.*}}, %[[SEL]] : f32
// FUSION-NEXT: linalg.yield %[[SUM]] : f32
// FUSION-NOT: linalg.generic
// FUSION: return

// Allocate only the result, with no intermediate fill or slice copy.
// BUFFER-LABEL: func.func @prefix
// BUFFER: memref.alloc()
// BUFFER-NOT: memref.alloc()
// BUFFER: arith.select
// BUFFER-NEXT: arith.addf
// BUFFER-NOT: memref.alloc()
// BUFFER: return

// AVE requires i32 predicates, not vector broadcasts of index values.
// VECTOR-LABEL: func.func @prefix_outlined_vf_0
// VECTOR: arith.index_cast %{{.*}} : index to i32
// VECTOR: vector.broadcast %{{.*}} : i32 to vector<1x64xi32>
// VECTOR: arith.cmpi ult, %{{.*}}, %{{.*}} : vector<1x64xi32>
// VECTOR: %[[SEL:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<1x64xi1>, vector<1x64xf32>
// VECTOR: arith.addf %{{.*}}, %[[SEL]] : vector<1x64xf32>
// VECTOR-LABEL: func.func @prefix(
// VECTOR: call @prefix_outlined_vf_0
func.func @prefix(%x: tensor<4x1024xf32>, %a: tensor<4x1024xf32>, %n: index)
    -> tensor<4x1024xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x1024xf32>
  %base = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %slice = tensor.extract_slice %x[0, 0] [%n, 1024] [1, 1] : tensor<4x1024xf32> to tensor<?x1024xf32>
  %masked = tensor.insert_slice %slice into %base[0, 0] [%n, 1024] [1, 1] : tensor<?x1024xf32> into tensor<4x1024xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  return %out : tensor<4x1024xf32>
}
