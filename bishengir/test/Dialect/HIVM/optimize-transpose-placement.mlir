// RUN: bishengir-opt -hivm-optimize-transpose-placement %s | FileCheck %s --check-prefixes=CHECK,PLACED
// RUN: bishengir-opt -hivm-optimize-transpose-placement -hivm-inline-fixpipe %s | FileCheck %s --check-prefixes=CHECK,INLINE

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {

// HSTU shape: fixpipe -> vmul(scalar) -> vadd(vcast(vtranspose)) -> vcast -> vtranspose[1,0].
// Both transposes cancel, so the chain is retyped and one transpose sits on the fixpipe.
// CHECK-LABEL: func.func @sink_hstu_like
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: hivm.hir.vtranspose ins(%[[FP]]
// PLACED-NOT: hivm.hir.vtranspose
// PLACED: hivm.hir.vadd
// PLACED: return
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
// INLINE: return
func.func @sink_hstu_like(%fpSrc : tensor<128x32xf32>, %loaded : tensor<32x128xbf16>) -> tensor<32x128xbf16> {
  %scale = arith.constant 2.000000e+00 : f32
  %sibDst = tensor.empty() : tensor<128x32xbf16>
  %sib = hivm.hir.vtranspose ins(%loaded : tensor<32x128xbf16>) outs(%sibDst : tensor<128x32xbf16>) permutation = [1, 0] -> tensor<128x32xbf16>
  %castDst = tensor.empty() : tensor<128x32xf32>
  %cast = hivm.hir.vcast ins(%sib : tensor<128x32xbf16>) outs(%castDst : tensor<128x32xf32>) -> tensor<128x32xf32>
  %fpDst = tensor.empty() : tensor<128x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<128x32xf32>) outs(%fpDst : tensor<128x32xf32>) -> tensor<128x32xf32>
  %mulDst = tensor.empty() : tensor<128x32xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<128x32xf32>, f32) outs(%mulDst : tensor<128x32xf32>) -> tensor<128x32xf32>
  %addDst = tensor.empty() : tensor<128x32xf32>
  %add = hivm.hir.vadd ins(%cast, %mul : tensor<128x32xf32>, tensor<128x32xf32>) outs(%addDst : tensor<128x32xf32>) -> tensor<128x32xf32>
  %tailDst = tensor.empty() : tensor<128x32xbf16>
  %tail = hivm.hir.vcast ins(%add : tensor<128x32xf32>) outs(%tailDst : tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %trDst = tensor.empty() : tensor<32x128xbf16>
  %tr = hivm.hir.vtranspose ins(%tail : tensor<128x32xbf16>) outs(%trDst : tensor<32x128xbf16>) permutation = [1, 0] -> tensor<32x128xbf16>
  return %tr : tensor<32x128xbf16>
}

// Overlap corner: %t1 is both a standalone candidate and the cancellable
// sibling of %t2's chain. Plans apply in top-down program order, so %t1
// sinks first and %t2's plan is re-analyzed once against the rewritten IR
// (its cancellable sibling becomes the rebuilt vmul on the moved
// transpose). %t2 still fires: fp1 ends up feeding a vmul directly and one
// transpose remains, on fp2. This pins the deterministic order and the
// revalidation safety net, not greedy-order equivalence.
// CHECK-LABEL: func.func @sink_nested_candidate_overlap
// PLACED: %[[FP1:.*]] = hivm.hir.fixpipe
// PLACED: %[[FP2:.*]] = hivm.hir.fixpipe
// PLACED: %[[TR:.*]] = hivm.hir.vtranspose ins(%[[FP2]]
// PLACED-NOT: hivm.hir.vtranspose
// PLACED: %[[MA:.*]] = hivm.hir.vmul ins(%[[TR]], %{{.*}} : tensor<4x8xf32>, f32)
// PLACED: %[[MB:.*]] = hivm.hir.vmul ins(%[[FP1]], %{{.*}} : tensor<4x8xf32>, f32)
// PLACED: hivm.hir.vadd ins(%[[MA]], %[[MB]]
// PLACED-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.vadd
func.func @sink_nested_candidate_overlap(%src1 : tensor<4x8xf32>, %src2 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fp1Dst = tensor.empty() : tensor<4x8xf32>
  %fp1 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src1 : tensor<4x8xf32>) outs(%fp1Dst : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul1Dst = tensor.empty() : tensor<4x8xf32>
  %mul1 = hivm.hir.vmul ins(%fp1, %scale : tensor<4x8xf32>, f32) outs(%mul1Dst : tensor<4x8xf32>) -> tensor<4x8xf32>
  %t1Dst = tensor.empty() : tensor<8x4xf32>
  %t1 = hivm.hir.vtranspose ins(%mul1 : tensor<4x8xf32>) outs(%t1Dst : tensor<8x4xf32>) permutation = [1, 0] -> tensor<8x4xf32>
  %fp2Dst = tensor.empty() : tensor<8x4xf32>
  %fp2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src2 : tensor<8x4xf32>) outs(%fp2Dst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mul2Dst = tensor.empty() : tensor<8x4xf32>
  %mul2 = hivm.hir.vmul ins(%fp2, %scale : tensor<8x4xf32>, f32) outs(%mul2Dst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %addDst = tensor.empty() : tensor<8x4xf32>
  %add = hivm.hir.vadd ins(%mul2, %t1 : tensor<8x4xf32>, tensor<8x4xf32>) outs(%addDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %t2Dst = tensor.empty() : tensor<4x8xf32>
  %t2 = hivm.hir.vtranspose ins(%add : tensor<8x4xf32>) outs(%t2Dst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %t2 : tensor<4x8xf32>
}

// Bucket B: a scalar vbrc is rebuilt at the swapped shape, so it costs no
// transpose. delta stays 1.
// CHECK-LABEL: func.func @sink_scalar_vbrc
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: %[[TR:.*]] = hivm.hir.vtranspose ins(%[[FP]]
// PLACED-NOT: hivm.hir.vtranspose
// PLACED: %[[BRC:.*]] = hivm.hir.vbrc ins(%{{.*}} : f32) outs(%{{.*}} : tensor<32x64xf32>) -> tensor<32x64xf32>
// PLACED: hivm.hir.vadd ins(%[[TR]], %[[BRC]]
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.vbrc
// INLINE: hivm.hir.vadd
func.func @sink_scalar_vbrc(%fpSrc : tensor<64x32xf32>) -> tensor<32x64xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %brcDst = tensor.empty() : tensor<64x32xf32>
  %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brcDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %addDst = tensor.empty() : tensor<64x32xf32>
  %add = hivm.hir.vadd ins(%fp, %brc : tensor<64x32xf32>, tensor<64x32xf32>) outs(%addDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%add : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}

// One chain, six sibling branches: A with a vcast replay, A bare, B scalar
// vbrc, B tensor.empty, C block argument, C computed by vmul.
// The two C siblings each would need a fresh transpose, which the cost model
// treats as prohibitive (kCostlySiblingCost), so delta < 1 and the pass does
// not fire. The IR is left unchanged.
// CHECK-LABEL: func.func @sink_mixed_buckets
// CHECK: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// CHECK: hivm.hir.vtranspose {{.*}}permutation = [1, 0]
// CHECK-NOT: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
func.func @sink_mixed_buckets(
    %fpSrc : tensor<64x32xf32>,
    %costly : tensor<64x32xf32>,
    %vmulLhs : tensor<64x32xf32>,
    %vmulRhs : tensor<64x32xf32>,
    %loadedCast : tensor<32x64xbf16>,
    %loadedBare : tensor<32x64xf32>) -> tensor<32x64xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %brcDst = tensor.empty() : tensor<64x32xf32>
  %brc = hivm.hir.vbrc ins(%cst : f32) outs(%brcDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %aCastDst = tensor.empty() : tensor<64x32xbf16>
  %aCastTr = hivm.hir.vtranspose ins(%loadedCast : tensor<32x64xbf16>) outs(%aCastDst : tensor<64x32xbf16>) permutation = [1, 0] -> tensor<64x32xbf16>
  %aCastCastDst = tensor.empty() : tensor<64x32xf32>
  %aCast = hivm.hir.vcast ins(%aCastTr : tensor<64x32xbf16>) outs(%aCastCastDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %aBareDst = tensor.empty() : tensor<64x32xf32>
  %aBare = hivm.hir.vtranspose ins(%loadedBare : tensor<32x64xf32>) outs(%aBareDst : tensor<64x32xf32>) permutation = [1, 0] -> tensor<64x32xf32>
  %vmulDst = tensor.empty() : tensor<64x32xf32>
  %vmul = hivm.hir.vmul ins(%vmulLhs, %vmulRhs : tensor<64x32xf32>, tensor<64x32xf32>) outs(%vmulDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %emptySib = tensor.empty() : tensor<64x32xf32>
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s0Dst = tensor.empty() : tensor<64x32xf32>
  %s0 = hivm.hir.vadd ins(%fp, %brc : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s0Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s1Dst = tensor.empty() : tensor<64x32xf32>
  %s1 = hivm.hir.vadd ins(%s0, %aCast : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s1Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s2Dst = tensor.empty() : tensor<64x32xf32>
  %s2 = hivm.hir.vadd ins(%s1, %costly : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s2Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s3Dst = tensor.empty() : tensor<64x32xf32>
  %s3 = hivm.hir.vadd ins(%s2, %aBare : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s3Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s4Dst = tensor.empty() : tensor<64x32xf32>
  %s4 = hivm.hir.vadd ins(%s3, %vmul : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s4Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %s5Dst = tensor.empty() : tensor<64x32xf32>
  %s5 = hivm.hir.vadd ins(%s4, %emptySib : tensor<64x32xf32>, tensor<64x32xf32>) outs(%s5Dst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%s5 : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}

// Unary chain, no sibling transpose. delta = 1, still worth sinking.
// CHECK-LABEL: func.func @sink_unary_scale
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: %[[TR:.*]] = hivm.hir.vtranspose ins(%[[FP]]
// PLACED: hivm.hir.vmul ins(%[[TR]]
// PLACED-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.vmul
func.func @sink_unary_scale(%fpSrc : tensor<64x32xf32>) -> tensor<32x64xf32> {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %mulDst = tensor.empty() : tensor<64x32xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<64x32xf32>, f32) outs(%mulDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}

// Bucket C sibling: sinking would create a new transpose, which the cost
// model (kCostlySiblingCost) treats as prohibitive, so delta < 1. No sink.
// CHECK-LABEL: func.func @no_sink_costly_sibling
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vtranspose
// CHECK-NOT: hivm.hir.vtranspose
func.func @no_sink_costly_sibling(%fpSrc : tensor<64x32xf32>, %other : tensor<64x32xf32>) -> tensor<32x64xf32> {
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %addDst = tensor.empty() : tensor<64x32xf32>
  %add = hivm.hir.vadd ins(%fp, %other : tensor<64x32xf32>, tensor<64x32xf32>) outs(%addDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%add : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}

// A last-axis swap on rank 4 is a real transpose, but not the [1, 0] pair
// InlineFixpipe can absorb. v1 leaves it alone.
// CHECK-LABEL: func.func @no_sink_rank4
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vtranspose {{.*}}permutation = [0, 1, 3, 2]
func.func @no_sink_rank4(%fpSrc : tensor<2x2x64x32xf32>) -> tensor<2x2x32x64xf32> {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<2x2x64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<2x2x64x32xf32>) outs(%fpDst : tensor<2x2x64x32xf32>) -> tensor<2x2x64x32xf32>
  %mulDst = tensor.empty() : tensor<2x2x64x32xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<2x2x64x32xf32>, f32) outs(%mulDst : tensor<2x2x64x32xf32>) -> tensor<2x2x64x32xf32>
  %trDst = tensor.empty() : tensor<2x2x32x64xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<2x2x64x32xf32>) outs(%trDst : tensor<2x2x32x64xf32>) permutation = [0, 1, 3, 2] -> tensor<2x2x32x64xf32>
  return %tr : tensor<2x2x32x64xf32>
}

// Fixpipe has two real users, so InlineFixpipe would not fold a transpose either.
// CHECK-LABEL: func.func @no_sink_multi_user_fixpipe
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe
// CHECK: hivm.hir.vmul ins(%[[FP]]
// CHECK: hivm.hir.vtranspose
func.func @no_sink_multi_user_fixpipe(%fpSrc : tensor<64x32xf32>) -> (tensor<32x64xf32>, tensor<64x32xf32>) {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %mulDst = tensor.empty() : tensor<64x32xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<64x32xf32>, f32) outs(%mulDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %otherDst = tensor.empty() : tensor<64x32xf32>
  %other = hivm.hir.vmul ins(%fp, %scale : tensor<64x32xf32>, f32) outs(%otherDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr, %other : tensor<32x64xf32>, tensor<64x32xf32>
}

// Already adjacent: this pass leaves it, and InlineFixpipe folds it to NZ2DN.
// CHECK-LABEL: func.func @no_sink_already_adjacent
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe
// PLACED: %[[TR:.*]] = hivm.hir.vtranspose ins(%[[FP]]
// PLACED: hivm.hir.vmul ins(%[[TR]]
// PLACED-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
// INLINE: hivm.hir.vmul
func.func @no_sink_already_adjacent(%fpSrc : tensor<64x32xf32>) -> tensor<32x64xf32> {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%fp : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  %mulDst = tensor.empty() : tensor<32x64xf32>
  %mul = hivm.hir.vmul ins(%tr, %scale : tensor<32x64xf32>, f32) outs(%mulDst : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %mul : tensor<32x64xf32>
}

// [1, 0] whose result shape is not the swapped source. Rewriting it used to
// produce a return type that failed verification.
// CHECK-LABEL: func.func @no_sink_unswapped_shape
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe
// CHECK: %[[M:.*]] = hivm.hir.vmul ins(%[[FP]]
// CHECK: hivm.hir.vtranspose ins(%[[M]]
func.func @no_sink_unswapped_shape(%src : tensor<8x4xf32>) -> tensor<8x4xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<8x4xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<8x4xf32>) permutation = [1, 0] -> tensor<8x4xf32>
  return %tr : tensor<8x4xf32>
}

// A memref temp_buffer is a DPS input. Cloning it as a missing sibling used
// to crash.
// CHECK-LABEL: func.func @no_sink_temp_buffer
// CHECK: %[[FP:.*]] = hivm.hir.fixpipe
// CHECK: %[[M:.*]] = hivm.hir.vmul ins(%[[FP]]
// CHECK-SAME: temp_buffer
// CHECK: hivm.hir.vtranspose ins(%[[M]]
func.func @no_sink_temp_buffer(%src : tensor<8x4xf32>, %tmp : memref<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) temp_buffer(%tmp : memref<8x4xf32>) -> tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %tr : tensor<4x8xf32>
}

// A mark and a debug op on the chain follow the swapped value.
// CHECK-LABEL: func.func @sink_keeps_mark_and_debug
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: %[[TR:.*]] = hivm.hir.vtranspose ins(%[[FP]]
// PLACED: %[[M:.*]] = hivm.hir.vmul ins(%[[TR]]
// PLACED: annotation.mark %[[M]]
// PLACED-SAME: otp.keep
// PLACED: prefix = "otp_probe"{{.*}}%[[M]]
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE: otp.keep
// INLINE: prefix = "otp_probe"
// INLINE-NOT: hivm.hir.vtranspose
func.func @sink_keeps_mark_and_debug(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  annotation.mark %mul {otp.keep = 1 : i64} : tensor<8x4xf32>
  hivm.hir.debug {debugtype = "print", hex = false, prefix = "otp_probe"} %mul : tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %tr : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @sink_keeps_disable_align
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe
// PLACED: hivm.hir.vtranspose ins(%[[FP]]
// PLACED-SAME: disable_align = true
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE-NOT: hivm.hir.vtranspose
func.func @sink_keeps_disable_align(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] disable_align = true -> tensor<4x8xf32>
  return %tr : tensor<4x8xf32>
}

// The value that replaces the transpose keeps that destination's address
// space and inserted-tensor marker. The moved transpose keeps the fixpipe's.
// CHECK-LABEL: func.func @sink_result_buffer_attrs
// PLACED: hivm.address_space = #hivm.address_space<cbuf>
// PLACED: hivm.hir.fixpipe
// PLACED: hivm.hir.vtranspose
// PLACED: hivm.address_space = #hivm.address_space<ub>
// PLACED-SAME: "hivm.inserted-tensor"
// PLACED: hivm.hir.vmul
func.func @sink_result_buffer_attrs(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() {hivm.address_space = #hivm.address_space<cbuf>} : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %trDst = tensor.empty() {hivm.address_space = #hivm.address_space<ub>, "hivm.inserted-tensor"} : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %tr : tensor<4x8xf32>
}

// An older mark on the fixpipe used to hide the transpose from inline-fixpipe.
// CHECK-LABEL: func.func @sink_mark_on_fixpipe
// PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: annotation.mark %[[FP]]
// PLACED-SAME: otp.fp
// PLACED: hivm.hir.vtranspose ins(%[[FP]]
// INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
// INLINE: otp.fp
// INLINE-NOT: hivm.hir.vtranspose
func.func @sink_mark_on_fixpipe(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %scale = arith.constant 2.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  annotation.mark %fp {otp.fp = 1 : i64} : tensor<8x4xf32>
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %tr : tensor<4x8xf32>
}

// A live tensor.dim user makes the chain op fail the single-real-use gate,
// so the transpose stays put and InlineFixpipe sees nothing to fold. v1
// fired here only because the greedy driver folded the static dim first;
// a canonicalizer before this pass restores that case. (The inline-fixpipe
// pipeline folds the dim afterwards, but the sink has already declined.)
// CHECK-LABEL: func.func @no_sink_live_dim
// CHECK: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// PLACED: tensor.dim
// CHECK: hivm.hir.vtranspose {{.*}}permutation = [1, 0]
// CHECK-NOT: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
func.func @no_sink_live_dim(%src : tensor<8x4xf32>) -> (tensor<4x8xf32>, index) {
  %fpDst = tensor.empty() : tensor<8x4xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %scale = arith.constant 2.000000e+00 : f32
  %mulDst = tensor.empty() : tensor<8x4xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %mul, %c0 : tensor<8x4xf32>
  %trDst = tensor.empty() : tensor<4x8xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
  return %tr, %dim : tensor<4x8xf32>, index
}

// A chain that crosses an scf.for boundary is a fixed-point placement
// problem (loop-carried layout through iter_args and yield). The policy is
// conservative: the backward walk stops at the loop result, the transpose
// stays put, and InlineFixpipe sees nothing to fold.
// CHECK-LABEL: func.func @no_sink_scf_for_boundary
// CHECK: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
// CHECK: scf.for
// CHECK: hivm.hir.vtranspose {{.*}}permutation = [1, 0]
// CHECK-NOT: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
func.func @no_sink_scf_for_boundary(%fpSrc : tensor<64x32xf32>, %lb : index, %ub : index, %step : index) -> tensor<32x64xf32> {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %fp) -> (tensor<64x32xf32>) {
    %mulDst = tensor.empty() : tensor<64x32xf32>
    %mul = hivm.hir.vmul ins(%acc, %scale : tensor<64x32xf32>, f32) outs(%mulDst : tensor<64x32xf32>) -> tensor<64x32xf32>
    scf.yield %mul : tensor<64x32xf32>
  }
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%res : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}

}

// Membase target: the pass is regbase-only.
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
// CHECK-LABEL: func.func @no_sink_membase
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vtranspose
func.func @no_sink_membase(%fpSrc : tensor<64x32xf32>) -> tensor<32x64xf32> {
  %scale = arith.constant 1.000000e+00 : f32
  %fpDst = tensor.empty() : tensor<64x32xf32>
  %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%fpSrc : tensor<64x32xf32>) outs(%fpDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %mulDst = tensor.empty() : tensor<64x32xf32>
  %mul = hivm.hir.vmul ins(%fp, %scale : tensor<64x32xf32>, f32) outs(%mulDst : tensor<64x32xf32>) -> tensor<64x32xf32>
  %trDst = tensor.empty() : tensor<32x64xf32>
  %tr = hivm.hir.vtranspose ins(%mul : tensor<64x32xf32>) outs(%trDst : tensor<32x64xf32>) permutation = [1, 0] -> tensor<32x64xf32>
  return %tr : tensor<32x64xf32>
}
}

// The nearest module that carries hacc.target wins.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  // CHECK-LABEL: func.func @no_sink_inner_membase
  // CHECK: %[[FP:.*]] = hivm.hir.fixpipe
  // CHECK: hivm.hir.vmul ins(%[[FP]]
  // CHECK: hivm.hir.vtranspose
  func.func @no_sink_inner_membase(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
    %scale = arith.constant 2.000000e+00 : f32
    %fpDst = tensor.empty() : tensor<8x4xf32>
    %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
    %mulDst = tensor.empty() : tensor<8x4xf32>
    %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
    %trDst = tensor.empty() : tensor<4x8xf32>
    %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
    return %tr : tensor<4x8xf32>
  }
  }
}

module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @sink_inner_regbase
  // PLACED: %[[FP:.*]] = hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2nd>
  // PLACED: hivm.hir.vtranspose ins(%[[FP]]
  // INLINE: hivm.hir.fixpipe {{.*}}dma_mode = #hivm.dma_mode<nz2dn>
  // INLINE-NOT: hivm.hir.vtranspose
  func.func @sink_inner_regbase(%src : tensor<8x4xf32>) -> tensor<4x8xf32> {
    %scale = arith.constant 2.000000e+00 : f32
    %fpDst = tensor.empty() : tensor<8x4xf32>
    %fp = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : tensor<8x4xf32>) outs(%fpDst : tensor<8x4xf32>) -> tensor<8x4xf32>
    %mulDst = tensor.empty() : tensor<8x4xf32>
    %mul = hivm.hir.vmul ins(%fp, %scale : tensor<8x4xf32>, f32) outs(%mulDst : tensor<8x4xf32>) -> tensor<8x4xf32>
    %trDst = tensor.empty() : tensor<4x8xf32>
    %tr = hivm.hir.vtranspose ins(%mul : tensor<8x4xf32>) outs(%trDst : tensor<4x8xf32>) permutation = [1, 0] -> tensor<4x8xf32>
    return %tr : tensor<4x8xf32>
  }
  }
}
