//===- FractalLinearLayoutConversions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linear layout conversion for fractal shared memory
// encodings (zN/nZ) used by Ascend Cube hardware.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TritonExt/IR/FractalLinearLayoutConversions.h"
#include "bishengir/Dialect/TritonExt/IR/TritonExtAttrs.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

#define S(v) StringAttr::get(ctx, (v))

// Replicated from triton's LinearLayoutConversions.cpp (anonymous namespace).
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  MLIRContext *ctx = layout.getContext();
  StringAttr kBlock = S("block");

  int rank = static_cast<int>(layout.getCTAOrder().size());
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];
    assert(ctas % split == 0);
    ret *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
           LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
  }
  return ret.transposeOuts(outDimNames);
}

LinearLayout combineCtaCgaWithShape(LinearLayout ctaLayout,
                                    CTALayoutAttr cgaLayoutAttr,
                                    ArrayRef<int64_t> shape) {
  int rank = static_cast<int>(shape.size());
  assert(ctaLayout.getNumOutDims() == rank);
  assert(static_cast<int>(cgaLayoutAttr.getCTAOrder().size()) == rank);
  MLIRContext *ctx = cgaLayoutAttr.getContext();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape)) {
    labeledShape[dim] = size;
  }

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(makeCgaLayout(cgaLayoutAttr), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(llvm::to_vector(ctaLayout.getOutDimNames()) ==
         llvm::to_vector(cgaLayout.getOutDimNames()));
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout ret = (ctaLayout * cgaLayout).transposeOuts(outDimNames);
  for (auto dim : ret.getOutDimNames()) {
    assert(ret.getOutDimSize(dim) == labeledShape[dim]);
  }
  return ret;
}

} // anonymous namespace

namespace bishengir::triton_ext {

LinearLayout fractalSharedToLinearLayout(ArrayRef<int64_t> shape,
                                         FractalSharedEncodingAttr fractal) {
  MLIRContext *ctx = fractal.getContext();
  auto shapePerCTA = getShapePerCTA(fractal, shape);
  int rank = static_cast<int>(shape.size());
  assert(rank >= 2);

  auto outDimNames = standardOutDimNames(ctx, rank);

  int mDim = rank - 2;
  int nDim = rank - 1;

  StringAttr mName = outDimNames[mDim];
  StringAttr nName = outDimNames[nDim];

  int64_t fM = fractal.getFractalM0();
  int64_t fN = fractal.getFractalN0();

  int64_t outerM = shapePerCTA[mDim] / fM;
  int64_t outerN = shapePerCTA[nDim] / fN;

  bool isZN = fractal.getLayoutType() == FractalLayoutType::zN;

  std::vector<std::vector<int>> bases2D;

  if (isZN) {
    // Inner Z-shape: columns first (fastest varying), then rows.
    for (int c = 1; c < fN; c *= 2)
      bases2D.push_back({0, c});
    for (int r = 1; r < fM; r *= 2)
      bases2D.push_back({r, 0});

    // Outer N-shape: M-blocks first (column-major block ordering), then
    // N-blocks.
    for (int bm = 1; bm < outerM; bm *= 2)
      bases2D.push_back({static_cast<int>(bm * fM), 0});
    for (int bn = 1; bn < outerN; bn *= 2)
      bases2D.push_back({0, static_cast<int>(bn * fN)});
  } else {
    // Inner N-shape: rows first (fastest varying), then columns.
    for (int r = 1; r < fM; r *= 2)
      bases2D.push_back({r, 0});
    for (int c = 1; c < fN; c *= 2)
      bases2D.push_back({0, c});

    // Outer Z-shape: N-blocks first (row-major block ordering), then M-blocks.
    for (int bn = 1; bn < outerN; bn *= 2)
      bases2D.push_back({0, static_cast<int>(bn * fN)});
    for (int bm = 1; bm < outerM; bm *= 2)
      bases2D.push_back({static_cast<int>(bm * fM), 0});
  }

  LinearLayout ctaLayout =
      LinearLayout({{S("offset"), bases2D}}, {mName, nName});

  // Higher dimensions (batch, etc.) as identity.
  for (int i = 0; i < rank - 2; ++i) {
    if (shapePerCTA[i] > 1)
      ctaLayout *=
          LinearLayout::identity1D(shapePerCTA[i], S("offset"), outDimNames[i]);
  }

  return combineCtaCgaWithShape(ctaLayout, fractal.getCTALayout(), shape);
}

} // namespace bishengir::triton_ext
