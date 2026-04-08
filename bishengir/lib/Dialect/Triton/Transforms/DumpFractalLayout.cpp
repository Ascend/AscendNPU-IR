//===- DumpFractalLayout.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a debug/test pass that prints the fractal shared
// memory layout mapping for a given tensor shape and element offset.
//
// Usage (from bishengir-opt):
//   bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN \
//                                        probe-offset=1536" %s
//
// Output (to stdout):
//   fractal-layout: offset=1536 -> dim0=32 dim1=16
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "bishengir/Dialect/TritonExt/IR/FractalLinearLayoutConversions.h"
#include "bishengir/Dialect/TritonExt/IR/TritonExtAttrs.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_DUMPFRACTALLAYOUT
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace bishengir::triton_ext;

// Parse a comma-separated string like "64,64" into a SmallVector<int64_t>.
static SmallVector<int64_t> parseShape(StringRef s) {
  SmallVector<int64_t> result;
  SmallVector<StringRef> parts;
  s.split(parts, ',');
  for (auto part : parts) {
    int64_t v;
    if (!part.trim().getAsInteger(10, v))
      result.push_back(v);
  }
  return result;
}

class DumpFractalLayoutPass
    : public impl::DumpFractalLayoutBase<DumpFractalLayoutPass> {
public:
  explicit DumpFractalLayoutPass(const DumpFractalLayoutOptions &options)
      : impl::DumpFractalLayoutBase<DumpFractalLayoutPass>(options) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    SmallVector<int64_t> shape = parseShape(shapeStr);
    if (shape.size() < 2) {
      getOperation().emitError(
          "dump-fractal-layout: 'shape' must have at least 2 dimensions");
      return signalPassFailure();
    }

    auto ctaLayout = CTALayoutAttr::getDefault(ctx, shape.size());
    auto lt = symbolizeFractalLayoutType(layoutTypeStr);
    if (!lt) {
      getOperation().emitError("dump-fractal-layout: unknown layoutType '")
          << layoutTypeStr << "', expected 'zN' or 'nZ'";
      return signalPassFailure();
    }
    auto fractalAttr = FractalSharedEncodingAttr::get(
        ctx, fractalM0, fractalN0, *lt, ctaLayout);

    LinearLayout layout = fractalSharedToLinearLayout(shape, fractalAttr);

    auto offsetAttr = StringAttr::get(ctx, "offset");
    auto blockAttr = StringAttr::get(ctx, "block");

    auto logical = layout.apply(
        {{offsetAttr, static_cast<int32_t>(probeOffset)}, {blockAttr, 0}});

    llvm::outs() << "fractal-layout: offset=" << probeOffset << " ->";
    for (auto [name, val] : logical)
      llvm::outs() << " " << name.getValue() << "=" << val;
    llvm::outs() << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createDumpFractalLayoutPass(const DumpFractalLayoutOptions &options) {
  return std::make_unique<DumpFractalLayoutPass>(options);
}

} // namespace triton
} // namespace bishengir
