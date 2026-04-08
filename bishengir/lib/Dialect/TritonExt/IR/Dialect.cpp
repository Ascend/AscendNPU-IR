//===- Dialect.cpp - TritonExt dialect implementation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BishengIR TritonExt dialect, which provides
// Ascend-specific extensions to the TritonGPU shared memory layout system.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TritonExt/IR/TritonExtAttrs.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace bishengir::triton_ext;

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TritonExt/IR/Dialect.cpp.inc"

void bishengir::triton_ext::TritonExtDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/TritonExt/IR/AttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Enum definitions
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TritonExt/IR/OpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute definitions (tablegen-generated bodies)
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/TritonExt/IR/AttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Helper: parse CTA attributes from a dictionary
//===----------------------------------------------------------------------===//

static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       NamedAttribute attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute a : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(a);
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(), "expected an integer in ") << desc;
      return failure();
    }
    res.push_back(intAttr.getInt());
  }
  return success();
}

static std::optional<mlir::triton::gpu::CTALayoutAttr>
parseCTAAttrs(AsmParser &parser, NamedAttrList attrList, unsigned rank) {
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : attrList) {
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  auto *ctx = parser.getContext();
  return mlir::triton::gpu::CTALayoutAttr::get(
      ctx,
      CTAsPerCGA.value_or(SmallVector<unsigned>(rank, 1)),
      CTASplitNum.value_or(SmallVector<unsigned>(rank, 1)),
      CTAOrder.value_or(SmallVector<unsigned>(llvm::reverse(
          llvm::to_vector(llvm::seq<unsigned>(rank))))));
}

static void
maybePrintCTALayout(MLIRContext *context, AsmPrinter &printer,
                    mlir::triton::gpu::CTALayoutAttr layout, unsigned rank) {
  if (layout != mlir::triton::gpu::CTALayoutAttr::getDefault(context, rank)) {
    printer << ", CTAsPerCGA = [" << ArrayRef(layout.getCTAsPerCGA()) << "]"
            << ", CTASplitNum = [" << ArrayRef(layout.getCTASplitNum()) << "]"
            << ", CTAOrder = [" << ArrayRef(layout.getCTAOrder()) << "]";
  }
}

//===----------------------------------------------------------------------===//
// FractalSharedEncoding - parse / print / verify
//===----------------------------------------------------------------------===//

Attribute FractalSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  int64_t fractalM0 = 0;
  int64_t fractalN0 = 0;
  std::optional<FractalLayoutType> layoutType;
  NamedAttrList remainingAttrs;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "fractalM0") {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(attr.getValue());
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(), "expected integer for fractalM0");
        return {};
      }
      fractalM0 = intAttr.getInt();
    } else if (attr.getName() == "fractalN0") {
      auto intAttr = mlir::dyn_cast<IntegerAttr>(attr.getValue());
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(), "expected integer for fractalN0");
        return {};
      }
      fractalN0 = intAttr.getInt();
    } else if (attr.getName() == "layoutType") {
      auto strAttr = mlir::dyn_cast<StringAttr>(attr.getValue());
      if (!strAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected string for layoutType");
        return {};
      }
      layoutType = symbolizeFractalLayoutType(strAttr.getValue());
      if (!layoutType) {
        parser.emitError(parser.getNameLoc(),
                         "unknown layoutType: expected 'zN' or 'nZ'");
        return {};
      }
    } else {
      remainingAttrs.push_back(attr);
    }
  }

  if (!layoutType) {
    parser.emitError(parser.getNameLoc(), "missing required 'layoutType'");
    return {};
  }

  // Fractal layouts are inherently 2D.
  if (auto CTALayout = parseCTAAttrs(parser, remainingAttrs, /*rank=*/2))
    return parser.getChecked<FractalSharedEncodingAttr>(
        parser.getContext(), fractalM0, fractalN0, *layoutType, *CTALayout);
  return {};
}

void FractalSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "fractalM0 = " << getFractalM0()
          << ", fractalN0 = " << getFractalN0()
          << ", layoutType = \""
          << stringifyFractalLayoutType(getLayoutType()) << "\"";
  maybePrintCTALayout(getContext(), printer, getCTALayout(), /*rank=*/2);
  printer << "}>";
}

LogicalResult FractalSharedEncodingAttr::verify(
    ::llvm::function_ref<InFlightDiagnostic()> emitError, int64_t fractalM0,
    int64_t fractalN0, FractalLayoutType layoutType,
    mlir::triton::gpu::CTALayoutAttr ctaLayout) {
  if (!(llvm::isPowerOf2_32(fractalM0) && llvm::isPowerOf2_32(fractalN0)))
    return emitError() << "fractal block must be power-of-2";
  return success();
}
