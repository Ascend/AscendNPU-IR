//===- TensorViewTypes.cpp - TensorView types -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h"
#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

#define GET_TYPEDEF_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.cpp.inc"

namespace mlir::tv {

// Registered here (rather than in TensorViewDialect.cpp) so that the storage
// classes are complete for the StorageUniquer.
void TensorViewDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/TensorView/IR/TensorViewTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TensorViewType custom assembly / verifier
//===----------------------------------------------------------------------===//

Type TensorViewType::parse(AsmParser &parser) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // shape (`x`-separated, dynamic `?`) followed by the element type.
  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return {};
  Type elementType;
  if (parser.parseType(elementType))
    return {};

  // `, strides=[...]`
  if (parser.parseComma() || parser.parseKeyword("strides") ||
      parser.parseEqual())
    return {};
  SmallVector<int64_t> strides;
  auto parseStride = [&]() -> ParseResult {
    int64_t s;
    if (succeeded(parser.parseOptionalQuestion()))
      s = ShapedType::kDynamic;
    else if (parser.parseInteger(s))
      return failure();
    strides.push_back(s);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseStride))
    return {};

  // optional `, <encoding>`
  Attribute encoding;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(encoding))
      return {};
  }

  if (parser.parseGreater())
    return {};

  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    shape, elementType, strides, encoding);
}

void TensorViewType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t d : getShape()) {
    if (ShapedType::isDynamic(d))
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer << getElementType();
  printer << ", strides=[";
  llvm::interleaveComma(getStrides(), printer, [&](int64_t s) {
    if (ShapedType::isDynamic(s))
      printer << "?";
    else
      printer << s;
  });
  printer << "]";
  if (Attribute enc = getEncoding())
    printer << ", " << enc;
  printer << ">";
}

LogicalResult
TensorViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, Type elementType,
                       ArrayRef<int64_t> strides, Attribute encoding) {
  if (shape.size() != strides.size())
    return emitError() << "expected strides rank (" << strides.size()
                       << ") to match shape rank (" << shape.size() << ")";
  if (encoding && !isViewEncoding(encoding))
    return emitError() << "invalid tensor_view encoding: " << encoding;
  return success();
}

} // namespace mlir::tv
