//===- TensorViewAttrs.cpp - TensorView attributes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.h"

#include "bishengir/Dialect/TensorView/IR/TensorViewDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h" // required by `Attrs.cpp.inc`
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Attrs.cpp.inc`

#include "bishengir/Dialect/TensorView/IR/TensorViewEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.cpp.inc"

namespace mlir::tv {

// Registered here (rather than in TensorViewDialect.cpp) so that the storage
// classes are complete for the StorageUniquer.
void TensorViewDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Custom assembly helpers for the encoding attributes.
//
//   #tv.partition_view<tile = [128], dim_map = [0], padding_value = zero>
//===----------------------------------------------------------------------===//

// Parse `<keyword> = [i64, i64, ...]`.
static ParseResult parseKeywordIntArray(AsmParser &parser, StringRef keyword,
                                        SmallVectorImpl<int64_t> &out) {
  if (parser.parseKeyword(keyword) || parser.parseEqual())
    return failure();
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        [&]() -> ParseResult {
                                          int64_t v;
                                          if (parser.parseInteger(v))
                                            return failure();
                                          out.push_back(v);
                                          return success();
                                        });
}

static void printIntArray(AsmPrinter &printer, StringRef keyword,
                          ArrayRef<int64_t> values) {
  printer << keyword << " = [";
  llvm::interleaveComma(values, printer, [&](int64_t v) { printer << v; });
  printer << "]";
}

static ParseResult parsePaddingValue(AsmParser &parser, PaddingValue &value) {
  if (parser.parseKeyword("padding_value") || parser.parseEqual())
    return failure();
  std::string spelling;
  if (parser.parseKeywordOrString(&spelling))
    return failure();
  auto padding = symbolizePaddingValue(spelling);
  if (!padding)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected zero, nan, inf, or \"-inf\"");
  value = *padding;
  return success();
}

static void printPaddingValue(AsmPrinter &printer, PaddingValue value) {
  printer << "padding_value = ";
  if (value == PaddingValue::NEG_INF)
    printer << "\"-inf\"";
  else
    printer << stringifyPaddingValue(value);
}

//===----------------------------------------------------------------------===//
// PartitionViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute PartitionViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, dimMap;
  PaddingValue paddingValue;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() || parseKeywordIntArray(parser, "dim_map", dimMap) ||
      parser.parseComma() || parsePaddingValue(parser, paddingValue) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, dimMap, paddingValue);
}

void PartitionViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "dim_map", getDimMap());
  printer << ", ";
  printPaddingValue(printer, getPaddingValue());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// StridedViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute StridedViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, dimMap, traversalStrides;
  PaddingValue paddingValue;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() || parseKeywordIntArray(parser, "dim_map", dimMap) ||
      parser.parseComma() ||
      parseKeywordIntArray(parser, "traversal_strides", traversalStrides) ||
      parser.parseComma() || parsePaddingValue(parser, paddingValue) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, dimMap, traversalStrides, paddingValue);
}

void StridedViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "dim_map", getDimMap());
  printer << ", ";
  printIntArray(printer, "traversal_strides", getTraversalStrides());
  printer << ", ";
  printPaddingValue(printer, getPaddingValue());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// GatherScatterViewAttr assembly
//===----------------------------------------------------------------------===//
Attribute GatherScatterViewAttr::parse(AsmParser &parser, Type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> tile, sparseDim;
  PaddingValue paddingValue;
  if (parser.parseLess() || parseKeywordIntArray(parser, "tile", tile) ||
      parser.parseComma() ||
      parseKeywordIntArray(parser, "sparse_dim", sparseDim) ||
      parser.parseComma() || parsePaddingValue(parser, paddingValue) ||
      parser.parseGreater())
    return {};
  return getChecked([&] { return parser.emitError(loc); }, parser.getContext(),
                    tile, sparseDim, paddingValue);
}

void GatherScatterViewAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printIntArray(printer, "tile", getTile());
  printer << ", ";
  printIntArray(printer, "sparse_dim", getSparseDim());
  printer << ", ";
  printPaddingValue(printer, getPaddingValue());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Encoding verifiers (structural checks that do not depend on the base rank).
//===----------------------------------------------------------------------===//

LogicalResult
PartitionViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> tile, ArrayRef<int64_t> dimMap,
                          PaddingValue) {
  if (tile.empty())
    return emitError() << "partition_view tile must be non-empty";
  if (tile.size() != dimMap.size())
    return emitError()
           << "partition_view tile and dim_map must have equal length";
  if (llvm::any_of(tile, [](int64_t value) { return value <= 0; }))
    return emitError() << "partition_view tile sizes must be positive";
  return success();
}

LogicalResult
StridedViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<int64_t> tile, ArrayRef<int64_t> dimMap,
                        ArrayRef<int64_t> traversalStrides, PaddingValue) {
  if (tile.empty())
    return emitError() << "strided_view tile must be non-empty";
  if (tile.size() != dimMap.size())
    return emitError()
           << "strided_view tile and dim_map must have equal length";
  if (tile.size() != traversalStrides.size())
    return emitError()
           << "strided_view tile and traversal_strides must have equal length";
  if (llvm::any_of(tile, [](int64_t value) { return value <= 0; }))
    return emitError() << "strided_view tile sizes must be positive";
  if (llvm::any_of(traversalStrides, [](int64_t value) { return value <= 0; }))
    return emitError() << "strided_view traversal strides must be positive";
  return success();
}

LogicalResult
GatherScatterViewAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t> tile,
                              ArrayRef<int64_t> sparseDim, PaddingValue) {
  if (tile.empty())
    return emitError() << "gather_scatter_view tile must be non-empty";
  if (sparseDim.empty())
    return emitError() << "gather_scatter_view sparse_dim must be non-empty";
  if (llvm::any_of(tile, [](int64_t value) { return value <= 0; }))
    return emitError() << "gather_scatter_view tile sizes must be positive";
  llvm::SmallDenseSet<int64_t> uniqueDims;
  for (int64_t dim : sparseDim)
    if (!uniqueDims.insert(dim).second)
      return emitError()
             << "gather_scatter_view sparse_dim entries must be unique";
  return success();
}

//===----------------------------------------------------------------------===//
// View-encoding helpers.
//===----------------------------------------------------------------------===//

bool isViewEncoding(Attribute enc) {
  return llvm::isa_and_nonnull<PartitionViewAttr, StridedViewAttr,
                               GatherScatterViewAttr>(enc);
}

llvm::SmallVector<int64_t> getEncodingTileShape(Attribute enc) {
  return llvm::TypeSwitch<Attribute, llvm::SmallVector<int64_t>>(enc)
      .Case<PartitionViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Case<StridedViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Case<GatherScatterViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Default([](Attribute) { return llvm::SmallVector<int64_t>{}; });
}

llvm::SmallVector<int64_t> getEncodingTraversal(Attribute enc) {
  return llvm::TypeSwitch<Attribute, llvm::SmallVector<int64_t>>(enc)
      .Case<PartitionViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Case<StridedViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTraversalStrides()); })
      .Case<GatherScatterViewAttr>(
          [](auto a) { return llvm::to_vector(a.getTile()); })
      .Default([](Attribute) { return llvm::SmallVector<int64_t>{}; });
}

llvm::SmallVector<int64_t> getEncodingSparseDims(Attribute enc) {
  return llvm::TypeSwitch<Attribute, llvm::SmallVector<int64_t>>(enc)
      .Case<GatherScatterViewAttr>(
          [](auto a) { return llvm::to_vector(a.getSparseDim()); })
      .Default([](Attribute) { return llvm::SmallVector<int64_t>{}; });
}

PaddingValue getEncodingPaddingValue(Attribute enc) {
  return llvm::TypeSwitch<Attribute, PaddingValue>(enc)
      .Case<PartitionViewAttr>([](auto a) { return a.getPaddingValue(); })
      .Case<StridedViewAttr>([](auto a) { return a.getPaddingValue(); })
      .Case<GatherScatterViewAttr>([](auto a) { return a.getPaddingValue(); })
      .Default([](Attribute) { return PaddingValue::ZERO; });
}

} // namespace mlir::tv
