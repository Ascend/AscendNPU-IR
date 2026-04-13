//===- HFusionOps.cpp - Implementation of HFusion Dialect Ops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <variant>

#if BSPUB_DAVINCI_BISHENGIR
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif

using namespace mlir;
using namespace mlir::hfusion;
//===----------------------------------------------------------------------===//
// Support for named HFusion ops defined in ods-gen.
//===----------------------------------------------------------------------===//

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes,
                      [](Type t) { return llvm::isa<ShapedType>(t); }));

  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(
          isa<MemRefType, RankedTensorType>(t) ? getElementTypeOrSelf(t) : t);

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Creates a structured operation given `inputs`, `outputs`, and `attributes`.
/// The result types are derived automatically if `resultTensorTypes` is none.
/// The body of the operation is filled using `regionBuilder`. All ods-gen
/// created structured operations use the method to implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              std::optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            [](Type type) { return llvm::isa<RankedTensorType>(type); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  state.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                              static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

void addOperandSegmentSizesAttr(
    OpAsmParser &parser, OperationState &result,
    const SmallVector<OpAsmParser::UnresolvedOperand, 4> &inputsOperands,
    const SmallVector<OpAsmParser::UnresolvedOperand, 4> &outputsOperands) {
  if (result.propertiesAttr) {
    NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
    attrs.append("operandSegmentSizes",
                 parser.getBuilder().getDenseI32ArrayAttr(
                     {static_cast<int32_t>(inputsOperands.size()),
                      static_cast<int32_t>(outputsOperands.size())}));
    result.propertiesAttr = attrs.getDictionary(parser.getContext());
  } else {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputsOperands.size()),
                             static_cast<int32_t>(outputsOperands.size())}));
  }
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc attrsLoc;
  SMLoc inputsOperandsLoc;
  SMLoc outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }
  attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    // This is a bit complex because we're trying to be backward compatible with
    // operation syntax that mix the inherent attributes and the discardable
    // ones in the same dictionary. If the properties are used, we append the
    // operandSegmentSizes there directly. Otherwise we append it to the
    // discardable attributes dictionary where it is handled by the generic
    // Operation::create(...) method.
    addOperandSegmentSizesAttr(parser, result, inputsOperands, outputsOperands);
  }
  if (!result.propertiesAttr) {
    std::optional<RegisteredOperationName> info =
        result.name.getRegisteredInfo();
    if (info) {
      if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
            return parser.emitError(attrsLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          })))
        return failure();
    }
  }
  return success();
}

static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

//===----------------------------------------------------------------------===//
// Specific parsing and printing for named structured ops created by ods-gen.
//===----------------------------------------------------------------------===//

static ParseResult parseNamedStructuredOpRegion(
    OpAsmParser &parser, Region &region, unsigned numRegionArgs,
    TypeRange inputTypes, TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    RegionBuilderFn regionBuilder) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, region, inputTypes, outputTypes, attrs,
                         regionBuilder);
  return success();
}

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes;
  SmallVector<Type, 1> outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

static void printNamedStructuredOp(OpAsmPrinter &p, Operation *op,
                                   ValueRange inputs, ValueRange outputs) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operandSegmentSizes",
                       // See generated code in
                       // HFusionNamedStructuredOps.yamlgen.cpp.inc
                       "hfusion.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, inputs, outputs);

  // Results printing.
  printNamedStructuredOpResults(p, op->getResultTypes());

  // Region is elided.
}

//===----------------------------------------------------------------------===//
// Region builder helper.
// TODO: Move this to a utility library.
// The public methods on this class are referenced directly from generated code.
// Helper build the unary, binary, and type conversion functions defined by the
// DSL. See HFusionNamedStructuredOps.yamlgen.cpp.inc for the code that uses
// this class.
//
// Implementations of the math functions must be polymorphic over numeric types,
// internally performing necessary casts. If the function application makes no
// sense, then the only recourse is to assert and return nullptr. This can be
// extended later if it becomes possible to fail construction of the region. The
// invariant should be enforced at a higher level.
//
// TODO: These helpers are currently type polymorphic over the class of integer
// and floating point types, but they will not internally cast within bit
// widths of a class (mixed precision such as i8->i32) or across classes
// (i.e. mixed float and integer). Many such combinations are ambiguous or need
// to be handled with care and work is being considered to extend the op
// language to make such cases explicit. In the mean-time, violating this will
// fail verification, which is deemed acceptable.
//===----------------------------------------------------------------------===//

namespace {

class RegionBuilderHelper {
public:
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Build the unary functions defined by OpDSL.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg) {
    OpBuilder builder = getBuilder();
    switch (unaryFn) {
    case UnaryFn::sqrt:
      return builder.create<math::SqrtOp>(arg.getLoc(), arg);
    case UnaryFn::rsqrt:
      return builder.create<math::RsqrtOp>(arg.getLoc(), arg);
    case UnaryFn::tanh:
      return builder.create<math::TanhOp>(arg.getLoc(), arg);
    case UnaryFn::tan:
      return builder.create<math::TanOp>(arg.getLoc(), arg);
    case UnaryFn::sin:
      return builder.create<math::SinOp>(arg.getLoc(), arg);
    case UnaryFn::cos:
      return builder.create<math::CosOp>(arg.getLoc(), arg);
    case UnaryFn::atan:
      return builder.create<math::AtanOp>(arg.getLoc(), arg);
    case UnaryFn::absi:
      return builder.create<math::AbsIOp>(arg.getLoc(), arg);
    case UnaryFn::erf:
      return builder.create<math::ErfOp>(arg.getLoc(), arg);
    case UnaryFn::log2:
      return builder.create<math::Log2Op>(arg.getLoc(), arg);
    case UnaryFn::log10:
      return builder.create<math::Log10Op>(arg.getLoc(), arg);
    case UnaryFn::log1p:
      return builder.create<math::Log1pOp>(arg.getLoc(), arg);
    case UnaryFn::exp2:
      return builder.create<math::Exp2Op>(arg.getLoc(), arg);
    case UnaryFn::expm1:
      return builder.create<math::ExpM1Op>(arg.getLoc(), arg);
    case UnaryFn::ilogb:
      return builder.create<mathExt::IlogbOp>(arg.getLoc(), arg);
    case UnaryFn::relu:
      return buildUnaryRelu(builder, arg);
    case UnaryFn::rec:
      return buildUnaryRec(builder, arg);
    case UnaryFn::vnot:
      return buildUnaryVNot(builder, arg);
    }
    llvm_unreachable("unsupported unary function");
  }

  Value buildUnaryRelu(OpBuilder &builder, Value arg) {
    if (isFloatingPoint(arg)) {
      Type type = arg.getType();
      Value zero = builder.create<arith::ConstantOp>(
          arg.getLoc(), type, builder.getFloatAttr(type, 0.0));
      return builder.create<arith::MaximumFOp>(arg.getLoc(), zero, arg);
    }
    if (isInteger(arg)) {
      Type type = arg.getType();
      Value zero = builder.create<arith::ConstantOp>(
          arg.getLoc(), type, builder.getIntegerAttr(type, 0));
      return builder.create<arith::MaxSIOp>(arg.getLoc(), zero, arg);
    }
    llvm_unreachable("unsupported type for relu");
  }

  Value buildUnaryRec(OpBuilder &builder, Value arg) {
    if (isFloatingPoint(arg)) {
      Type type = arg.getType();
      Value one = builder.create<arith::ConstantOp>(
          arg.getLoc(), type, builder.getFloatAttr(type, 1.0));
      return builder.create<arith::DivFOp>(arg.getLoc(), one, arg);
    }
    if (isInteger(arg)) {
      Type type = arg.getType();
      Value one = builder.create<arith::ConstantOp>(
          arg.getLoc(), type, builder.getIntegerAttr(type, 1));
      return builder.create<arith::DivSIOp>(arg.getLoc(), one, arg);
    }
    llvm_unreachable("unsupported type for reciprocal");
  }

  Value buildUnaryVNot(OpBuilder &builder, Value arg) {
    if (isInteger(arg)) {
      Type type = arg.getType();
      Value negOne = builder.create<arith::ConstantOp>(
          arg.getLoc(), type, builder.getIntegerAttr(type, -1));
      return builder.create<arith::XOrIOp>(arg.getLoc(), negOne, arg);
    } else {
      llvm_unreachable("unsupported type for not");
    }
  }

  // Build the binary functions defined by OpDSL.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (binaryFn) {
    case BinaryFn::vor:
      if (allInteger)
        return builder.create<arith::OrIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vor");
    case BinaryFn::vxor:
      if (allInteger)
        return builder.create<arith::XOrIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vxor");
    case BinaryFn::vand:
      if (allInteger)
        return builder.create<arith::AndIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vand");
    case BinaryFn::minf:
      if (allFloatingPoint)
        return builder.create<arith::MinimumFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vmin");
    case BinaryFn::maxf:
      if (allFloatingPoint)
        return builder.create<arith::MaximumFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vmax");
    case BinaryFn::minnumf:
      if (allFloatingPoint)
        return builder.create<arith::MinNumFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vmin");
    case BinaryFn::maxnumf:
      if (allFloatingPoint)
        return builder.create<arith::MaxNumFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vmax");
    case BinaryFn::powf:
      if (allFloatingPoint)
        return builder.create<math::PowFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vpow");
    case BinaryFn::powi:
      if (allInteger)
        return builder.create<math::IPowIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for vpowi");
    case BinaryFn::mod:
      if (allInteger)
        return builder.create<arith::RemSIOp>(arg0.getLoc(), arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::RemFOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for mod");
    case BinaryFn::shli:
      if (allInteger)
        return builder.create<arith::ShLIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for shli");
    case BinaryFn::shrsi:
      if (allInteger)
        return builder.create<arith::ShRSIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for shrsi");
    case BinaryFn::shrui:
      if (allInteger)
        return builder.create<arith::ShRUIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for shrui");
    case BinaryFn::ldexp:
      if (allFloatingPoint)
        return builder.create<mathExt::LdexpOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for ldexp");
    case BinaryFn::floordivsi:
      if (allInteger)
        return builder.create<arith::FloorDivSIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for floordivsi");
    case BinaryFn::ceildivsi:
      if (allInteger)
        return builder.create<arith::CeilDivSIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for ceildivsi");
    case BinaryFn::ceildivui:
      if (allInteger)
        return builder.create<arith::CeilDivUIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for ceildivui");
    case BinaryFn::modui:
      if (allInteger)
        return builder.create<arith::RemUIOp>(arg0.getLoc(), arg0, arg1);
      llvm_unreachable("unsupported type for modui");
    case BinaryFn::divfhp:
      Type type = arg0.getType();
      return builder.create<mathExt::DivFHPOp>(arg0.getLoc(), type, arg0, arg1);
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the compare functions defined by OpDSL.
  Value buildCompareFn(CompareFn compareFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (compareFn) {
    case CompareFn::veq:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::eq, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OEQ, arg0, arg1);
      llvm_unreachable("unsupported type for veq");
    case CompareFn::vne:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::ne, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::UNE, arg0, arg1);
      llvm_unreachable("unsupported type for vne");
    case CompareFn::vle:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sle, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OLE, arg0, arg1);
      llvm_unreachable("unsupported type for vle");
    case CompareFn::vlt:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::slt, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OLT, arg0, arg1);
      llvm_unreachable("unsupported type for vlt");
    case CompareFn::vge:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sge, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OGE, arg0, arg1);
      llvm_unreachable("unsupported type for vge");
    case CompareFn::vgt:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sgt, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OGT, arg0, arg1);
      llvm_unreachable("unsupported type for vgt");
    case CompareFn::vule:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::ule, arg0, arg1);
      llvm_unreachable("unsupported type for ule");
    case CompareFn::vuge:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::uge, arg0, arg1);
      llvm_unreachable("unsupported type for uge");
    case CompareFn::vugt:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::ugt, arg0, arg1);
      llvm_unreachable("unsupported type for ugt");
    case CompareFn::vult:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::ult, arg0, arg1);
      llvm_unreachable("unsupported type for ult");
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the Ternary functions defined by OpDSL.
  Value buildTernaryFn(TernaryFn ternaryFn, Value arg0, Value arg1,
                       Value arg2) {
    bool allComplex = isComplex(arg1) && isComplex(arg2);
    bool allFloatingPoint = isFloatingPoint(arg1) && isFloatingPoint(arg2);
    bool allInteger = isInteger(arg1) && isInteger(arg2);
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (ternaryFn) {
    case TernaryFn::select:
      if (allInteger || allFloatingPoint)
        return builder.create<arith::SelectOp>(arg0.getLoc(), arg0, arg1, arg2);
      llvm_unreachable("unsupported type for select");
    }
    llvm_unreachable("unsupported select function");
  }

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand) {
    switch (typeFn) {
    case TypeFn::cast_signed:
      return cast(toType, operand, false);
    case TypeFn::cast_unsigned:
      return cast(toType, operand, true);
    case TypeFn::bitcast:
      OpBuilder builder = getBuilder();
      Location loc = operand.getLoc();
      auto op = builder.create<arith::BitcastOp>(loc, toType, operand);
      return op;
    }
    llvm_unreachable("unsupported type conversion function");
  }

  // Build the type functions defined by OpDSL.
  Value buildRoundMode(TypeFn cast, RoundMode round, UnsignedMode unsignedMode,
                       Type toType, Value operand) {
    bool isUnsignedCast = false;
    if (operand.getType().isInteger(1) && toType.getIntOrFloatBitWidth() > 1) {
      // TODO: general support for unsigned cast
      isUnsignedCast = true;
    }

    if (cast == TypeFn::cast_unsigned) {
      isUnsignedCast = true;
    }

    Value castedOp = castRound(toType, operand, isUnsignedCast);
    Operation *defOp = castedOp.getDefiningOp();
    OpBuilder builder = getBuilder();

    if (!defOp) {
      return castedOp;
    }
    auto roundingAttr = builder.getAttr<hfusion::RoundModeAttr>(round);
    auto unsignedAttr = builder.getAttr<hfusion::UnsignedModeAttr>(unsignedMode);

    if (!roundingAttr) {
      llvm_unreachable("Round type not supported");
    }
    if (!unsignedAttr) {
      llvm_unreachable("Unsigned mode not supported");
    }

    defOp->setAttr("round_mode", roundingAttr);
    defOp->setAttr("unsigned_mode", unsignedAttr);
    return castedOp;
  }

  // Build the enable_saturate attr defined by OpDSL.
  void buildEnableSaturate(bool enable_saturateVal, Value operand) {
    Operation *defOp = operand.getDefiningOp();
    OpBuilder builder = getBuilder();

    if (!defOp) {
      return;
    }

    defOp->setAttr("enable_saturate", builder.getBoolAttr(enable_saturateVal));
  }

  // Build the type functions defined by OpDSL.
  Value buildAtomicKind(AtomicKind atkind, Type toType, Value operand) {
    return cast(toType, operand, false);
  }

  void yieldOutputs(ValueRange values) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    builder.create<linalg::YieldOp>(loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<arith::ConstantOp>(loc, ::cast<TypedAttr>(valueAttr));
  }

  Value index(int64_t dim) {
    OpBuilder builder = getBuilder();
    return builder.create<linalg::IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(context, width);
  }

  Type getFloat32Type() { return Float32Type::get(context); }
  Type getFloat64Type() { return Float64Type::get(context); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();
    return convertScalarToDtype(builder, loc, operand, toType, isUnsignedCast);
  }

  Value castRound(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();
    if (operand.getType() == toType && dyn_cast<FloatType>(toType)) {
      return builder.create<math::RoundOp>(loc, operand);
    }
    return convertScalarToDtype(builder, loc, operand, toType, isUnsignedCast);
  }

  bool isComplex(Value value) {
    return llvm::isa<ComplexType>(value.getType());
  }
  bool isFloatingPoint(Value value) {
    return llvm::isa<FloatType>(value.getType());
  }
  bool isInteger(Value value) {
    return llvm::isa<IntegerType>(value.getType());
  }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }

  MLIRContext *context;
  Block &block;
};

template <typename CumOpTy>
LogicalResult verifyCumOp(CumOpTy op) {
  ArrayRef<int64_t> cumDims = op.getCumDims();
  if (cumDims.empty()) {
    return op.emitOpError() << "have empty cum dims array";
  }

  ShapedType inputType = cast<ShapedType>(op.getInput().getType());
  if (static_cast<int64_t>(cumDims.size()) > inputType.getRank()) {
    return op.emitOpError() << "have too many indices in the cum dims array";
  }

  std::set<int64_t> cumDimSet;
  ShapedType outputType = cast<ShapedType>(op.getOutput().getType());
  for (int64_t idx : cumDims) {
    if (idx < 0 || idx >= outputType.getRank()) {
      return op.emitOpError()
             << "have invalid index '" << idx << "' inside cum dims array";
    }
    if (cumDimSet.find(idx) != cumDimSet.end()) {
      return op.emitOpError()
             << "have duplicate index '" << idx << "' inside cum dims array";
    }
    cumDimSet.insert(idx);
  }

  if (cumDimSet.size() > 1) {
    return op.emitOpError() << "have more than one cumulative dims";
  }
  return success();
}

} // namespace

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    linalg::LinalgOp linalgOp) {
  for (auto [index, operand] : llvm::enumerate(linalgOp.getDpsInputs())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(
        MemoryEffects::Read::get(), &linalgOp->getOpOperand(index), /*stage=*/0,
        /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get());
  }

  for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(&operand)) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

namespace mlir {
namespace hfusion {

ParseResult parseHFusionDeinterleave(OpAsmParser &parser,
                                     IntegerAttr &channelIndex) {
  if (failed(parser.parseKeyword("channel")) || failed(parser.parseLess())) {
    parser.emitError(parser.getCurrentLocation())
        << "expects the keyword `channel<`";
    return failure();
  }
  // Check if it's "all" or a number
  auto builder = parser.getBuilder();
  if (succeeded(parser.parseOptionalKeyword("all"))) {
    channelIndex = builder.getI64IntegerAttr(-1);
  } else {
    // Parse a number
    int64_t channelVal;
    if (failed(parser.parseInteger(channelVal))) {
      parser.emitError(parser.getCurrentLocation())
          << "expects a channel integer or keyword `all`";
      return failure();
    }
    channelIndex = builder.getI64IntegerAttr(channelVal);
  }
  if (failed(parser.parseGreater())) {
    parser.emitError(parser.getCurrentLocation())
        << "expects a closing bracket `>`";
    return failure();
  }
  return success();
}

void printHFusionDeinterleave(OpAsmPrinter &printer, Operation *op,
                              IntegerAttr foo) {
  auto &s = printer.getStream();
  s << "channel<";
  if (foo.getInt() == -1)
    s << "all";
  else
    s << foo.getInt();
  s << ">";
}

} // namespace hfusion
} // namespace mlir

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionOps.cpp.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ReduceWithIndexOp
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hfusion {

void ReduceWithIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              TypeRange types, ValueRange inputs,
                              ValueRange inits,
                              ReduceWithIndexKindAttr reduce_kind,
                              BoolAttr unsigned_src, BoolAttr tie_break_left,
                              DenseI64ArrayAttr dimensions) {
  odsState.addAttribute("reduce_kind", reduce_kind);
  odsState.addAttribute("unsigned_src", unsigned_src);
  odsState.addAttribute("tie_break_left", tie_break_left);
  odsState.addAttribute("dimensions", dimensions);
  buildStructuredOp(odsBuilder, odsState, types, inputs, inits, {},
                    ReduceWithIndexOp::getRegionBuilder());
}

void ReduceWithIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              TypeRange types, ValueRange inputs,
                              ValueRange inits,
                              ReduceWithIndexKindAttr reduce_kind,
                              BoolAttr unsigned_src, BoolAttr tie_break_left,
                              ArrayRef<int64_t> dimensions) {
  odsState.addAttribute("reduce_kind", reduce_kind);
  odsState.addAttribute("unsigned_src", unsigned_src);
  odsState.addAttribute("tie_break_left", tie_break_left);
  odsState.addAttribute("dimensions",
                        odsBuilder.getDenseI64ArrayAttr(dimensions));
  buildStructuredOp(odsBuilder, odsState, types, inputs, inits, {},
                    ReduceWithIndexOp::getRegionBuilder());
}

MutableOperandRange ReduceWithIndexOp::getDpsInitsMutable() {
  return getInitsMutable();
}

SmallVector<utils::IteratorType> ReduceWithIndexOp::getIteratorTypesArray() {
  int64_t inputRank =
      llvm::cast<ShapedType>(getInputs()[0].getType()).getRank();
  auto result = SmallVector<utils::IteratorType>(inputRank,
                                                 utils::IteratorType::parallel);
  auto reductionDims = getDimensions();
  for (int64_t d : reductionDims)
    result[d] = utils::IteratorType::reduction;
  return result;
}

ArrayAttr ReduceWithIndexOp::getIndexingMaps() {
  int64_t inputRank =
      llvm::cast<ShapedType>(getInputs()[0].getType()).getRank();
  SmallVector<AffineMap> affineMaps(
      getNumDpsInputs(),
      AffineMap::getMultiDimIdentityMap(inputRank, getContext()));
  AffineMap resultMap =
      AffineMap::getMultiDimIdentityMap(inputRank, getContext())
          .dropResults(
              getDimensions()); // reduction dimensions don't get result indices
  for (int64_t i = 0, e = getNumDpsInits(); i < e; ++i)
    affineMaps.push_back(resultMap);
  return Builder(getContext()).getAffineMapArrayAttr(affineMaps);
}

template <typename BinaryOp, typename CmpOp, typename CmpPred>
void codeGenWithoutIndex(
    OpBuilder &builder, Value inValue, Value outValue, Value outIndex,
    Type indexType, int64_t theDimension,
    ::std::variant<arith::CmpFPredicate, arith::CmpIPredicate> cmpPred) {
  auto resultValue =
      builder.create<BinaryOp>(inValue.getLoc(), inValue, outValue);
  auto predicate =
      builder.create<CmpOp>(resultValue.getLoc(), ::std::get<CmpPred>(cmpPred),
                            resultValue, outValue);
  auto linalgIndex = builder.create<arith::IndexCastOp>(
      predicate.getLoc(), indexType,
      builder.create<linalg::IndexOp>(predicate.getLoc(), theDimension));
  auto resultIndex = builder.create<arith::SelectOp>(
      linalgIndex.getLoc(), predicate, linalgIndex, outIndex);
  builder.create<linalg::YieldOp>(resultValue.getLoc(),
                                  ValueRange({resultValue, resultIndex}));
}

template <typename BinaryOp, typename CmpOp, typename CmpPred>
void codeGenWithIndex(
    OpBuilder &builder, Value inValue, Value inIndex, Value outValue,
    Value outIndex,
    ::std::variant<arith::CmpFPredicate, arith::CmpIPredicate> cmpPred) {
  auto resultValue =
      builder.create<BinaryOp>(inValue.getLoc(), inValue, outValue);
  auto predicate =
      builder.create<CmpOp>(resultValue.getLoc(), ::std::get<CmpPred>(cmpPred),
                            resultValue, outValue);
  auto resultIndex = builder.create<arith::SelectOp>(
      inIndex.getLoc(), predicate, inIndex, outIndex);
  builder.create<linalg::YieldOp>(resultValue.getLoc(),
                                  ValueRange({resultValue, resultIndex}));
}

void codeGenWithoutIndexDispatch(OpBuilder &builder, Block &block,
                                 Type elemType, int64_t theDimension,
                                 ReduceWithIndexKind reduce_kind) {
  /// Region (use <max> as example):
  /// ^bb0(inValue, outValue, outIndex):
  ///   resultValue = max(inValue, outValue)
  ///   predicate = resultValue > outValue
  ///   linalgIndex = linalg.index(d)
  ///   resultIndex = predicate ? linalgIndex, outIndex
  ///   yield resultValue, resultIndex
  Value inValue = block.getArgument(0);
  Value outValue = block.getArgument(1);
  Value outIndex = block.getArgument(2);
  // get index type
  Type indexType = outIndex.getType();
  // generate code
  if (isa<FloatType>(elemType)) {
    if (reduce_kind == ReduceWithIndexKind::MAX) {
      codeGenWithoutIndex<arith::MaximumFOp, arith::CmpFOp,
                          arith::CmpFPredicate>(
          builder, inValue, outValue, outIndex, indexType, theDimension,
          arith::CmpFPredicate::OGT);
    } else {
      codeGenWithoutIndex<arith::MinimumFOp, arith::CmpFOp,
                          arith::CmpFPredicate>(
          builder, inValue, outValue, outIndex, indexType, theDimension,
          arith::CmpFPredicate::OLT);
    }
  } else if (isa<IntegerType>(elemType)) {
    IntegerType::SignednessSemantics sgn =
        cast<IntegerType>(elemType).getSignedness();
    if (reduce_kind == ReduceWithIndexKind::MAX) {
      if (sgn == IntegerType::SignednessSemantics::Signed ||
          sgn == IntegerType::SignednessSemantics::Signless) {
        codeGenWithoutIndex<arith::MaxSIOp, arith::CmpIOp,
                            arith::CmpIPredicate>(
            builder, inValue, outValue, outIndex, indexType, theDimension,
            arith::CmpIPredicate::sgt);
      } else {
        llvm::report_fatal_error(
            "Unsigned reduce_with_index is not currently supported by HFusion");
      }
    } else {
      if (sgn == IntegerType::SignednessSemantics::Signed ||
          sgn == IntegerType::SignednessSemantics::Signless) {
        codeGenWithoutIndex<arith::MinSIOp, arith::CmpIOp,
                            arith::CmpIPredicate>(
            builder, inValue, outValue, outIndex, indexType, theDimension,
            arith::CmpIPredicate::slt);
      } else {
        llvm::report_fatal_error(
            "Unsigned reduce_with_index is not currently supported by HFusion");
      }
    }
  } else {
    llvm::report_fatal_error("unsupported element type for reduce_with_index");
  }
}

void codeGenWithIndexDispatch(OpBuilder &builder, Block &block, Type elemType,
                              ReduceWithIndexKind reduce_kind) {
  /// Region (use <max> as example):
  /// ^bb0(inValue, inIndex, outValue, outIndex):
  ///   resultValue = max(inValue, outValue)
  ///   predicate = resultValue > outValue
  ///   resultIndex = predicate ? inIndex : outIndex
  ///   yield resultValue, resultIndex
  Value inValue = block.getArgument(0);
  Value inIndex = block.getArgument(1);
  Value outValue = block.getArgument(2);
  Value outIndex = block.getArgument(3);
  // generate code
  if (isa<FloatType>(elemType)) {
    if (reduce_kind == ReduceWithIndexKind::MAX) {
      codeGenWithIndex<arith::MaximumFOp, arith::CmpFOp, arith::CmpFPredicate>(
          builder, inValue, inIndex, outValue, outIndex,
          arith::CmpFPredicate::OGT);
    } else {
      codeGenWithIndex<arith::MinimumFOp, arith::CmpFOp, arith::CmpFPredicate>(
          builder, inValue, inIndex, outValue, outIndex,
          arith::CmpFPredicate::OLT);
    }
  } else if (isa<IntegerType>(elemType)) {
    IntegerType::SignednessSemantics sgn =
        cast<IntegerType>(elemType).getSignedness();
    if (reduce_kind == ReduceWithIndexKind::MAX) {
      if (sgn == IntegerType::SignednessSemantics::Signed ||
          sgn == IntegerType::SignednessSemantics::Signless) {
        codeGenWithIndex<arith::MaxSIOp, arith::CmpIOp, arith::CmpIPredicate>(
            builder, inValue, inIndex, outValue, outIndex,
            arith::CmpIPredicate::sgt);
      } else {
        llvm::report_fatal_error(
            "Unsigned reduce_with_index is not currently supported by HFusion");
      }
    } else {
      if (sgn == IntegerType::SignednessSemantics::Signed ||
          sgn == IntegerType::SignednessSemantics::Signless) {
        codeGenWithIndex<arith::MinSIOp, arith::CmpIOp, arith::CmpIPredicate>(
            builder, inValue, inIndex, outValue, outIndex,
            arith::CmpIPredicate::slt);
      } else {
        llvm::report_fatal_error(
            "Unsigned reduce_with_index is not currently supported by HFusion");
      }
    }
  } else {
    llvm::report_fatal_error("unsupported element type for reduce_with_index");
  }
}

std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)>
ReduceWithIndexOp::getRegionBuilder() {
  return [](ImplicitLocOpBuilder &b, Block &block,
            ArrayRef<NamedAttribute> attrs) {
    // check numArgs
    constexpr int kNumArgsWithoutIndex = 3;
    auto numArgs = block.getNumArguments();
#ifndef NDEBUG
    constexpr int kNumArgsWithIndex = 4;
    assert((numArgs == kNumArgsWithoutIndex || numArgs == kNumArgsWithIndex) &&
           "ReduceWithIndexOp regionBuilder expects 3 or 4 block args");
#endif
    // obtain reduce_kind
    ReduceWithIndexKind reduce_kind = ReduceWithIndexKind::MIN;
    auto reduce_kind_iter =
        llvm::find_if(attrs, [&](const NamedAttribute &attr) {
          return attr.getName() == "reduce_kind";
        });
    assert(reduce_kind_iter != attrs.end() && "reduce_kind not found");
    auto reduce_kind_attr =
        llvm::dyn_cast<ReduceWithIndexKindAttr>(reduce_kind_iter->getValue());
    assert(reduce_kind_attr && "failed to get reduce_kind_attr");
    reduce_kind = reduce_kind_attr.getReduceWithIndexKind();
    // get elem type
    Type elemType =
        block.getArgument(0)
            .getType(); // these are wrappers of pointers to shared storage
    // TODO: currently only supports one reduction dimension
    // get **the** reduction dimension
    int64_t theDimension = -1;
    auto dimensions_iter =
        llvm::find_if(attrs, [&](const NamedAttribute &attr) {
          return attr.getName() == "dimensions";
        });
    auto dimensions_attr =
        llvm::dyn_cast<DenseI64ArrayAttr>(dimensions_iter->getValue());
    theDimension = dimensions_attr[0];
    // build region (TODO: how to properly update loc?)
    OpBuilder builder(block.getArgument(0).getContext());
    builder.setInsertionPointToEnd(&block);
    if (numArgs == kNumArgsWithoutIndex) {
      codeGenWithoutIndexDispatch(builder, block, elemType, theDimension,
                                  reduce_kind);
    } else {
      codeGenWithIndexDispatch(builder, block, elemType, reduce_kind);
    }
  };
}

std::string ReduceWithIndexOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ReduceWithIndexOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<linalg::LinalgOp>(getOperation()));
}

ParseResult ReduceWithIndexOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // ParseResult parser.parse...()
  // For "ParseResult", failure is true in a Boolean context

  // cannot reuse Linalg's parseDstStyleOp since it ignores <max>/<min> before
  // ins/outs

  // parse attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // parse <max> or <min>
  // The second argument of ReduceWithIndexKindAttr::parse is **not** used in
  // its implementation.
  result.addAttribute("reduce_kind",
                      ReduceWithIndexKindAttr::parse(parser, Type{}));

  // parse ins and outs
  // TODO: parseCommonStructuredOpParts also handles
  // optional result.propertiesAttr and optional result.attributes,
  // which are **not** needed here.
  SmallVector<Type, 2> inputTypes;
  SmallVector<Type, 2> outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // parse dimensions
  if (parser.parseKeyword("dimensions") || parser.parseEqual())
    return failure();
  result.addAttribute("dimensions", DenseI64ArrayAttr::parse(parser, Type{}));

  // parse optional result types
  if (!(parser.parseOptionalArrow())) { // TODO: this is a complicated bool
                                        // condition
    SmallVector<Type, 2> outputTensorsTypes;
    if (parser.parseTypeList(outputTensorsTypes)) // incorrect parser type
      return failure();
    result.addTypes(outputTensorsTypes);
  }

  // build the region
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, *(result.addRegion()), inputTypes,
                         outputTypes, result.attributes.getAttrs(),
                         getRegionBuilder());

  return success();
}

void ReduceWithIndexOp::print(OpAsmPrinter &p) {
  // attr-dict
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "reduce_kind",
                                           "dimensions",
                                           "hfusion.memoized_indexing_maps"});
  p << ' ';

  // reduce_kind
  auto reduceKindAttr = getReduceKindAttr();
  reduceKindAttr.print(p);
  p << ' ';

  // inputs
  auto inputs = getInputs();
  if (!inputs.empty())
    p << "ins(" << inputs << " : " << inputs.getTypes() << ") ";

  // inits
  auto inits = getInits();
  if (!inits.empty())
    p << "outs(" << inits << " : " << inits.getTypes() << ") ";

  // dimensions
  auto dimensionsAttr = getDimensionsAttr();
  p << "dimensions = ";
  dimensionsAttr.print(p);
  p << ' ';

  // result type
  auto resultTypes = getOperation()->getResultTypes();
  if (resultTypes.begin() != resultTypes.end()) {
    p << " -> ";
    llvm::interleaveComma(resultTypes, p);
  }
}

/// If result index is not used, replace hfusion.reduce_with_index with
/// linalg.reduce.
struct ReplaceWithLinalgReduce : public OpRewritePattern<ReduceWithIndexOp> {
  using OpRewritePattern<ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOp reduceWithIndexOp,
                                PatternRewriter &rewriter) const override {
    if (!(reduceWithIndexOp.hasPureTensorSemantics()))
      return failure();

    Value indexResult = reduceWithIndexOp.getResult()[1];
    if (!(indexResult.getUses().empty()))
      return failure();

    Value inValue = reduceWithIndexOp.getInputs()[0];
    Value outValue = reduceWithIndexOp.getInits()[0];
    Value shapeResult = reduceWithIndexOp.getResult()[0];
    auto &region = reduceWithIndexOp.getRegion();
    assert(region.hasOneBlock() && "reduce_with_index has more than one block");
    auto &block = *(region.begin());
    Operation &binOp = *(block.begin());
    Value binOpBlockArg0 = block.getArgument(0);
    Value binOpBlockArg1 = reduceWithIndexOp.getInputs().size() == 1
                               ? block.getArgument(1)
                               : block.getArgument(2);
    auto linalgReduceOp = rewriter.create<linalg::ReduceOp>(
        inValue.getLoc(), ValueRange{inValue}, ValueRange{outValue},
        reduceWithIndexOp.getDimensions(),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          IRMapping mapping;
          mapping.map(binOpBlockArg0, blockArgs[0]);
          mapping.map(binOpBlockArg1, blockArgs[1]);
          Operation *newBinOp = nestedBuilder.clone(binOp, mapping);
          Value newResultValue = newBinOp->getResult(0);
          nestedBuilder.create<linalg::YieldOp>(newResultValue.getLoc(),
                                                ValueRange({newResultValue}));
        });
    rewriter.replaceAllUsesWith(shapeResult,
                                linalgReduceOp.getODSResults(0)[0]);
    rewriter.eraseOp(reduceWithIndexOp);
    return success();
  }
};

/// canonicalization pattern that replaces reduction if all reduction axes are
/// unit dim
///
/// %1,%2 = hfusion.reduce_with_index <max_with_index>
/// ins(%a:tensor<23x1x20xf32>,%b:tensor<23x1x20xi32>)
/// outs(%c:tensor<23x20xf32>, %d:tensor<23x20xi32>) dimension[1]
///
/// becomes
///
/// %a_collapsed = tensor.collapse_shape %a : tensor<23x1x20xf32> ->
/// tensor<23x20xf32>
/// %cst_0 = arith.constant 0 : i32
/// %d_filled = tensor.empty(): tensor<23x20xi32>
/// %filled = linalg.fill %cst_0, %d_filled : (i32, tensor<23x20xi32>) ->
/// tensor<23x20xi32>

struct ReduceWithIndexUnitDimCanonicalization
    : public OpRewritePattern<ReduceWithIndexOp> {
  using OpRewritePattern<ReduceWithIndexOp>::OpRewritePattern;

  SmallVector<ReassociationIndices>
  buildReassociationForCollapse(ArrayRef<int64_t> shape,
                                ArrayRef<bool> dropMask) const {
    auto rank = shape.size();
    SmallVector<int, 8> kept;
    for (size_t i = 0; i < rank; ++i)
      if (!dropMask[i])
        kept.push_back(i);

    SmallVector<ReassociationIndices> reassoc;
    if (kept.empty()) {
      // Collapse everything into a scalar => one group with all indices.
      ReassociationIndices g;
      for (size_t i = 0; i < rank; ++i)
        g.push_back(i);
      reassoc.push_back(g);
      return reassoc;
    }

    // Attach dropped dims to the following kept dim (or prior if leading).
    for (size_t ki = 0; ki < kept.size(); ++ki) {
      int prev = (ki == 0 ? -1 : kept[ki - 1]);
      int start = prev + 1;
      int end = kept[ki];
      ReassociationIndices group;
      for (int d = start; d <= end; ++d)
        group.push_back(d);
      reassoc.push_back(group);
    }
    // attach trailing dims after last kept to last group
    size_t lastKept = static_cast<size_t>(kept.back());
    for (size_t d = lastKept + 1; d < rank; ++d)
      reassoc.back().push_back(d);

    return reassoc;
  }

  LogicalResult matchAndRewrite(ReduceWithIndexOp reduceWithIndexOp,
                                PatternRewriter &rewriter) const override {
    auto mod = reduceWithIndexOp->getParentOfType<ModuleOp>();
    if (!hacc::utils::isRegBasedArch(mod))
      return rewriter.notifyMatchFailure(reduceWithIndexOp,
                                         "Pattern works on only RegBased arch");
    if (!(reduceWithIndexOp.hasPureTensorSemantics()))
      return failure();
    SmallVector<Value> inputs = reduceWithIndexOp.getInputs();
    assert(inputs.size() == 2);
    Value inValue = inputs[0];
    Value inIdx = inputs[1];
    auto loc = reduceWithIndexOp->getLoc();
    auto dataInTensorType = dyn_cast<RankedTensorType>(inValue.getType());
    auto indexInTensorType = dyn_cast<RankedTensorType>(inIdx.getType());
    auto reduceDims = reduceWithIndexOp.getDimensions();
    // bail out if there exists dynamic axes for now
    if (dataInTensorType.getNumDynamicDims() > 0) {
      return failure();
    }
    int64_t rank = dataInTensorType.getRank();
    SmallVector<bool, 8> dimsToDrop(rank, false);
    for (auto dim : reduceDims) {
      int64_t dimSize = dataInTensorType.getDimSize(dim);
      if (dimSize == 1) {
        dimsToDrop[dim] = true;
      } else {
        // bail if any reduction dim not 1
        return failure();
      }
    }
    // construct new shape for in_data and in_index
    SmallVector<int64_t, 8> newShape;
    for (int i = 0; i < rank; ++i) {
      if (!dimsToDrop[i])
        newShape.push_back(dataInTensorType.getDimSize(i));
    }
    auto dataInElemType = dataInTensorType.getElementType();
    auto collapsedType = RankedTensorType::get(newShape, dataInElemType);
    SmallVector<ReassociationIndices> reassoc =
        buildReassociationForCollapse(dataInTensorType.getShape(), dimsToDrop);
    Value collapsedInValue = rewriter.create<tensor::CollapseShapeOp>(
        loc, collapsedType, inValue, reassoc);
    auto cstZeroI32 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value emptyIdx = rewriter.create<tensor::EmptyOp>(
        loc, newShape, indexInTensorType.getElementType());
    Value fillVal = rewriter
                        .create<linalg::FillOp>(loc, ValueRange{cstZeroI32},
                                                ValueRange{emptyIdx})
                        ->getResult(0);
    rewriter.replaceOp(reduceWithIndexOp, {collapsedInValue, fillVal});
    return success();
  }
};

struct ExtractReduceWithExpandedUnitDims
    : public OpRewritePattern<ReduceWithIndexOp> {
  using OpRewritePattern<ReduceWithIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceWithIndexOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> reducedDims(op.getDimensions());
    if (reducedDims.size() == 1)
      return failure();
    auto inputType = cast<ShapedType>(op.getInputs()[0].getType());
    auto oldResultType = cast<ShapedType>(op.getInits()[0].getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> oldResultShape = oldResultType.getShape();
    if (failed(validateReductionConstraints(reducedDims, inputShape)))
      return failure();

    // Compute new reduced dimensions
    int64_t newReducedDim = computeCollapsedReducedDim(reducedDims, inputShape);

    // Compute new result shape (input shape without reduced dims)
    SmallVector<int64_t> newResultShape = llvm::to_vector(inputShape);
    newResultShape.erase(std::next(newResultShape.begin(), newReducedDim));

    // Compute reassociations for expand/collapse
    SmallVector<ReassociationIndices> expandReassociation;
    SmallVector<ReassociationIndices> collapseReassociation;
    if (failed(computeReassociations(llvm::to_vector(oldResultShape),
                                     newResultShape, expandReassociation,
                                     collapseReassociation)))
      return failure();

    // Apply the transformation
    OpBuilder builder(op->getContext());
    op.setDimensions({newReducedDim});

    expandInits(op, builder, newResultShape, expandReassociation);
    // Update result types to match expanded init types
    for (auto [i, result] : llvm::enumerate(op->getResults()))
      result.setType(op.getInits()[i].getType());

    collapseResults(op, builder, oldResultShape, collapseReassociation);

    return success();
  }

private:
  /// Validates that reduced dimensions meet the constraints:
  /// - At most one non-unit dimension
  /// - Dimensions are contiguous
  static LogicalResult
  validateReductionConstraints(ArrayRef<int64_t> reducedDims,
                               ArrayRef<int64_t> inputShape) {
    int64_t nonUnitCount = llvm::count_if(
        reducedDims, [&](int64_t dim) { return inputShape[dim] != 1; });
    if (nonUnitCount > 1)
      return failure();
    for (size_t i = 1; i < reducedDims.size(); ++i) {
      if (reducedDims[i] - reducedDims[i - 1] != 1)
        return failure();
    }
    return success();
  }

  /// Computes the collapsed reduced dimensions.
  /// Keeps the first dimension, but updates it to the non-unit dimension if
  /// exists.
  static int64_t computeCollapsedReducedDim(ArrayRef<int64_t> reducedDims,
                                            ArrayRef<int64_t> inputShape) {
    // Find the non-unit dimension if it exists
    auto collapsedDim = llvm::find_if(
        reducedDims, [&inputShape](auto dim) { return inputShape[dim] != 1; });
    if (collapsedDim == reducedDims.end())
      collapsedDim = reducedDims.begin();
    return *collapsedDim;
  }

  /// Computes expand and collapse reassociations between old and new result
  /// shapes.
  static LogicalResult computeReassociations(
      SmallVector<int64_t> oldResultShape, SmallVector<int64_t> newResultShape,
      SmallVector<ReassociationIndices> &expandReassociation,
      SmallVector<ReassociationIndices> &collapseReassociation) {

    SmallVector<ReassociationIndices> placeholder;
    SmallVector<int64_t> placeholderShape;

    // Compute expand reassociation: oldResultShape -> newResultShape
    if (!areLooseReassociationsCompatible(expandReassociation, placeholder,
                                          oldResultShape, newResultShape,
                                          placeholderShape))
      return failure();

    utils::renumberReassociation(expandReassociation);

    // Compute collapse reassociation: newResultShape -> oldResultShape
    if (!areLooseReassociationsCompatible(placeholder, collapseReassociation,
                                          newResultShape, oldResultShape,
                                          placeholderShape))
      return failure();

    utils::renumberReassociation(collapseReassociation);

    return success();
  }

  /// Expands init operands to the new result shape.
  static void expandInits(ReduceWithIndexOp op, OpBuilder &builder,
                          ArrayRef<int64_t> newResultShape,
                          ArrayRef<ReassociationIndices> reassociation) {

    builder.setInsertionPoint(op);
    for (auto &init : op.getInitsMutable()) {
      auto tensorType = cast<RankedTensorType>(init.get().getType());
      auto elementType = tensorType.getElementType();
      auto expandOp = builder.create<tensor::ExpandShapeOp>(
          op->getLoc(), RankedTensorType::get(newResultShape, elementType),
          init.get(), reassociation);
      init.assign(expandOp.getResult());
    }
  }

  /// Collapses results back to the original result shape.
  void collapseResults(ReduceWithIndexOp op, OpBuilder &builder,
                       ArrayRef<int64_t> oldResultShape,
                       ArrayRef<ReassociationIndices> reassociation) const {

    builder.setInsertionPointAfter(op);

    for (auto [init, result] :
         llvm::zip_equal(op.getInits(), op->getResults())) {
      auto elementType = cast<ShapedType>(init.getType()).getElementType();

      auto collapseOp = builder.create<tensor::CollapseShapeOp>(
          op->getLoc(), RankedTensorType::get(oldResultShape, elementType),
          result, reassociation);

      result.replaceAllUsesExcept(collapseOp.getResult(), collapseOp);
    }
  }
};

void ReduceWithIndexOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<ReplaceWithLinalgReduce, ReduceWithIndexUnitDimCanonicalization,
              ExtractReduceWithExpandedUnitDims>(context);
}

LogicalResult ReduceWithIndexOp::verify() {
  if (getDimensions().size() != 1)
    return emitOpError(
        "currently ReduceWithIndexOp only supports one reduction dimension");

  auto inputType = llvm::cast<ShapedType>(getInputs()[0].getType());
  auto initType = llvm::cast<ShapedType>(getInits()[0].getType());

  DenseSet<int64_t> dimensionsToReduce;
  ArrayRef<int64_t> dimensionsRef = getDimensions();
  for (int64_t dimension : dimensionsRef) {
    if (dimension < 0 || dimension >= inputType.getRank()) {
      return emitOpError()
             << "dimensions for reduction should be in the range [0, "
             << (inputType.getRank() - 1) << "].";
    }
    dimensionsToReduce.insert(dimension);
  }

  auto inputDims = inputType.getShape();
  auto initDims = initType.getShape();

  // Input dimensions that will be left after the reduction.
  SmallVector<int64_t> reducedInputDims;
  for (const auto &en : llvm::enumerate(inputDims)) {
    if (dimensionsToReduce.count(en.index()) == 0)
      reducedInputDims.push_back(en.value());
  }

  if (reducedInputDims.size() != static_cast<size_t>(initType.getRank())) {
    return emitOpError() << "number of dimensions after reduction "
                         << reducedInputDims.size()
                         << " doesn't match the init rank "
                         << initType.getRank();
  }

  if (reducedInputDims != initDims)
    return emitOpError() << "init dimensions [" << initDims
                         << "] doesn't match input dimensions after reduction ["
                         << reducedInputDims << "]";
  return success();
}

} // namespace hfusion
} // namespace mlir

static LogicalResult appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = llvm::dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    if (failed(appendMangledType(ss, memref.getElementType())))
      return failure();
    if (auto as = memref.getMemorySpace()) {
      if (auto attr = llvm::dyn_cast<IntegerAttr>(as))
        ss << "as" << attr.getInt();
      else
        return failure();
    }
    return success();
  }
  if (auto vec = llvm::dyn_cast<VectorType>(t)) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    if (failed(appendMangledType(ss, vec.getElementType())))
      return failure();
    return success();
  }
  if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
    return success();
  }
  return failure();
}

std::string mlir::hfusion::generateLibraryCallName(Operation *op) {
  assert(isa<linalg::LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  std::string fun = "";
  for (NamedAttribute kv : op->getAttrs()) {
    if (UnaryFnAttr ufa = llvm::dyn_cast<UnaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(ufa.getValue()).str() + "_";
    } else if (BinaryFnAttr bfa = llvm::dyn_cast<BinaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(bfa.getValue()).str() + "_";
    }
  }
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_" << fun;
  for (Type t : op->getOperandTypes()) {
    if (failed(appendMangledType(ss, t)))
      return std::string();
    ss << "_";
  }
  std::string res = ss.str();
  res.pop_back();
  return res;
}

/// Pattern to fold cast into emtpy.
///
/// Before:
/// tensor.empty(shape1, dtype1) + hfusion.cast(dtype2)
///
/// After:
/// tensor.empty(shape1, dtype2)
///
/// Restrictions:
/// the output of cast op should be an empty op
struct FoldCastEmpty : public OpRewritePattern<hfusion::CastOp> {
  using OpRewritePattern<hfusion::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto defEmptyOp = castOp.getInputs()[0].getDefiningOp<tensor::EmptyOp>();
    if (!defEmptyOp)
      return failure();
    auto output = castOp.getOutputs()[0];
    if (!output.getDefiningOp<tensor::EmptyOp>())
      return failure();
    rewriter.replaceOp(castOp, output);
    return success();
  }
};

struct ConstantFolding : public OpRewritePattern<hfusion::CastOp> {
  using OpRewritePattern<hfusion::CastOp>::OpRewritePattern;
  template <typename T>
  inline T roundToOdd(T x) const {
    T rounded = std::round(x);
    if (std::fabs(x - std::floor(x)) == 0.5) {
      if (static_cast<int>(rounded) % 2 != 0) {
        if (x > 0) {
          rounded = std::floor(x);
        } else {
          rounded = std::ceil(x);
        }
      }
    }
    return rounded;
  }

  const llvm::fltSemantics &getFltSemantics(Type eltType) const {
    if (eltType.isF16())
      return llvm::APFloatBase::IEEEhalf();
    if (eltType.isF32())
      return llvm::APFloatBase::IEEEsingle();
    if (eltType.isF64()) {
      return llvm::APFloatBase::IEEEdouble();
    }
    return llvm::APFloatBase::Bogus();
  }

  APFloat::roundingMode getLLVMRoundingMode(RoundMode rMode) const {
    APFloat::roundingMode rm;
    switch (rMode) {
    case RoundMode::RINT:
      rm = APFloat::rmNearestTiesToEven;
      break;
    case RoundMode::ROUND:
      rm = APFloat::rmNearestTiesToAway;
      break;
    case RoundMode::FLOOR:
      rm = APFloat::rmTowardNegative;
      break;
    case RoundMode::CEIL:
      rm = APFloat::rmTowardPositive;
      break;
    case RoundMode::TRUNC:
      rm = APFloat::rmTowardZero;
      break;
    case RoundMode::ODD:
      // let Dynamic denote round to odd
      rm = llvm::RoundingMode::Dynamic;
      break;
    default:
      rm = llvm::RoundingMode::Invalid;
    }
    return rm;
  }

  LogicalResult castfpToInt(APSInt &ret, const APFloat &oldAPVal,
                            RoundMode rMode) const {
    APFloat::roundingMode rm = getLLVMRoundingMode(rMode);
    if (rm == llvm::RoundingMode::Dynamic) {
      ret =
          static_cast<int64_t>(roundToOdd<double>(oldAPVal.convertToDouble()));
      return success();
    }
    bool isExact;
    auto status = oldAPVal.convertToInteger(ret, rm, &isExact);
    if ((status != APFloat::opStatus::opOK) &&
        (status != APFloat::opStatus::opInexact)) {
      return failure();
    }
    return success();
  }

  void castIntToFp(APInt &oldAPVal, APFloat &ret, bool signless,
                   const llvm::fltSemantics &sem, RoundMode rMode) const {
    APFloat::roundingMode rm = getLLVMRoundingMode(rMode);
    if (rm == llvm::RoundingMode::Dynamic) {
      ret = APFloat(sem, oldAPVal);
      return;
    }
    ret.convertFromAPInt(oldAPVal, !signless, rm);
  }

  std::optional<DenseIntOrFPElementsAttr>
  intToIntAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
               DenseIntOrFPElementsAttr &denseAttr) const {
    auto origArray = denseAttr.getValues<IntegerAttr>();
    SmallVector<APInt> newArray;
    Type outputDataType = outputTensorType.getElementType();
    bool signless = outputDataType.isSignlessInteger();
    if (denseAttr.isSplat()) {
      const auto size = origArray.size();
      APInt oldAPVal = origArray[0].getValue();
      APInt newAPVal =
          signless
              ? oldAPVal.zextOrTrunc(outputDataType.getIntOrFloatBitWidth())
              : oldAPVal.sextOrTrunc(outputDataType.getIntOrFloatBitWidth());
      newArray = SmallVector<APInt>(size, newAPVal);
    } else {
      for (auto ele : origArray) {
        APInt oldAPVal = ele.getValue();
        APInt newAPVal =
            signless
                ? oldAPVal.zextOrTrunc(outputDataType.getIntOrFloatBitWidth())
                : oldAPVal.sextOrTrunc(outputDataType.getIntOrFloatBitWidth());
        newArray.push_back(newAPVal);
      }
    }
    return DenseIntElementsAttr::get(outputTensorType, newArray);
  }

  std::optional<DenseIntOrFPElementsAttr>
  intToFpAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
              DenseIntOrFPElementsAttr &denseAttr) const {
    auto origArray = denseAttr.getValues<IntegerAttr>();
    bool signless = denseAttr.getElementType().isSignlessInteger();
    SmallVector<APFloat> newArray;
    Type outputDataType = outputTensorType.getElementType();
    if (denseAttr.isSplat()) {
      const auto size = origArray.size();
      APInt oldAPVal = origArray[0].getValue();
      if (&getFltSemantics(outputDataType) == &llvm::APFloatBase::Bogus()) {
        return std::nullopt;
      }
      APFloat newAPVal(getFltSemantics(outputDataType));
      castIntToFp(oldAPVal, newAPVal, signless, getFltSemantics(outputDataType),
                  roundMode);
      newArray = SmallVector<APFloat>(size, newAPVal);
    } else {
      for (auto ele : origArray) {
        APInt oldAPVal = ele.getValue();
        if (&getFltSemantics(outputDataType) == &llvm::APFloatBase::Bogus()) {
          return std::nullopt;
        }
        APFloat newAPVal(getFltSemantics(outputDataType));
        castIntToFp(oldAPVal, newAPVal, signless,
                    getFltSemantics(outputDataType), roundMode);
        newArray.push_back(newAPVal);
      }
    }
    return DenseFPElementsAttr::get(outputTensorType, newArray);
  }

  std::optional<DenseIntOrFPElementsAttr>
  fpToIntAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
              DenseIntOrFPElementsAttr &denseAttr) const {
    auto origArray = denseAttr.getValues<FloatAttr>();
    SmallVector<APInt> newArray;
    if (denseAttr.isSplat()) {
      const auto size = origArray.size();
      APFloat oldAPVal = origArray[0].getValue();
      APSInt ret(outputTensorType.getElementTypeBitWidth(),
                 outputTensorType.isUnsignedInteger());
      if (failed(castfpToInt(ret, oldAPVal, roundMode))) {
        return std::nullopt;
      }
      newArray = SmallVector<APInt>(size, ret);
    } else {
      for (auto ele : origArray) {
        APFloat oldAPVal = ele.getValue();
        APSInt ret(outputTensorType.getElementTypeBitWidth(),
                   outputTensorType.isUnsignedInteger());
        if (failed(castfpToInt(ret, oldAPVal, roundMode))) {
          return std::nullopt;
        }
        newArray.push_back(ret);
      }
    }
    return DenseIntElementsAttr::get(outputTensorType, newArray);
  }

  LogicalResult getRoundToOddVal(Type &outputDataType, APFloat &aPVal) const {
    if (outputDataType.isF32()) {
      float f32Val = aPVal.convertToFloat();
      float f32ValAfterRounding = roundToOdd<float>(f32Val);
      aPVal = APFloat(f32ValAfterRounding);
    } else if (outputDataType.isF64()) {
      double f64Val = aPVal.convertToDouble();
      double f64ValAfterRounding = roundToOdd<double>(f64Val);
      aPVal = APFloat(f64ValAfterRounding);
    } else {
      return failure();
    }
    return success();
  }

  LogicalResult sameFloatTypeCast(const RoundMode &roundMode,
                                  const Type &outputDataType,
                                  APFloat &aPVal) const {
    // fp -> int -> fp if src and dst are the same type
    APSInt temp(64, 0);
    if (failed(castfpToInt(temp, aPVal, roundMode))) {
      return failure();
    }
    castIntToFp(temp, aPVal, true, getFltSemantics(outputDataType), roundMode);
    return success();
  }

  LogicalResult fpToFpSingle(APFloat &aPVal, Type inputDataType,
                             Type outputDataType, RoundMode roundMode) const {
    bool loseInfo;
    APFloat::roundingMode rMode = getLLVMRoundingMode(roundMode);
    if (rMode == llvm::RoundingMode::Dynamic) {
      return getRoundToOddVal(outputDataType, aPVal);
    }
    if (inputDataType == outputDataType) {
      return sameFloatTypeCast(roundMode, outputDataType, aPVal);
    }
    aPVal.convert(getFltSemantics(outputDataType), rMode, &loseInfo);
    return success();
  }

  std::optional<DenseIntOrFPElementsAttr>
  fpToFpAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
             DenseIntOrFPElementsAttr &denseAttr) const {
    auto origArray = denseAttr.getValues<FloatAttr>();

    Type outputDataType = outputTensorType.getElementType();
    SmallVector<APFloat> newArray;
    if (denseAttr.isSplat()) {
      const auto size = origArray.size();
      APFloat aPVal = origArray[0].getValue();
      if (failed(fpToFpSingle(aPVal, denseAttr.getElementType(), outputDataType,
                              roundMode))) {
        return std::nullopt;
      }
      newArray = SmallVector<APFloat>(size, aPVal);
    } else {
      for (auto ele : origArray) {
        APFloat aPVal = ele.getValue();
        if (failed(fpToFpSingle(aPVal, denseAttr.getElementType(),
                                outputDataType, roundMode))) {
          return std::nullopt;
        }
        newArray.push_back(aPVal);
      }
    }
    return DenseFPElementsAttr::get(outputTensorType, newArray);
  }

  std::optional<DenseIntOrFPElementsAttr>
  intToAnyAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
               DenseIntOrFPElementsAttr &denseAttr) const {
    if (isa<IntegerType>(outputTensorType.getElementType()))
      return intToIntAttr(outputTensorType, roundMode, denseAttr);
    if (isa<FloatType>(outputTensorType.getElementType()))
      return intToFpAttr(outputTensorType, roundMode, denseAttr);
    return std::nullopt;
  }

  std::optional<DenseIntOrFPElementsAttr>
  fpToAnyAttr(RankedTensorType &outputTensorType, RoundMode &roundMode,
              DenseIntOrFPElementsAttr &denseAttr) const {
    if (isa<IntegerType>(outputTensorType.getElementType()))
      return fpToIntAttr(outputTensorType, roundMode, denseAttr);
    if (isa<FloatType>(outputTensorType.getElementType()))
      return fpToFpAttr(outputTensorType, roundMode, denseAttr);
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(hfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto output = castOp.getOutputs()[0];
    auto tensorEmptyOp = output.getDefiningOp<tensor::EmptyOp>();
    if (!tensorEmptyOp)
      return failure();
    auto input = castOp.getInputs()[0];
    auto cstOp = input.getDefiningOp<arith::ConstantOp>();
    if (!cstOp || !isa<RankedTensorType>(cstOp.getType()))
      return failure();

    auto roundMode = castOp.getRoundMode();
    auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(cstOp.getValue());
    if (!denseAttr)
      return failure();
    RankedTensorType outTensorType =
        dyn_cast<RankedTensorType>(output.getType());
    Type denseElmType = denseAttr.getElementType();
    // BF16 is not handled by this pattern
    if (denseElmType.isBF16() || denseElmType.isBF16()) {
      return failure();
    }

    std::optional<DenseIntOrFPElementsAttr> newArrAttr;

    if (isa<IntegerType>(denseElmType))
      newArrAttr = intToAnyAttr(outTensorType, roundMode, denseAttr);
    else if (isa<FloatType>(denseElmType))
      newArrAttr = fpToAnyAttr(outTensorType, roundMode, denseAttr);
    else
      return failure();

    if (!newArrAttr.has_value())
      return failure();

    rewriter.replaceOp(
        castOp, rewriter.create<arith::ConstantOp>(
                    castOp.getLoc(), output.getType(), (TypedAttr)*newArrAttr));
    return success();
  }
};

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldCastEmpty, ConstantFolding>(context);
}

LogicalResult CastOp::verify() {
  auto roundMode = getRoundMode();
  if (roundMode == hfusion::RoundMode::TRUNCWITHOVERFLOW) {
    auto inputType = getElementTypeOrSelf(getInputs()[0].getType());
    auto outputType = getElementTypeOrSelf(getOutputs()[0].getType());
    // TODO: constraint src to be only float type after bug fix
    if ((llvm::isa<FloatType>(inputType) || inputType.isInteger()) &&
        outputType.isInteger()) {
      return success();
    }
    return emitOpError(
        "inputs of castOp in TRUNCWITHOVERFLOW rounding mode "
        "must be float or integer type and outputs must be integer type");
  }
  return success();
}

LogicalResult InterleaveOp::verify() {
  if (static_cast<int64_t>(getInput().size()) != getInterLeaveChannelNums()) {
    return emitOpError("num of interleave op input must equal channel num");
  }

  auto outputType = llvm::dyn_cast<ShapedType>(getOutput().getType());
  int64_t interleaveAxis = outputType.getRank() - 1;
  if (outputType.isDynamicDim(interleaveAxis)) {
    // not check interleave axis with dynamic size
    return success();
  }
  if (outputType.getDimSize(interleaveAxis) % getInterLeaveChannelNums() != 0)
    return emitOpError("last dimension size of output RankedTensorType must be "
                       "multiples of current channel num");
  return success();
}

LogicalResult InterleaveOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  ShapedType firstShape = cast<ShapedType>(getInput().front().getType());
  int64_t rank = firstShape.getRank();
  int64_t interleaveAxis = rank - 1;
  int64_t chanNum = this->getInterLeaveChannelNums();

  // output[i] = input[i],            if i != interleave_axis
  // output[i] = input[i] * chan_num, if i == interleave_axis
  SmallVector<OpFoldResult> outputShape =
      tensor::getMixedSizes(b, this->getLoc(), getInput().front());
  AffineExpr mulExpr = b.getAffineSymbolExpr(0) * b.getAffineSymbolExpr(1);
  auto reifySize = affine::makeComposedFoldedAffineApply(
      b, this->getLoc(), mulExpr,
      {outputShape.back(), b.getIndexAttr(chanNum)});
  outputShape[interleaveAxis] = reifySize;

  reifiedReturnShapes.push_back(outputShape);
  return success();
}

LogicalResult DeinterleaveOp::verify() {
  auto inputType = llvm::dyn_cast<ShapedType>(getInput().getType());
  int64_t deinterleaveAxis = inputType.getRank() - 1;
  if (inputType.isDynamicDim(deinterleaveAxis)) {
    // not check deinterleave axis with dynamic size
    return success();
  }
  if (inputType.getDimSize(deinterleaveAxis) % getDeInterLeaveChannelNum() != 0)
    return emitOpError("last dimension size of input RankedTensorType must be "
                       "multiples of 2");

  if (static_cast<int64_t>(getOutput().size()) >= getDeInterLeaveChannelNum())
    return emitOpError("num of deinterleave op output is either one or zero");

  return success();
}

int64_t DeinterleaveOp::getDeInterLeaveChannelIdx() {
  return static_cast<int64_t>(getChannelIndex());
}

LogicalResult DeinterleaveOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  ShapedType shapedType = cast<ShapedType>(getInput().getType());
  int64_t rank = shapedType.getRank();
  int64_t deinterleaveAxis = rank - 1;
  int64_t chanNum = this->getDeInterLeaveChannelNum();

  // output[i] = input[i],            if i != deinterleave_axis
  // output[i] = input[i] / chan_num, if i == deinterleave_axis
  SmallVector<OpFoldResult> outputShape =
      tensor::getMixedSizes(b, this->getLoc(), getInput());
  AffineExpr divExpr =
      b.getAffineSymbolExpr(0).floorDiv(b.getAffineSymbolExpr(1));
  auto reifySize = affine::makeComposedFoldedAffineApply(
      b, this->getLoc(), divExpr,
      {outputShape.back(), b.getIndexAttr(chanNum)});
  outputShape[deinterleaveAxis] = reifySize;

  reifiedReturnShapes.push_back(outputShape);
  return success();
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//
void ArangeOp::getOffsetFromValue(OpBuilder &builder, Location loc,
                                  Value &offset) {
  offset = offset == nullptr
               ? builder.createOrFold<arith::ConstantIndexOp>(loc, 0)
               : offset;
}

void ArangeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Value init) {
  SmallVector<Value, 8> strides;
  hfusion::ArangeOp::getStridesFromValue(odsBuilder, odsState.location, init,
                                         strides);
  hfusion::ArangeOp::build(odsBuilder, odsState, strides, init);
}

void ArangeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     ValueRange strides, Value init) {
  Value offset = Value();
  hfusion::ArangeOp::getOffsetFromValue(odsBuilder, odsState.location, offset);
  SmallVector<Value, 8> inputs{offset};
  inputs.append(strides.begin(), strides.end());
  odsState.addOperands(offset);
  odsState.addOperands(strides);
  odsState.addOperands(init);
  if (isa<TensorType>(init.getType()))
    odsState.addTypes(init.getType());
  odsState.addAttribute("operandSegmentSizes",
                        odsBuilder.getDenseI32ArrayAttr(
                            {1, static_cast<int32_t>(strides.size()), 1}));
  Region &region = *odsState.addRegion();
  fillStructuredOpRegion(odsBuilder, region, ValueRange(inputs), init.getType(),
                         odsState.attributes.getAttrs(),
                         ArangeOp::getRegionBuilder());
}

void ArangeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Value offset, ValueRange strides, Value init) {
  hfusion::ArangeOp::getOffsetFromValue(odsBuilder, odsState.location, offset);
  SmallVector<Value, 8> inputs{offset};
  inputs.append(strides.begin(), strides.end());
  odsState.addOperands(inputs);
  odsState.addOperands(init);
  if (isa<TensorType>(init.getType()))
    odsState.addTypes(init.getType());
  odsState.addAttribute("operandSegmentSizes",
                        odsBuilder.getDenseI32ArrayAttr(
                            {1, static_cast<int32_t>(strides.size()), 1}));
  Region &region = *odsState.addRegion();
  fillStructuredOpRegion(odsBuilder, region, ValueRange(inputs), init.getType(),
                         odsState.attributes.getAttrs(),
                         ArangeOp::getRegionBuilder());
}

void ArangeOp::print(OpAsmPrinter &printer) {
  if (getOffset())
    printer << " offset[" << getOffset() << ']';
  printer << " strides[" << getStrides() << ']';

  printCommonStructuredOpParts(printer, {}, getInit());

  if (getResultTensor())
    printer << " -> " << getResultTensor().getType();
}

ParseResult ArangeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Storage for operandSegmentSizes attribute, include the 1 for the must-have
  // init operand
  SmallVector<int32_t, 8> operandSizes;
  auto indexTy = IndexType::get(parser.getContext());
  bool hasOffset = false;
  if (succeeded(parser.parseOptionalKeyword("offset"))) {
    OpAsmParser::UnresolvedOperand offset;
    if (parser.parseLSquare() || parser.parseOperand(offset) ||
        parser.parseRSquare())
      return parser.emitError(parser.getNameLoc(), "Expecting offset");

    if (parser.resolveOperand(offset, indexTy, result.operands))
      return parser.emitError(parser.getNameLoc(),
                              "Expecting offset of index type");
    operandSizes.push_back(1);
    hasOffset = true;
  } else
    operandSizes.push_back(0);

  // There should be as many strides as dimensions of the init operand.
  SmallVector<OpAsmParser::UnresolvedOperand> strides;
  if (parser.parseKeyword("strides") ||
      parser.parseOperandList(strides, OpAsmParser::Delimiter::Square))
    return failure();

  if (parser.resolveOperands(strides, indexTy, result.operands))
    return parser.emitError(parser.getNameLoc(),
                            "Expecting strides to be of index type");

  SmallVector<Type, 1> inputTys;
  SmallVector<Type, 1> outputTys;
  if (parseCommonStructuredOpParts(parser, result, inputTys, outputTys, false))
    return failure();

  // Number of strides should equal to rank
  auto shapeTy = cast<ShapedType>(outputTys.back());
  int rank = shapeTy.getRank();

  operandSizes.push_back(rank);

  // Parse optional result type for tensors only
  if (parser.parseOptionalArrowTypeList(result.types)) {
    if (!isa<MemRefType>(shapeTy))
      return parser.emitError(parser.getCurrentLocation(),
                              "expecting tensor output for tensor init value");
  }

  // Insert operandSegmentSize attribute, push back another one for the init
  // operand
  operandSizes.push_back(1);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(operandSizes));

  // Generate the implicit block
  auto unknownLoc = UnknownLoc::get(parser.getContext());
  Block &block = result.addRegion()->emplaceBlock();
  // Create the block arguments
  SmallVector<Type, 8> argTypes(rank, indexTy);
  if (hasOffset)
    argTypes.push_back(indexTy);
  argTypes.push_back(shapeTy.getElementType());
  block.addArguments(argTypes,
                     SmallVector<Location>(argTypes.size(), unknownLoc));
  ImplicitLocOpBuilder builder(unknownLoc, parser.getContext());
  builder.setInsertionPointToStart(&block);
  // Build the region
  getRegionBuilder()(builder, block, result.attributes.getAttrs());

  return success();
}

MutableOperandRange ArangeOp::getDpsInitsMutable() { return getInitMutable(); }

SmallVector<utils::IteratorType> ArangeOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr ArangeOp::getIndexingMaps() {
  SmallVector<AffineMap, 8> maps;
  auto builder = Builder(getContext());
  int rank = cast<ShapedType>(getInit().getType()).getRank();
  auto scalarMap = AffineMap::get(rank, 0, getContext());
  maps.append(rank, scalarMap);
  if (getOffset())
    maps.push_back(scalarMap);
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, getContext()));

  return builder.getAffineMapArrayAttr(maps);
}

std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)>
ArangeOp::getRegionBuilder() {
  return [](ImplicitLocOpBuilder &builder, Block &block,
            ArrayRef<NamedAttribute> attrs) {
    OpBuilder::InsertionGuard guard(builder);

    auto segmentSizes = cast_or_null<DenseI32ArrayAttr>(
        llvm::find_if(attrs, [](NamedAttribute attr) {
          return attr.getName() == "operandSegmentSizes";
        })->getValue());
    assert(segmentSizes && "Must have operandSegmentSizes attribute");
    // Check if offset exists
    Value offset;
    int argIdx = 0;
    int dim = 0;

    Type resultTy = block.getArguments().back().getType();
    if (segmentSizes[0])
      offset = block.getArgument(argIdx++);

    Value result = builder.create<arith::MulIOp>(
        block.getArgument(argIdx++), builder.create<linalg::IndexOp>(dim++));

    while (segmentSizes[1] > dim) {
      result = builder.create<arith::AddIOp>(
          result, builder.create<arith::MulIOp>(
                      block.getArgument(argIdx++),
                      builder.create<linalg::IndexOp>(dim++)));
    }

    if (offset)
      result = builder.create<arith::AddIOp>(result, offset);

    auto casted =
        builder
            .create<arith::IndexCastOp>(TypeRange{resultTy}, ValueRange{result})
            .getResult();
    builder.create<linalg::YieldOp>(casted);
  };
}

void ArangeOp::getStridesFromValue(OpBuilder &builder, Location loc, Value val,
                                   SmallVectorImpl<Value> &strides) {
  auto shapedTy = cast<ShapedType>(val.getType());
  Value constOne = builder.createOrFold<arith::ConstantIndexOp>(loc, 1);
  int rank = shapedTy.getRank();
  // Number of strides equal to number of ranks, fill with one's
  strides.append(rank, constOne);
  // Reverse iterater to fill rank from back to forward
  for (int dim = rank - 1; dim > 0; --dim) {
    Value size;
    if (isa<MemRefType>(shapedTy))
      size = builder.createOrFold<memref::DimOp>(loc, val, dim);
    else if (isa<TensorType>(shapedTy))
      size = builder.createOrFold<tensor::DimOp>(loc, val, dim);
    else
      llvm_unreachable(
          "Expected arange to be initialized with tensor or memref type.");
    strides[dim - 1] =
        builder.createOrFold<arith::MulIOp>(loc, strides[dim], size);
  }
}

std::string ArangeOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ArangeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<linalg::LinalgOp>(getOperation()));
}

//===----------------------------------------------------------------------===//
// isFiniteOp
//===----------------------------------------------------------------------===//

/// isFiniteOp decompose:
/// eg.
/// isFiniteOp = !(isnanOp(x) || isinfOp(x))
FailureOr<SmallVector<Value>> IsFiniteOp::decomposeOperation(OpBuilder &b) {
  auto loc = getLoc();
  auto input = getInput();

  auto emptyInf = utils::createEmptyOp(b, loc, getResult());
  auto emptyNan = utils::createEmptyOp(b, loc, getResult());

  // create IsInfOp and IsNanOp
  auto isInf = b.create<hfusion::IsInfOp>(loc, emptyInf.getType(), input);
  auto isNan = b.create<hfusion::IsNanOp>(loc, emptyNan.getType(), input);
  auto isInfReuslt = isInf.getResult();
  auto isNanResult = isNan.getResult();

  auto emptyVorOp = utils::createEmptyOp(b, loc, getResult());
  auto vorOp =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          b, loc, hfusion::BinaryFn::vor, {isInfReuslt, isNanResult},
          ValueRange(emptyVorOp));

  auto emptyVnot = utils::createEmptyOp(b, loc, getResult());
  auto vnotOp = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp,
                                       hfusion::UnaryFn, hfusion::UnaryFnAttr>(
      b, loc, hfusion::UnaryFn::vnot, ValueRange{vorOp->getResults()},
      ValueRange{emptyVnot});

  return SmallVector<Value>{vnotOp->getResults()};
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

MutableOperandRange GatherOp::getDpsInitsMutable() { return getInitMutable(); }

SmallVector<utils::IteratorType> GatherOp::getIteratorTypesArray() {
  SmallVector<utils::IteratorType> result(getInit().getType().getRank() + 1,
                                          utils::IteratorType::parallel);
  // The gather dim for indicies and src each take a loop, since we want the src
  // loop (reduction dim) to be on the inside, we set the gatherDim+1 to reduce
  result[getAxis() + 1] = utils::IteratorType::gather;
  return result;
}

/// The source gather axis will be inside the index gather axis. For src
/// <ixjxk>, indices <ixlxk> and gather axis 1, we want the resulting loop nest
/// to look like this:
///
/// for i ...
///   for l ...    <- parallel axis in indices corresponding to gather axis
///     for j ...  <- gather axis (cannot be tiled)
///       for k ...
ArrayAttr GatherOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  int64_t numIters = getInit().getType().getRank() + 1;
  SmallVector<AffineExpr> dims(numIters);
  auto dimsArrayRef = MutableArrayRef(dims);
  bindDimsList(ctx, dimsArrayRef);

  // Create the src and indexing affine expressions according to the desired
  // loop order
  const auto gatherDim = getAxis();
  auto dimsBeforeGatherAxis = dimsArrayRef.take_front(gatherDim);
  // The gather dim for indicies and src each take a loop, thus the +2
  auto dimsAfterGatherAxis = dimsArrayRef.drop_front(2 + gatherDim);

  // We want src dim to be on the inside
  AffineExpr idxGatherDim = dims[gatherDim];
  AffineExpr srcGatherDim = dims[gatherDim + 1];

  // The dims up until the gather axis are the same for src and idx
  auto srcDims = llvm::concat<AffineExpr>(
      dimsBeforeGatherAxis, MutableArrayRef{srcGatherDim}, dimsAfterGatherAxis);
  auto idxDims = llvm::concat<AffineExpr>(
      dimsBeforeGatherAxis, MutableArrayRef{idxGatherDim}, dimsAfterGatherAxis);

  // Dance to convert from concat_range to ArrayRef used by affine map
  auto srcDimVec = llvm::to_vector(srcDims);
  auto idxDimVec = llvm::to_vector(idxDims);

  auto srcMap = AffineMapAttr::get(AffineMap::get(numIters, 0, srcDimVec, ctx));
  auto indexMap =
      AffineMapAttr::get(AffineMap::get(numIters, 0, idxDimVec, ctx));
  // Init has the same indexing map as index
  return ArrayAttr::get(ctx, {srcMap, indexMap, indexMap});
}

std::string GatherOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, cast<linalg::LinalgOp>(getOperation()));
}

void GatherOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value src,
                     Value indices, Value init, int64_t gather_axis) {
  odsState.addAttribute(getAttributeNames()[0],
                        IntegerAttr::get(odsBuilder.getI64Type(), gather_axis));
  auto resultTy = dyn_cast<TensorType>(init.getType());
  buildStructuredOp(odsBuilder, odsState, resultTy, {src, indices}, init, {},
                    getRegionBuilder());
}

/// Creates the following body:
///   %iter = linalg.index <gatherAxis>
///   %cmp = arith.cmpi eq, <indexVal>, %iter
///   %sel = arith.select %cmp, <srcVal>, <outVal>
///   linalg.yield %sel
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)>
GatherOp::getRegionBuilder() {
  return [](ImplicitLocOpBuilder &builder, Block &block,
            ArrayRef<NamedAttribute> attrs) {
    assert(block.getNumArguments() == 3 &&
           "GatherOp expecting 3 block arguments");
    Value srcVal = block.getArgument(0);
    Value indexVal = block.getArgument(1);
    Value outVal = block.getArgument(2);
    StringRef kAxisName = GatherOp::getGatherAxisAttrName();
    const NamedAttribute *axisAttr =
        llvm::find_if(attrs, [kAxisName](NamedAttribute attr) {
          return attr.getName().strref() == kAxisName;
        });

    assert(axisAttr && "gather axis attribute must exist");
    assert(isa<IntegerAttr>(axisAttr->getValue()) &&
           "gather axis attribute must be an integer");
    int64_t gatherAxis = cast<IntegerAttr>(axisAttr->getValue()).getInt();
    // Since the src index is the inside the indices loop, need to increment 1
    // to get the value at the corresponding index
    Value iterIdx = builder.create<linalg::IndexOp>(gatherAxis + 1);
    if (iterIdx.getType() != indexVal.getType())
      iterIdx = builder.create<arith::IndexCastOp>(indexVal.getType(), iterIdx);

    Value isIndexVal = builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                     iterIdx, indexVal);
    Value yieldVal =
        builder.create<arith::SelectOp>(isIndexVal, srcVal, outVal);
    builder.create<linalg::YieldOp>(yieldVal);
  };
}

void GatherOp::print(OpAsmPrinter &p) {
  // attr-dict
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/getAttributeNames());
  printCommonStructuredOpParts(p, {getSrc(), getIndex()}, getInit());
  p << ' ';
  p.printKeywordOrString(getGatherAxisAttrName());
  p << " = " << getAxis();
  if (getNumResults())
    p.printArrowTypeList(getResultTypes());
}

ParseResult GatherOp::parse(OpAsmParser &p, OperationState &result) {
  // Parse attr-dict
  if (p.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type, 2> inputTypes;
  SmallVector<Type, 1> outputTypes;
  if (parseCommonStructuredOpParts(p, result, inputTypes, outputTypes,
                                   /*OperandSegmentSizes*/ false))
    return failure();

  StringRef kAxisAttrName = getAttributeNames()[0];
  int64_t axis;
  if (p.parseKeyword(kAxisAttrName) || p.parseEqual() || p.parseInteger(axis))
    return failure();

  result.addAttribute(
      kAxisAttrName,
      IntegerAttr::get(IntegerType::get(p.getContext(), 64), axis));

  // Parse optional result type
  if (p.parseOptionalArrowTypeList(result.types))
    return failure();

  // Build implicit region
  OpBuilder opBuilder(p.getContext());
  fillStructuredOpRegion(opBuilder, *(result.addRegion()), inputTypes,
                         outputTypes, result.attributes.getAttrs(),
                         getRegionBuilder());

  return success();
}

LogicalResult GatherOp::verify() {
  unsigned gatherAxis = getAxis();
  for (auto [dim, srcDim, initDim] : llvm::enumerate(
           getSrc().getType().getShape(), getInit().getType().getShape())) {
    if (dim == gatherAxis) {
      continue;
    }
    if (srcDim != initDim)
      return emitOpError("All dimensions must match except the gather axis");
  }

  // Result must be same type as init if present
  if (hasPureTensorSemantics()) {
    if (getNumResults() != 1)
      return emitOpError(
          "Expecting single result for gather op with tensor semantics");
    if (getResult().front().getType() != getInit().getType())
      return emitOpError(
          "Expecting gather op to have same result type as init type");
  }
  return success();
}

/// Since hardware vgather instruction can only support gathering on the last
/// dimension, we decompose gather ops that does not have the axis as the last
/// dimension, into loops performing gather with scalar. e.g.
///
/// hfusion.gather from <16x16x16> with index <16x2x16>
/// ==== Transform into ===>
/// scf.for i = 0 -> 16
///   scf.for j = 0 -> 2
///     scf.for k = 0 -> 16
///       idx = tensor.extract index[i, j, k]
///       extract = tensor.extract src[i, idx, k]
///       insert = tensor.insert extract into dest[i, j, k]
///
FailureOr<SmallVector<Value>> GatherOp::decomposeOperation(OpBuilder &b) {
  // According to numpy.take_along_axis (which triton.gather calls), the
  // dimensions that are not the gather axis are just broadcasts of the index
  OpBuilder::InsertionGuard guard(b);

  // Only match gathers with tensor semantics. Otherwise if the hardware can
  // support this instruction (gather axis = innermost dim and no cast needed),
  // then we do not match
  if (!this->hasPureTensorSemantics())
    return failure();

  b.setInsertionPoint(getOperation());

  Location loc = getLoc();
  Value src = getSrc();
  Value idx = getIndex();
  Value init = getInit();
  unsigned gatherAxis = getAxis();

  auto idxElmTy = getElementTypeOrSelf(idx);
  if (idxElmTy.isInteger(64)) {
    auto idx32 = castTo(b, idx, b.getI32Type());
    auto newGatherOp = b.create<GatherOp>(loc, src, idx32, init, gatherAxis);
    return SmallVector<Value>{newGatherOp.getResult()};
  }

  const auto rank = static_cast<unsigned>(getSrc().getType().getRank());
  auto srcElmTy = getElementTypeOrSelf(src);
  // Do not decompose if gather axis is last axis - can use gather instruction
  if (gatherAxis == rank - 1 && !srcElmTy.isInteger(64))
    return failure();

  Value cst0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value cst1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value idxGatherDimSize = b.create<tensor::DimOp>(loc, idx, gatherAxis);

  SmallVector<scf::ForOp> loopNest;

  // Create nested for loops encapsulating the dimensions that needs
  // decompose
  auto nestFor = [&](Value upperBound) -> Value {
    Value iterArg =
        loopNest.empty() ? init : loopNest.back().getRegionIterArg(0);
    auto forOp = b.create<scf::ForOp>(loc, cst0, upperBound, cst1, iterArg);
    if (!loopNest.empty())
      b.create<scf::YieldOp>(loc, forOp.getResult(0));
    loopNest.push_back(forOp);
    b.setInsertionPointToStart(forOp.getBody());
    return forOp.getInductionVar();
  };

  SmallVector<Value> idxOffset;
  for (unsigned i = 0; i < rank; ++i) {
    if (i == gatherAxis) {
      idxOffset.push_back(nestFor(idxGatherDimSize));
    } else {
      Value upperBound = b.create<tensor::DimOp>(loc, src, i);
      idxOffset.push_back(nestFor(upperBound));
    }
  }

  // Index needs to extract the single element
  Value idxElement = b.create<tensor::ExtractOp>(loc, idx, idxOffset);
  Type idxTy = b.getIndexType();
  // Cast to index type
  if (idxElement.getType() != idxTy)
    idxElement = b.create<arith::IndexCastOp>(loc, idxTy, idxElement);

  // Extract element from src according to the index in the gather dim
  SmallVector<Value> srcOffset(idxOffset);
  srcOffset[gatherAxis] = idxElement;
  Value srcElement = b.create<tensor::ExtractOp>(loc, src, srcOffset);

  // Insert element extracted from src into dst
  // the offset is the same as index
  Value target = loopNest.back().getRegionIterArg(0);
  Value result = b.create<tensor::InsertOp>(loc, srcElement, target, idxOffset);
  b.create<scf::YieldOp>(loc, result);

  return SmallVector<Value>{loopNest.front().getResult(0)};
}

/// canonicalization pattern that replaces gather if srcShapes[axis]==1
///
/// %a = hfusion.gather ins(%src, %index) outs(%init) axis
/// |
/// v
/// case1. if indexShape[axis] == 1:
///    %src replace %a
///
/// eg
/// %1 = hfusion.gather  ins(%arg0, %arg1 : tensor<5x6x1xf16>,
/// tensor<5x6x1xi32>) outs(%0 : tensor<5x6x1xf16>) axis = 2 ->
/// tensor<5x6x1xf16>
/// |
/// v
/// %arg0 replace %1
/// ---
/// case2. if indexShape[axis] != 1 && srcShape.size() != 1:
///    %newSrc = tensor.collapse_shape %src
///    %newA = linalg.broadcast ins(%newSrc) outs(%init) dimensions = [axis]
///    %newA replace %a
///
/// eg
/// %1 = hfusion.gather  ins(%arg0, %arg1 : tensor<5x6x1xf16>,
/// tensor<5x6x3xi32>) outs(%0 : tensor<5x6x3xf16>) axis = 2 ->
/// tensor<5x6x3xf16>
/// |
/// v
/// %newSrc = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<5x6x1xf16> into
/// tensor<5x6xf16> %arg0 = linalg.broadcast ins(%newSrc) outs(%init) dimensions
/// = [axis] %arg0 replace %1
struct GatherUnitDimCanonicalization : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    Value src = gatherOp.getDpsInputs()[0];
    Value index = gatherOp.getDpsInputs()[1];
    Value output = gatherOp.getDpsInits()[0];
    int64_t gatherAxis = (int64_t)gatherOp.getAxis();
    auto srcShape = utils::getShape(src.getType());
    auto indexShape = utils::getShape(index.getType());
    auto outShape = utils::getShape(output.getType());

    if (srcShape.size() == 1) {
      return failure();
    }
    if (srcShape[gatherAxis] != 1) {
      return failure();
    }
    // %src replace %a
    if (indexShape[gatherAxis] == 1) {
      rewriter.replaceOp(gatherOp, src);
      return success();
    }
    // convert to linalg.broadcast
    SmallVector<ReassociationIndices> reassociation;
    if (gatherAxis > 0) {
      for (int64_t i = 0; i < gatherAxis - 1; ++i)
        reassociation.push_back({i});
      reassociation.push_back({gatherAxis - 1, gatherAxis});
      for (int64_t i = gatherAxis + 1; i < (int)srcShape.size(); ++i)
        reassociation.push_back({i});
    } else {
      reassociation.push_back({0, 1});
      for (int64_t i = 2; i < (int)srcShape.size(); ++i)
        reassociation.push_back({i});
    }
    SmallVector<int64_t> newShape;
    for (int64_t i = 0; i < (int)srcShape.size(); ++i) {
      if (i == gatherAxis)
        continue;
      newShape.push_back(srcShape[i]);
    }
    RankedTensorType collapsedType =
        RankedTensorType::get(newShape, getElementTypeOrSelf(src));
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        src.getLoc(), collapsedType, src, reassociation);
    rewriter.setInsertionPointAfter(gatherOp);
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        gatherOp->getLoc(), collapseOp.getResult(), output, gatherAxis);
    rewriter.replaceAllUsesWith(gatherOp.getResults(),
                                broadcastOp->getResults());
    rewriter.eraseOp(gatherOp);
    return success();
  }
};
void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<GatherUnitDimCanonicalization>(context);
}
//===----------------------------------------------------------------------===//
// CumsumOp
//===----------------------------------------------------------------------===//

LogicalResult CumsumOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// CumprodOp
//===----------------------------------------------------------------------===//

LogicalResult CumprodOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// AtomicCasOp
//===----------------------------------------------------------------------===//

void AtomicCasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto [index, operand] : llvm::enumerate(this->getInput())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(index), /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
  OpOperand &operand = this->getDstMutable();
  if (!llvm::isa<MemRefType>(operand.get().getType()))
    return;
  effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AtomicXchgOp
//===----------------------------------------------------------------------===//

void AtomicXchgOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto [index, operand] : llvm::enumerate(this->getInput())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(index), /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
  OpOperand &operand = this->getDstMutable();
  if (!llvm::isa<MemRefType>(operand.get().getType()))
    return;
  effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage=*/0,
                       /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// EmbeddingGatherOp
//===----------------------------------------------------------------------===//

LogicalResult EmbeddingGatherOp::verify() { return success(); }

void EmbeddingGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// GatherLoadOp
//===----------------------------------------------------------------------===//

LogicalResult GatherLoadOp::verify() {
  auto inputType = getBase().getType();
  auto inputElemType = inputType.getElementType();
  auto indicesType = getIndices().getType();
  auto outputType = getResult().getType();
  auto outputElemType = outputType.getElementType();
  if (inputElemType != outputElemType) {
    return emitOpError("output of hfusion::GatherLoadOp must have the same "
                       "element type as base");
  }
  if (indicesType.getRank() != outputType.getRank()) {
    return emitOpError("indices of hfusion::GatherLoadOp must have the same "
                       "rank as output");
  }
  if (auto mask = getMask()) {
    if (mask.getType().getShape() != indicesType.getShape()) {
      return emitOpError("mask of hfusion::GatherLoadOp must have the same "
                         "shape and rank as indices");
    }
  }
  if (auto other = getOther()) {
    if (other.getType() != inputElemType) {
      return emitOpError("other of hfusion::GatherLoadOp must have the same "
                         "element type as base");
    }
  }
  return success();
}

void GatherLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ScatterStoreOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterStoreOp::verify() {
  auto inputType = getBase().getType();
  auto inputElemType = inputType.getElementType();
  auto indicesType = getIndices().getType();
  auto dataType = getData().getType();
  auto dataElemType = dataType.getElementType();
  if (inputElemType != dataElemType) {
    return emitOpError("data of hfusion::ScatterStoreOp must have the same "
                       "element type as base");
  }
  if (indicesType.getRank() != dataType.getRank()) {
    return emitOpError("indices of hfusion::ScatterStoreOp must have the same "
                       "rank as data");
  }
  if (auto mask = getMask()) {
    if (mask.getType().getShape() != dataType.getShape()) {
      return emitOpError("mask of hfusion::ScatterStoreOp must have the same "
                         "shape and rank as data");
    }
  }
  return success();
}

void ScatterStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IndirectLoadOp
//===----------------------------------------------------------------------===//

LogicalResult IndirectLoadOp::verify() {
  auto srcType = getSrc().getType();
  auto offsetsType = getOffsets().getType();
  auto offsetsTensorType = mlir::cast<TensorType>(offsetsType);
  if (!offsetsTensorType)
    return emitOpError("offsets must be a tensor type");
  auto dstType = getDst().getType();
  auto dstTensorType = mlir::cast<TensorType>(dstType);
  if (!dstTensorType)
    return emitOpError("dst must be a tensor type");
  auto srcMemrefType = mlir::cast<MemRefType>(srcType);
  if (!srcMemrefType)
    return emitOpError("src must be a memref type");

  auto srcElementType = srcMemrefType.getElementType();
  auto dstElementType = dstTensorType.getElementType();
  if (dstElementType != srcElementType) {
    return emitOpError("dst of hfusion::IndirectLoadOp must have the same "
                       "element type as src");
  }

  if (dstTensorType.getShape() != offsetsTensorType.getShape()) {
    return emitOpError(
        "dst of hfusion::IndirectLoadOp must have the same shape as offsets");
  }

  auto mask = getMask();
  auto maskType = mask.getType();
  auto maskTensorType = mlir::cast<TensorType>(maskType);
  if (!maskTensorType)
    return emitOpError("mask must be a tensor type");

  if (maskTensorType.getShape() != offsetsTensorType.getShape()) {
    return emitOpError("mask of hfusion::IndirectLoadOp must have the same "
                       "shape and rank as offsets");
  }

  auto other = getOther();
  auto otherType = other.getType();
  auto otherTensorType = mlir::cast<TensorType>(otherType);
  if (!otherTensorType)
    return emitOpError("other must be a tensor type");

  if (otherTensorType.getShape() != offsetsTensorType.getShape()) {
    return emitOpError("other of hfusion::IndirectLoadOp must have the same "
                       "shape and rank as offsets");
  }

  auto otherElementType = otherTensorType.getElementType();
  if (srcElementType != otherElementType) {
    return emitOpError("other of hfusion::IndirectLoadOp must have the same "
                       "element type as src");
  }

  return success();
}

void IndirectLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getMaskMutable()),
      SideEffects::DefaultResource::get();
  effects.emplace_back(MemoryEffects::Read::get(), &getOtherMutable()),
      SideEffects::DefaultResource::get();
}

//===----------------------------------------------------------------------===//
// IndirectStoreOp
//===----------------------------------------------------------------------===//

LogicalResult IndirectStoreOp::verify() {

  auto dstType = getDst().getType();
  auto dstMemrefType = mlir::cast<MemRefType>(dstType);
  if (!dstMemrefType)
    return emitOpError("dst must be a memref type");
  auto dstElementType = dstMemrefType.getElementType();
  auto srcType = getSrc().getType();
  auto srcTensorType = mlir::cast<TensorType>(srcType);
  if (!srcTensorType)
    return emitOpError("src must be a tensor type");
  auto srcElementType = srcTensorType.getElementType();
  if (dstElementType != srcElementType) {
    return emitOpError("src of hfusion::IndirectStoreOp must have the same "
                       "element type as dst");
  }

  auto offsetsType = getOffsets().getType();
  auto offsetsTensorType = mlir::cast<TensorType>(offsetsType);
  if (!offsetsTensorType)
    return emitOpError("offsets must be a tensor type");
  if(offsetsTensorType.getShape() != srcTensorType.getShape()) {
    return emitOpError("offsets of hfusion::IndirectStoreOp must have the same "
                         "shape and rank as src");
  }

  auto mask = getMask();
  if (mask) {
    auto maskType = mask.getType();
    auto maskTensorType = mlir::cast<TensorType>(maskType);
    if (!maskTensorType)
      return emitOpError("mask must be a tensor type");
    if (maskTensorType.getShape() != offsetsTensorType.getShape()) {
      return emitOpError("mask of hfusion::IndirectStoreOp must have the same "
                         "shape and rank as offsets");
    }
  }

  return success();
}

void IndirectStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetsMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());

  if (getMask()) {
    effects.emplace_back(
        MemoryEffects::Read::get(),
        &getOperation()->getOpOperand(getODSOperandIndexAndLength(3).first),
        SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// GatherTOp
//===----------------------------------------------------------------------===//

void GatherTOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IndexPutOp
//===----------------------------------------------------------------------===//

void IndexPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ScatterTOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterTOp::verify() {
  auto valueType = getValue().getType();
  auto valueTensorType = mlir::cast<TensorType>(valueType);
  if (!valueTensorType) {
    return emitOpError("value must be a tensor type");
  }
  auto indexTileType = getIndexTile().getType();
  auto indexTileTensorType = mlir::cast<TensorType>(indexTileType);
  if (!indexTileTensorType) {
    return emitOpError("index_tile must be a tensor type");
  }
  if (valueTensorType.getShape() != indexTileTensorType.getShape()) {
    return emitOpError("tensor of value and index_tile of hfusion::ScatterTOp "
                       "must have the same shape");
  }

  return success();
}

void ScatterTOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexTileMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// HFusionDialect
//===----------------------------------------------------------------------===//
void HFusionDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<
      mlir::linalg::InlineDenseSplatToGenericRegion<hfusion::ElemwiseBinaryOp>,
      mlir::linalg::InlineDenseSplatToGenericRegion<hfusion::ElemwiseUnaryOp>,
      mlir::linalg::InlineDenseSplatToGenericRegion<hfusion::CompareOp>,
      mlir::linalg::InlineDenseSplatToGenericRegion<hfusion::CastOp>,
      mlir::linalg::SimplifySplatDenseForBinary<hfusion::ElemwiseBinaryOp>,
      mlir::linalg::SimplifySplatDenseForBinary<hfusion::CompareOp>>(
      getContext());
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

int64_t SortOp::getSignedSortAxis() {
  return getSortAxisAttr().getValue().getSExtValue();
}

LogicalResult SortOp::verify() {
  int64_t sortAxis = getSignedSortAxis();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  if (sortAxis != srcVecType.getRank() - 1 && sortAxis != -1) {
    return emitOpError() << "Currently only tail axis sorting is supported";
  }
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// HistogramOp
//===----------------------------------------------------------------------===//
LogicalResult HistogramOp::verify() {
  auto inTy = mlir::dyn_cast<RankedTensorType>(getInput().getType());
  auto outTy = mlir::dyn_cast<RankedTensorType>(getOutput().getType());
  Value mask = getMask();
  // Input/output must be ranked tensors
  if (!inTy || !outTy)
    return emitOpError() << "requires ranked tensor types for input and output";

  // Input must be 1D
  if (inTy.getRank() != 1)
    return emitOpError() << "input must be rank-1";
  // Output must be 1D statically sized
  if (outTy.getRank() != 1)
    return emitOpError() << "output must be rank-1";
  if (!outTy.hasStaticShape())
    return emitOpError() << "output must have static shape";

  // Output length must match num_bins
  int64_t bins = static_cast<int64_t>(getNumBins());
  if (outTy.getDimSize(0) != bins)
    return emitOpError() << "output length (" << outTy.getDimSize(0)
                         << ") must equal num_bins (" << bins << ")";

  // If mask is provided, it must match input shape
  if (mask) {
    auto maskTy = mlir::dyn_cast<RankedTensorType>(mask.getType());
    if (!maskTy)
      return emitOpError() << "mask must be a ranked tensor";
    if (maskTy.getElementType() != IntegerType::get(getContext(), 1))
      return emitOpError() << "mask element type must be i1";
    if (maskTy.getShape() != inTy.getShape())
      return emitOpError() << "mask shape must match input shape";
  }

  return success();
}

namespace mlir::hfusion {
/// Helper function: Calculate element's bin index and validity (whether it
/// falls within [0, numBins)).
///
/// @param b OpBuilder for constructing MLIR operations.
/// @param loc Location information for error reporting and debugging.
/// @param input Input tensor containing elements to be histogrammed.
/// @param i Loop induction variable (current element index).
/// @param numBinsVal Total number of bins (as an IndexType value).
/// @param c0 Constant 0 (IndexType), used for range checks.
/// @return Pair of values:
///         - First: Element's bin index (converted from input element).
///         - Second: Boolean (i1) indicating if the element is in valid range
///         [0, numBins).
inline std::pair<Value, Value> getElementValidity(OpBuilder &b, Location loc,
                                                  Value input, Value i,
                                                  Value numBinsVal, Value c0) {
  // Extract element and convert to bin index
  Value elem = b.create<tensor::ExtractOp>(loc, input, ValueRange{i});
  Value elemSigned = b.create<arith::ExtUIOp>(loc, b.getI64Type(), elem);
  Value elemIdx =
      b.create<arith::IndexCastOp>(loc, b.getIndexType(), elemSigned);

  // Check if element is within valid range
  Value geZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, elemIdx, c0);
  Value ltNumBins = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                            elemIdx, numBinsVal);
  Value inRange = b.create<arith::AndIOp>(loc, geZero, ltNumBins);

  return {elemIdx, inRange};
}

/// Helper function: Generate histogram update logic with conditional execution.
///
/// @param b OpBuilder for constructing MLIR operations.
/// @param loc Location information for error reporting and debugging.
/// @param hist Current histogram tensor (to be updated).
/// @param elemIdx Bin index where the count should be incremented.
/// @param cond Boolean (i1) condition: only update histogram if true.
/// @param oneOut Constant 1 of the histogram's element type (used for
/// incrementing counts).
/// @return Updated histogram tensor (either modified or original, based on
/// `cond`).
inline Value histogramUpdate(OpBuilder &b, Location loc, Value hist,
                             Value elemIdx, Value cond, Value oneOut) {
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange{hist.getType()}, cond, true);

  // Then branch: Update histogram
  {
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    Value old =
        thenBuilder.create<tensor::ExtractOp>(loc, hist, ValueRange{elemIdx});
    Value neu = thenBuilder.create<arith::AddIOp>(loc, old, oneOut);
    Value upd = thenBuilder.create<tensor::InsertOp>(loc, neu, hist,
                                                     ValueRange{elemIdx});
    thenBuilder.create<scf::YieldOp>(loc, upd);
  }

  // Else branch: Return original histogram
  {
    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<scf::YieldOp>(loc, hist);
  }

  return ifOp.getResult(0);
}
} // namespace mlir::hfusion

/// Decompose the HistogramOp into a sequence of lower-level MLIR operations.
/// This method implements the histogram calculation by iterating over input
/// elements, checking their validity, and updating the histogram counts
/// conditionally.
///
/// @param b OpBuilder used to construct the decomposed operations.
/// @return Success: A vector containing the final histogram tensor.
///         Failure: If decomposition encounters an error.
FailureOr<SmallVector<Value>> HistogramOp::decomposeOperation(OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(getOperation());

  Location loc = getLoc();

  Value input = getInput();
  Value mask = getMask();
  auto inTy = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto outTy = mlir::dyn_cast<RankedTensorType>(getOutput().getType());

  Type outEltTy = outTy.getElementType();

  // Helpers
  auto cstIdx = [&](int64_t v) -> Value {
    return b.create<arith::ConstantIndexOp>(loc, v);
  };
  auto cstOut = [&](int64_t v) -> Value {
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outEltTy, v));
  };

  // Constants
  Value c0 = cstIdx(0);
  Value c1 = cstIdx(1);
  Value oneOut = cstOut(1);
  Value zeroOut = cstOut(0);
  auto numBins = getNumBins();
  Value numBinsVal = cstIdx(numBins);

  // Create zero-initialized histogram tensor
  Value histEmpty = b.create<tensor::EmptyOp>(loc, outTy.getShape(), outEltTy);
  Value histInit =
      b.create<linalg::FillOp>(loc, zeroOut, histEmpty).getResult(0);

  // Upper bound: number of elements in input
  Value ub = inTy.hasStaticShape() ? cstIdx(inTy.getDimSize(0))
                                   : b.create<tensor::DimOp>(loc, input, 0);

  // Single loop over input elements
  auto forOp = b.create<scf::ForOp>(loc, c0, ub, c1, ValueRange{histInit});
  {
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(forOp.getBody());

    Value i = forOp.getInductionVar();
    Value hist = forOp.getRegionIterArg(0);

    auto [elemIdx, inRange] =
        getElementValidity(b, loc, input, i, numBinsVal, c0);
    Value finalCond = inRange;
    if (mask) {
      Value maskCond = b.create<tensor::ExtractOp>(loc, mask, ValueRange{i});
      finalCond = b.create<arith::AndIOp>(loc, maskCond, inRange);
    }
    Value updatedHist =
        histogramUpdate(b, loc, hist, elemIdx, finalCond, oneOut);
    b.create<scf::YieldOp>(loc, updatedHist);
  }

  Value finalHist = forOp.getResult(0);
  return SmallVector<Value>{finalHist};
}

LogicalResult MatMulMxOp::verify() {
  auto inputATy = mlir::cast<ShapedType>(getInputA().getType());
  auto inputBTy = mlir::cast<ShapedType>(getInputB().getType());
  auto scaleATy = mlir::cast<ShapedType>(getScaleA().getType());
  auto scaleBTy = mlir::cast<ShapedType>(getScaleB().getType());
  auto resultTy = mlir::cast<ShapedType>(getResult().getType());

  // Input/Output must be ranked tensors
  if (!inputATy || !inputBTy || !scaleATy || !scaleBTy)
    return emitOpError() << "requires shaped types for input";

  static constexpr int twoD = 2;
  if (inputATy.getRank() != twoD || inputBTy.getRank() != twoD)
    return emitOpError() << "requires both input to have rank 2";

  static constexpr int dim0 = 0;
  static constexpr int dim1 = 1;
  if (inputATy.getDimSize(dim1) != inputBTy.getDimSize(dim0))
    return emitOpError()
           << "requires inner dimension of matmul matrix to match";

  if (resultTy.getDimSize(dim0) != inputATy.getDimSize(dim0) ||
      resultTy.getDimSize(dim1) != inputBTy.getDimSize(dim1))
    return emitOpError() << "requires output shape to match with input shapes";

  // if acc is provided
  if (getAcc()) {
    auto accTy = mlir::cast<ShapedType>(getAcc().getType());
    if (accTy.getRank() != resultTy.getRank())
      return emitOpError() << "acc and output should have the same shape";

    for (int dim = 0; dim < accTy.getRank(); dim++) {
      if (accTy.getDimSize(dim) != resultTy.getDimSize(dim))
        return emitOpError() << "acc and output should have the same shape";
    }
  }

  return success();
}

void markAsDotScaleKernel(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  funcOp->setAttr("IsDotScaleKernel", UnitAttr::get(context));
}

/// Input:
/// %6 = hfusion.matmul_mx ins(%a, %b, %scaleA, %scaleB : 
///   tensor<64x64xf8E5M2>, tensor<64x64xf8E5M2>, tensor<64x2xi8>, tensor<64x2xi8>) outs(%1 : tensor<64x64xf32>
/// ) -> tensor<64x64xf32>
///
/// Output:
/// %c7_i16 = arith.constant 7 : i16
/// %6 = tensor.empty() : tensor<64x2xi16>
/// %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} 
///   ins(%scaleA : tensor<64x2xi8>) outs(%6 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %8 = tensor.empty() : tensor<64x2xi16>
/// %9 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%scaleB: tensor<64x2xi8>) 
///   outs(%8 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %10 = linalg.fill ins(%c7_i16 : i16) outs(%6 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %11 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%7, %10 : tensor<64x2xi16>, tensor<64x2xi16>) 
/// outs(%6 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %12 = linalg.fill ins(%c7_i16 : i16) outs(%8 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %13 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%9, %12 : tensor<64x2xi16>, tensor<64x2xi16>) 
///   outs(%8 : tensor<64x2xi16>) -> tensor<64x2xi16>
/// %14 = tensor.empty() : tensor<64x2xbf16>
/// %15 = hfusion.bitcast ins(%11 : tensor<64x2xi16>) outs(%14 : tensor<64x2xbf16>) -> tensor<64x2xbf16>
/// %16 = tensor.empty() : tensor<64x2xbf16>
/// %17 = hfusion.bitcast ins(%13 : tensor<64x2xi16>) outs(%16 : tensor<64x2xbf16>) -> tensor<64x2xbf16>
/// %18 = tensor.empty() : tensor<64x2xf32>
/// %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%15 : tensor<64x2xbf16>) 
///   outs(%18 : tensor<64x2xf32>) -> tensor<64x2xf32>
/// %20 = tensor.empty() : tensor<64x2xf32>
/// %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%17 : tensor<64x2xbf16>) 
///   outs(%20 : tensor<64x2xf32>) -> tensor<64x2xf32>
/// %22 = tensor.empty() : tensor<64x64xf16>
/// %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<64x64xf8E5M2>) 
///   outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
/// %24 = tensor.empty() : tensor<64x64xf16>
/// %25 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<64x64xf8E5M2>) 
///   outs(%24 : tensor<64x64xf16>) -> tensor<64x64xf16>
/// %26 = tensor.empty() : tensor<64x2xf16>
/// %27 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%19 : tensor<64x2xf32>) 
///   outs(%26 : tensor<64x2xf16>) -> tensor<64x2xf16>
/// %28 = tensor.empty() : tensor<64x2xf16>
/// %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%21 : tensor<64x2xf32>) 
///   outs(%28 : tensor<64x2xf16>) -> tensor<64x2xf16>
/// %30 = tensor.empty() : tensor<64x2x32xf16>
/// %broadcasted = linalg.broadcast ins(%27 : tensor<64x2xf16>) outs(%30 : tensor<64x2x32xf16>) dimensions = [2] 
/// %collapsed = tensor.collapse_shape %broadcasted [[0], [1, 2]] : tensor<64x2x32xf16> into tensor<64x64xf16>
/// %31 = tensor.empty() : tensor<64x2x32xf16>
/// %broadcasted_6 = linalg.broadcast ins(%29 : tensor<64x2xf16>) outs(%31 : tensor<64x2x32xf16>) dimensions = [2] 
/// %collapsed_7 = tensor.collapse_shape %broadcasted_6 [[0], [1, 2]] : tensor<64x2x32xf16> into tensor<64x64xf16>
/// %32 = tensor.empty() : tensor<64x64xf16>
/// %transposed = linalg.transpose ins(%collapsed_7 : tensor<64x64xf16>) outs(%32 : tensor<64x64xf16>) 
///   permutation = [1, 0] 
/// %33 = tensor.empty() : tensor<64x64xf16>
/// %34 = tensor.empty() : tensor<64x64xf16>
/// %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
///   ins(%a, %collapsed : tensor<64x64xf16>, tensor<64x64xf16>) outs(%33 : tensor<64x64xf16>) -> tensor<64x64xf16>
/// %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
///   ins(%b, %transposed : tensor<64x64xf16>, tensor<64x64xf16>) outs(%34 : tensor<64x64xf16>) -> tensor<64x64xf16>
/// %37 = linalg.matmul ins(%35, %36 : tensor<64x64xf16>, tensor<64x64xf16>) 
///   outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
FailureOr<SmallVector<Value>> MatMulMxOp::decomposeOperation(OpBuilder &builder) {
  // Mark func as dot scale
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  markAsDotScaleKernel(funcOp);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(getOperation());
  Location location = getLoc();

  Value a = getInputA();
  Value b = getInputB();
  Value c = getAcc();
  Value scaleA = getScaleA();
  Value scaleB = getScaleB();

  auto scaleAType = cast<RankedTensorType>(scaleA.getType());
  auto scaleBType = cast<RankedTensorType>(scaleB.getType());
  auto shapeScaleA = scaleAType.getShape();
  auto shapeScaleB = scaleBType.getShape();

  // Cast scale to i16
  auto modeAttr = builder.getNamedAttr(hfusion::RoundModeAttr::getMnemonic(), 
                                       builder.getAttr<hfusion::RoundModeAttr>(hfusion::RoundMode::RINT));
  const Type i16Ty = builder.getI16Type();
  Value scaleAI16 = castTo(builder, scaleA, i16Ty);
  Value scaleBI16 = castTo(builder, scaleB, i16Ty);

  // scale <<= 7
  auto shlFnAttr = builder.getNamedAttr("fun", builder.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::shli));
  Value const7 = builder.create<arith::ConstantIntOp>(location, builder.getI16Type(), 7);

  Value emptyScaleAI16 = builder.create<tensor::EmptyOp>(
      location, RankedTensorType::get(shapeScaleA, i16Ty), ValueRange{});
  Value sevenA =
      builder.create<linalg::FillOp>(location, const7, emptyScaleAI16).getResult(0);
  Value scaleAShl = builder
                        .create<hfusion::ElemwiseBinaryOp>(
                            location, ValueRange{scaleAI16, sevenA},
                            ValueRange{emptyScaleAI16}, shlFnAttr)
                        ->getResult(0);
 
  Value emptyScaleBI16 = builder.create<tensor::EmptyOp>(
      location, RankedTensorType::get(shapeScaleB, i16Ty), ValueRange{});
  Value sevenB =
      builder.create<linalg::FillOp>(location, const7, emptyScaleBI16).getResult(0);
  Value scaleBShl = builder
                        .create<hfusion::ElemwiseBinaryOp>(
                            location, ValueRange{scaleBI16, sevenB},
                            ValueRange{emptyScaleBI16}, shlFnAttr)
                        ->getResult(0);

  // Bitcast to bf16
  const Type bf16Ty = builder.getBF16Type();
  Value emptyScaleABF16 = builder.create<tensor::EmptyOp>(
      location, RankedTensorType::get(shapeScaleA, bf16Ty), ValueRange{});
  Value scaleABF16 = builder.create<hfusion::BitcastOp>(
      location, TypeRange{emptyScaleABF16.getType()}, ValueRange{scaleAShl}, ValueRange{emptyScaleABF16}).getResult(0);
  Value emptyScaleBBF16 = builder.create<tensor::EmptyOp>(
      location, RankedTensorType::get(shapeScaleB, bf16Ty), ValueRange{});
  Value scaleBBF16 = builder.create<hfusion::BitcastOp>(
      location, TypeRange{emptyScaleBBF16.getType()}, ValueRange{scaleBShl}, ValueRange{emptyScaleBBF16}).getResult(0);

  // Cast scale to f32
  const Type f32Ty = builder.getF32Type();
  Value scaleAF32 = castTo(builder, scaleABF16, f32Ty);
  Value scaleBF32 = castTo(builder, scaleBBF16, f32Ty);

  // If input is fp8, cast to fp16
  auto aType = cast<RankedTensorType>(a.getType());
  auto bType = cast<RankedTensorType>(b.getType());
  const Type f16Ty = builder.getF16Type();
  if (isFP8(aType.getElementType(), builder)) {
    Value emptyAF16 = builder.create<tensor::EmptyOp>(
        location, RankedTensorType::get(aType.getShape(), f16Ty), ValueRange{});
    a = builder.create<hfusion::CastOp>(
        location, ValueRange{a}, ValueRange{emptyAF16}, modeAttr).getResult(0);
    aType = RankedTensorType::get(aType.getShape(), f16Ty);
  }
  if (isFP8(bType.getElementType(), builder)) {
    Value emptyBF16 = builder.create<tensor::EmptyOp>(
        location, RankedTensorType::get(bType.getShape(), f16Ty), ValueRange{});
    b = builder.create<hfusion::CastOp>(
        location, ValueRange{b}, ValueRange{emptyBF16}, modeAttr).getResult(0);
    bType = RankedTensorType::get(bType.getShape(), f16Ty);
  }

  // Cast scale to input type
  Value scaleAFinal = castTo(builder, scaleAF32, aType.getElementType());
  Value scaleBFinal = castTo(builder, scaleBF32, bType.getElementType());

  // Broadcast scale
  static constexpr int TILE_SIZE = 32;
  Value emptyScaleABroadcasted = builder.create<tensor::EmptyOp>(
      location, 
      RankedTensorType::get({shapeScaleA[0], shapeScaleA[1], TILE_SIZE}, aType.getElementType()), 
      ValueRange{});
  Value scaleABroadcasted = builder.create<linalg::BroadcastOp>(
      location, scaleAFinal, emptyScaleABroadcasted, ArrayRef<int64_t>{2}).getResult()[0];
  SmallVector<ReassociationIndices> reassocIdxScaleA{{0}, {1, 2}};
  Value scaleACollapsed = builder.create<tensor::CollapseShapeOp>(
      location, scaleABroadcasted, reassocIdxScaleA);
  Value emptyScaleBBroadcasted = builder.create<tensor::EmptyOp>(
      location, 
      RankedTensorType::get({shapeScaleB[0], shapeScaleB[1], TILE_SIZE}, bType.getElementType()), 
      ValueRange{});
  Value scaleBBroadcasted = builder.create<linalg::BroadcastOp>(
      location, scaleBFinal, emptyScaleBBroadcasted, ArrayRef<int64_t>{2}).getResult()[0];
  SmallVector<ReassociationIndices> reassocIdxScaleB{{0}, {1, 2}};
  Value scaleBCollapsed = builder.create<tensor::CollapseShapeOp>(
      location, scaleBBroadcasted, reassocIdxScaleB);

  // Transpose scale B
  auto scaleBCollapsedType = cast<RankedTensorType>(scaleBCollapsed.getType());
  auto shapeScaleBCollapsed = scaleBCollapsedType.getShape();
  SmallVector<int64_t> shapeScaleBTransposed{shapeScaleBCollapsed[1], shapeScaleBCollapsed[0]};
  Value emptyScaleBTransposed = builder.create<tensor::EmptyOp>(
      location, RankedTensorType::get(shapeScaleBTransposed, scaleBCollapsedType.getElementType()), ValueRange{});
  Value scaleBTransposed = builder.create<linalg::TransposeOp>(
      location, scaleBCollapsed, emptyScaleBTransposed, ArrayRef<int64_t>{1, 0})->getResult(0);

  // Multiply input and scale
  auto linalgFnAttr = builder.getNamedAttr("fun", builder.getAttr<linalg::BinaryFnAttr>(linalg::BinaryFn::mul));
  Value emptyA = builder.create<tensor::EmptyOp>(location, aType, ValueRange{});
  Value emptyB = builder.create<tensor::EmptyOp>(location, bType, ValueRange{});
  Value aFinal = builder.create<linalg::ElemwiseBinaryOp>(
      location, ValueRange{a, scaleACollapsed}, ValueRange{emptyA}, linalgFnAttr)->getResult(0);
  Value bFinal = builder.create<linalg::ElemwiseBinaryOp>(
      location, ValueRange{b, scaleBTransposed}, ValueRange{emptyB}, linalgFnAttr)->getResult(0);

  // Replace hfusion.matmul_mx with linalg.matmul
  return SmallVector<Value>{
      builder.create<linalg::MatmulOp>(location, ValueRange{aFinal, bFinal}, ValueRange{c})->getResult(0)};
}
