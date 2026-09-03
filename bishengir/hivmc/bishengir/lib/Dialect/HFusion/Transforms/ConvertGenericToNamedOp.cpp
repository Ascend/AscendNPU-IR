//===- ConvertGenericToNamedOp.cpp - Linalg Generic To Named ops Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <set>

namespace mlir {
#define GEN_PASS_DEF_CONVERTGENERICTONAMEDOP
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

#include "llvm/ADT/StringMap.h"
using namespace mlir;
using namespace mlir::hfusion;

static constexpr int64_t kUnaryInputSize = 1;
static constexpr int64_t kBinaryInputSize = 2;

namespace mlir::linalg {

static const llvm::StringMap<UnaryFn> unaryMap = {
    {"math.exp", UnaryFn::exp},
    {"math.absf", UnaryFn::abs},
    {"math.log", UnaryFn::log},
};

static const llvm::StringMap<BinaryFn> binaryMap = {
    {"arith.addf", BinaryFn::add},
    {"arith.mulf", BinaryFn::mul},
    {"arith.subf", BinaryFn::sub},
    {"arith.divf", BinaryFn::div},
    {"arith.addi", BinaryFn::add},
    {"arith.muli", BinaryFn::mul},
    {"arith.subi", BinaryFn::sub},
    {"arith.divi", BinaryFn::div},
    {"arith.maxui", BinaryFn::max_unsigned},
    {"arith.maxsi", BinaryFn::max_signed},
    {"arith.minui", BinaryFn::min_unsigned},
    {"arith.minsi", BinaryFn::min_signed},
};
} // namespace mlir::linalg

namespace mlir::hfusion {

static const llvm::StringMap<UnaryFn> unaryMap = {
    {"math.sqrt", UnaryFn::sqrt},
    {"math.rsqrt", UnaryFn::rsqrt},
    {"arith.divf", UnaryFn::rec},
    {"arith.maximumf", UnaryFn::relu}};

static const llvm::StringMap<BinaryFn> binaryMap = {
    {"arith.andi", BinaryFn::vand},
    {"arith.ori", BinaryFn::vor},
    {"arith.maximumf", BinaryFn::maxf},
    {"arith.minimumf", BinaryFn::minf},
    {"arith.xori", BinaryFn::vxor}
};
} // namespace mlir::hfusion

template <typename T>
static std::optional<T> getFnKind(const llvm::StringMap<T> &map,
                                  StringRef key) {
  auto iter = map.find(key);
  if (iter == map.end()) {
    return std::nullopt;
  }
  return iter->second;
}

static Operation *getSingleComputeOp(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return nullptr;
  auto *bodyOp = yieldOp.getValues().front().getDefiningOp();
  assert(bodyOp != nullptr);
  return bodyOp;
}

static std::string getOpName(Operation *op) {
  assert(op != nullptr && "op must not be nullptr");
  if (op == nullptr) {
    return "";
  }
  return op->getName().getStringRef().str();
}

static std::string getOpName(linalg::GenericOp op) {
  return getOpName(getSingleComputeOp(op));
}

template <typename AttrTy, typename T>
static std::optional<Attribute> getFnAttr(const llvm::StringMap<T> &map,
                                          Operation *bodyOp) {
  assert(bodyOp != nullptr && "bodyOp must not be nullptr");
  std::string opName = getOpName(bodyOp);
  MLIRContext *ctx = bodyOp->getContext();
  auto kind = getFnKind(map, opName);
  if (!kind.has_value()) {
    return std::nullopt;
  }
  return AttrTy::get(ctx, kind.value());
}

template <typename T>
static bool isConstantEqual(Value oper, T cstValue) {
  if (auto cstOp = oper.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(cstOp.getValue())) {
      const double epsilon = 1e-9;
      double lhs = floatAttr.getValueAsDouble();
      double rhs = static_cast<double>(cstValue);
      return (std::abs(lhs - rhs) < epsilon);
    }
  }
  return false;
}

template <typename T>
static bool isConstantOp(linalg::GenericOp genericOp, T cst) {
  auto *bodyOp = getSingleComputeOp(genericOp);
  auto isConstant = [&cst](Value v) { return isConstantEqual(v, cst); };
  return llvm::any_of(genericOp.getInputs(), isConstant) ||
         (bodyOp != nullptr && llvm::any_of(bodyOp->getOperands(), isConstant));
}

static bool isReluOp(linalg::GenericOp genericOp) {
  if (getOpName(genericOp) != "arith.maximumf")
    return false;
  return isConstantOp(genericOp, 0.0);
}

static bool isRecOp(linalg::GenericOp genericOp) {
  if (getOpName(genericOp) != "arith.divf")
    return false;
  return isConstantOp(genericOp, 1.0);
}

static std::optional<Attribute> getElemwiseFunAttr(linalg::GenericOp op) {
  std::string opName = getOpName(op);
  Operation *bodyOp = getSingleComputeOp(op);
  if (opName == "arith.maximumf" || opName == "arith.divf") {
    if (isReluOp(op) || isRecOp(op)) {
      return getFnAttr<hfusion::UnaryFnAttr>(hfusion::unaryMap, bodyOp);
    }

    // if opName is arith.maximumf, need to find the BinaryFnKind from
    // hfusion::BinaryFn otherwise will return an empty attr which can cause
    // compile error
    // TODO: need to refactor the logic here to solve the problem completely
    auto attr = getFnAttr<linalg::BinaryFnAttr>(linalg::binaryMap, bodyOp);
    if (!attr.has_value()) {
      attr = getFnAttr<hfusion::BinaryFnAttr>(hfusion::binaryMap, bodyOp);
    }
    return attr;
  }

  if (hfusion::unaryMap.contains(opName)) {
    return getFnAttr<hfusion::UnaryFnAttr>(hfusion::unaryMap, bodyOp);
  }
  if (linalg::unaryMap.contains(opName)) {
    return getFnAttr<linalg::UnaryFnAttr>(linalg::unaryMap, bodyOp);
  }
  if (hfusion::binaryMap.contains(opName)) {
    return getFnAttr<hfusion::BinaryFnAttr>(hfusion::binaryMap, bodyOp);
  }
  if (linalg::binaryMap.contains(opName)) {
    return getFnAttr<linalg::BinaryFnAttr>(linalg::binaryMap, bodyOp);
  }
  return std::nullopt;
}

static SmallVector<Value> getInputsValue(linalg::GenericOp genericOp,
                                         int64_t inputSize, Attribute attr) {
  SmallVector<Value> src;
  auto *bodyOp = getSingleComputeOp(genericOp);
  if (auto unaryFnAttr = dyn_cast<hfusion::UnaryFnAttr>(attr)) {
    if ((unaryFnAttr.getValue() == hfusion::UnaryFn::rec ||
         unaryFnAttr.getValue() == hfusion::UnaryFn::relu) &&
        inputSize == kBinaryInputSize) {
      auto oper0 = genericOp.getInputs()[0];
      auto oper1 = genericOp.getInputs()[1];
      if (oper0.getDefiningOp())
        src = {oper1};
      else
        src = {oper0};
      return src;
    }
  } else if ((isa<hfusion::BinaryFnAttr>(attr) ||
              isa<linalg::BinaryFnAttr>(attr)) &&
             inputSize == kUnaryInputSize) {
    if (bodyOp == nullptr) {
      return {};
    }
    auto oper0 = bodyOp->getOperands()[0];
    auto oper1 = bodyOp->getOperands()[1];
    if (oper0.getDefiningOp())
      src = {oper0, genericOp.getInputs()[0]};
    else
      src = {genericOp.getInputs()[0], oper1};
    return src;
  }
  return genericOp.getInputs();
}

static Operation *createNamedOp(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter, Attribute attr) {
  auto loc = genericOp.getLoc();
  int64_t inputSize = static_cast<int64_t>(genericOp.getInputs().size());
  SmallVector<Value> src = getInputsValue(genericOp, inputSize, attr);
  SmallVector<Value> dst = genericOp.getOutputs();
  SmallVector<NamedAttribute> attrs;
  auto nameAttr = StringAttr::get(genericOp.getContext(), "fun");
  attrs.push_back({nameAttr, attr});
  Operation *namedOp = nullptr;
  if (isa<hfusion::UnaryFnAttr>(attr)) {
    namedOp = rewriter.create<hfusion::ElemwiseUnaryOp>(loc, src, dst, attrs);
  } else if (isa<linalg::UnaryFnAttr>(attr)) {
    namedOp = rewriter.create<linalg::ElemwiseUnaryOp>(loc, src, dst, attrs);
  } else if (isa<hfusion::BinaryFnAttr>(attr)) {
    namedOp = rewriter.create<hfusion::ElemwiseBinaryOp>(loc, src, dst, attrs);
  } else if (isa<linalg::BinaryFnAttr>(attr)) {
    namedOp = rewriter.create<linalg::ElemwiseBinaryOp>(loc, src, dst, attrs);
  }
  return namedOp;
}

static bool atLeastOneComputeOp(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp) {
    return false;
  }
  auto *bodyOp = yieldOp.getValues()[0].getDefiningOp();
  return bodyOp != nullptr;
}

struct ConvertElemwiseLinalgGenericOps
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(op) || !atLeastOneComputeOp(op)) {
      return rewriter.notifyMatchFailure(
          op, "unsupport named structure for this generic type.");
    }
    auto attr = getElemwiseFunAttr(op);
    if (!attr.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "fail to get attribute for generic op");
    }
    Operation *namedOp = createNamedOp(op, rewriter, attr.value());
    if (!namedOp) {
      return rewriter.notifyMatchFailure(op, "fail to create named op");
    }
    rewriter.replaceOp(op, namedOp->getResults());
    return success();
  }
};

class ConvertGenericToNamedOp
    : public impl::ConvertGenericToNamedOpBase<ConvertGenericToNamedOp> {
public:
  void runOnOperation() override;
};

void ConvertGenericToNamedOp::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<ConvertElemwiseLinalgGenericOps>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::hfusion::createConvertGenericToNamedOpPass() {
  return std::make_unique<ConvertGenericToNamedOp>();
}
