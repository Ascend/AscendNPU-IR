//===- Matmul.cpp - HFusion to HIVM dialect conversion for matmul ---------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"
#include <type_traits>

using namespace mlir;

namespace {

constexpr static llvm::StringLiteral kPostVectorFuncTagName =
    "post_vector_func";

constexpr static llvm::StringLiteral kPostVectorFuncArgsTagName =
    "post_vector_func_args";

//===----------------------------------------------------------------------===//
// Conversion to HIVM Local MatmulOp
//===----------------------------------------------------------------------===//

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, linalg::MatmulOp> ||
                                      std::is_same_v<T, linalg::BatchMatmulOp>>>
class MmadL1InfoCollector {
public:
  explicit MmadL1InfoCollector(const T op) : op_(op) {
    mmadL1A_ = op_.getDpsInputOperand(0)->get();
    if (isTranposeLastAxis(mmadL1A_).has_value()) {
        transposeA_ = true;
        mmadL1A_ = isTranposeLastAxis(mmadL1A_).value();
    }
    mmadL1B_ = op_.getDpsInputOperand(1)->get();
    if (isTranposeLastAxis(mmadL1B_).has_value()) {
        transposeB_ = true;
        mmadL1B_ = isTranposeLastAxis(mmadL1B_).value();
    }

    std::string inputPrecisionStr{"input_precision"};
    if (auto attr = op_->getAttr(inputPrecisionStr)) {
      if (dyn_cast<StringAttr>(attr).getValue() == "hf32") {
        enableHF32_ = true;
      }
    }

    mmadL0C_ = op_.getDpsInitOperand(0)->get();
  }

  T getSourceMatmulOp() const { return op_; };

  template <typename ReplaceOpTy>
  Operation *getReplacementOp(PatternRewriter &rewriter) {
    // Stub value 0 for mkn
    auto constZero = rewriter.create<arith::ConstantIndexOp>(
        getSourceMatmulOp().getLoc(), 0);
    auto newOp = rewriter.template create<ReplaceOpTy>(
        getSourceMatmulOp().getLoc(),
        getMmadL1OpResultTypes(),          // result types
        mmadL1A_,                          // Matrix A on L1
        mmadL1B_,                          // Matrix B on L1
        initCondition_,                    // L0C init condition
        constZero,                         // MMAD Real M
        constZero,                         // MMAD Real K
        constZero,                         // MMAD Real N
        mmadL0C_,                          // init operand
        Value{},                           // per channel bias
        getMmadL1TransposeAFlag(rewriter), // transpose A
        getMmadL1TransposeBFlag(rewriter), // transpose B
        getMmadL1EnableHF32Flag(rewriter)  // enable hf32 mode
    );
    return newOp.getOperation();
  }

  /// MMAD Init condition is inferred from the scf.for enclosing the matmul
  /// operation.
  /// For example, for the below IR:
  /// \code
  /// %0 = tensor.empty() : tensor<?x?xf32>
  /// %cst = linalg.fill ins(%cst: f32) outs(%0: tensor<?x?xf32>)
  /// ...
  /// scf.for %arg0 = lower_bound ... iter_args(%arg1 = %cst) { // K loop
  ///  scf.for %arg2 = lower_bound1 ... iter_args(%arg3 = %arg1) { // K loop
  ///    %ret = linalg.matmul ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
  ///                         outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  /// \endcode
  /// the init condition is (%arg0 == lower_bound) && (%arg2 == lower_bound1).
  ///
  /// Another example, for the below IR:
  /// \code
  /// %0 = tensor.empty() : tensor<?x?xf32>
  /// %cst = linalg.fill ins(%cst: f32) outs(%0: tensor<?x?xf32>)
  /// ...
  /// %res = scf.for %arg0 = lower_bound to upper_bound ... iter_args(%arg1 = %cst) { // K loop
  ///  %ret = linalg.matmul ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
  ///                       outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  ///  yiled %ret
  /// %res1 = scf.for %arg2 = lower_bound1 to upper_bound1 ... iter_args(%arg3 = %res) { // K loop
  ///  %ret = linalg.matmul ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
  ///                       outs(%arg3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  ///  yiled %ret
  /// \endcode
  /// the init condition is (%arg2 == lower_bound1) && (lower_bound >= upper_bound).
  void extractInitCondition(PatternRewriter &rewriter);

  /// Judge whether the input of mmad can be trasposed along being loaded
  std::optional<Value> isTranposeLastAxis(Value v) {
    auto l1TransposeOp = v.getDefiningOp<linalg::TransposeOp>();
    if (!l1TransposeOp)
      return std::nullopt;

    auto perm = l1TransposeOp.getPermutation();
    const auto rank = static_cast<int>(perm.size());
    if (rank < 2)
      llvm_unreachable("rank for matmul need not less than 2");
    if ((perm[rank - 1] == rank - 2) && (perm[rank - 2] == rank - 1))
      return l1TransposeOp.getInput();

    return std::nullopt;
  }

private:
  /// Information related to the init tensor.
  struct InitTensorInfo {
    /// Current value to inspect.
    Value currentValue;
    /// Init condition.
    Value currentCondition;
    /// Init tensor's argument index.
    unsigned int initTensorIterArgIndex{0};
    /// Pointer to the outermost scf::ForOp that uses the init tensor.
    Operation *initTensorOutermostLoop{nullptr};
  };

  /// Help to judge MmadL1 destination(c) is initialized from one empty space,
  /// if so, `init` flag will be set true, then real MmadL1 will clean up
  /// destination data at first
  /// \Note Currently, judgement requires tensor must be a linalg.fill op with
  /// zero value or just a tensor.empty op
  static bool isZeroOrEmptyTensor(Value op);

  /// Helper function to set init flag to true when prove MmadL1 destination(c)
  /// is empty space with `isZeroOrEmptyTensor` func
  /// For loop state which outermost intialization of dst satisfies empty space,
  /// here use recursion to gradually build up the init flag
  LogicalResult buildInitCondition(InitTensorInfo &info,
                                   PatternRewriter &rewriter) const;

  /// Insert and use new init tensor in linalg::MatmulOp/BatchMatmulOp.
  void insertAndUseNewInitTensor(InitTensorInfo info,
                                 PatternRewriter &rewriter);

  SmallVector<Type> getMmadL1OpResultTypes() const;
  UnitAttr getMmadL1TransposeAFlag(OpBuilder &rewriter) const;
  UnitAttr getMmadL1TransposeBFlag(OpBuilder &rewriter) const;
  UnitAttr getMmadL1EnableHF32Flag(OpBuilder &rewriter) const;

  /// Original Op
  T op_;
  /// Attributes for MmadL1Op
  bool transposeA_{false};
  bool transposeB_{false};
  bool enableHF32_{false};

  /// Operands for MmadL1Op
  Value mmadL1A_;
  Value mmadL1B_;
  Value mmadL0C_;
  Value initCondition_;
};

template <typename T, typename U>
SmallVector<Type> MmadL1InfoCollector<T, U>::getMmadL1OpResultTypes() const {
  return SmallVector<Type>{op_->getResultTypes()};
}

template <typename T, typename U>
UnitAttr
MmadL1InfoCollector<T, U>::getMmadL1TransposeAFlag(OpBuilder &rewriter) const {
  return transposeA_ ? rewriter.getUnitAttr() : UnitAttr();
}

template <typename T, typename U>
UnitAttr
MmadL1InfoCollector<T, U>::getMmadL1TransposeBFlag(OpBuilder &rewriter) const {
  return transposeB_ ? rewriter.getUnitAttr() : UnitAttr();
}

template <typename T, typename U>
UnitAttr
MmadL1InfoCollector<T, U>::getMmadL1EnableHF32Flag(OpBuilder &rewriter) const {
  return enableHF32_ ? rewriter.getUnitAttr() : UnitAttr();
}

template <typename T, typename U>
void MmadL1InfoCollector<T, U>::extractInitCondition(
    PatternRewriter &rewriter) {
  InitTensorInfo initInfo;
  initInfo.currentValue = mmadL0C_;

  // Defaultly create init flag as 'true' for state where MmadL1 destination
  // could be inferred as zero data
  initInfo.currentCondition = rewriter.create<arith::ConstantIntOp>(
      op_->getLoc(), /*value*/ 1, /*width*/ 1);
  // Get defining op for init tensor and build up condition
  if (succeeded(buildInitCondition(initInfo, rewriter))) {
    initCondition_ = initInfo.currentCondition;
    insertAndUseNewInitTensor(initInfo, rewriter);
    return;
  }

  // Otherwise, init flag should be `false` as MmadL1 destination(c) has
  // meaningful value
  initCondition_ = rewriter.create<arith::ConstantIntOp>(
      op_->getLoc(), /*value*/ 0, /*width*/ 1);
}

template <typename T, typename U>
bool MmadL1InfoCollector<T, U>::isZeroOrEmptyTensor(Value op) {
  auto emptyOp = op.getDefiningOp<tensor::EmptyOp>();
  if (emptyOp) {
    return true;
  }

  auto linalgFill = op.getDefiningOp<linalg::FillOp>();
  if (!linalgFill) {
    return false;
  }
  auto cstValue = linalgFill.getDpsInputOperand(0)
                      ->get()
                      .getDefiningOp<arith::ConstantOp>();
  if (!cstValue) {
    return false;
  }
  // Check if value is constant int or float zero.
  std::optional<int64_t> cstInt = getConstantIntValue(cstValue.getValue());
  if (cstInt && (*cstInt) == 0) {
    return true;
  }
  auto cstFloat = dyn_cast_if_present<FloatAttr>(cstValue.getValue());
  return cstFloat && cstFloat.getValue().isZero();
}


template <typename T, typename U>
LogicalResult
MmadL1InfoCollector<T, U>::buildInitCondition(InitTensorInfo &info,
                                              PatternRewriter &rewriter) const {
  // Base Case: If current destination value satisfies empty space, return
  if (isZeroOrEmptyTensor(info.currentValue)) {
    if (info.currentValue.hasOneUse()) {
      return success();
    }

    // TODO: add restriction when block argument have several users (even if this users are matmul)
    // for i iter_arg(%arg0 = ..., % arg1 = ...)
    // %res0 = mm outs(%arg0)
    // %res1 = mm outs(%arg0)
    // scf.yield %res0, %res1
    for (auto use : info.currentValue.getUsers()) {
      auto forOp = dyn_cast<scf::ForOp>(use);
      if (!forOp) {
        continue;
      }

      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        if (forOp.getInitArgs()[i] != info.currentValue) {
          continue;
        }
        
        BlockArgument regionIterArg = forOp.getRegionIterArg(i);
        OpOperand *tiedYielded = forOp.getTiedLoopYieldedValue(regionIterArg);
        if (!tiedYielded) {
          continue;
        }

        auto yieldedMatmul = hivm::traceDefOp<T>(tiedYielded->get());
        if (!yieldedMatmul.has_value()) {
          continue;
        }

        auto matmulOp = cast<T>(yieldedMatmul.value());
        if (matmulOp.getDpsInitOperand(0)->get() == info.currentValue) {
          return failure();
        }
      }
    }

    return success();
  }

  // TODO: to remove the code by doing bmm decomposition before it.
  if constexpr (std::is_same_v<T, linalg::BatchMatmulOp>) {
    return failure();
  }

  // Case A: L0C is an iteration argument of scf::ForOp
  if (auto blockArg = dyn_cast<BlockArgument>(info.currentValue)) {
    auto scfForOp =
        dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!scfForOp) {
      return failure();
    }

    if (OpOperand* tiedYielded = scfForOp.getTiedLoopYieldedValue(blockArg)) {
      // TODO: Change to potential definers analysis which returns set of definers to fix if
      if (!info.initTensorOutermostLoop && !hivm::traceDefOp<T>(tiedYielded->get()).has_value()) {
        return failure();
      }
    }

    OpOperand *iterArgOperand = scfForOp.getTiedLoopInit(blockArg);

    // Update information.
    info.initTensorOutermostLoop = scfForOp.getOperation();
    info.initTensorIterArgIndex = iterArgOperand->getOperandNumber();
    info.currentValue = iterArgOperand->get();

    auto loc = info.currentCondition.getLoc();
    // Init condition for current loop is `(iv == lower_bound)`
    auto isFirstIter = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, scfForOp.getLowerBound(),
        scfForOp.getInductionVar());

    // Joint condition is `(currentCondition && (iv == lower_bound))`
    info.currentCondition =
        rewriter.create<arith::AndIOp>(loc, info.currentCondition, isFirstIter);

    // Directly reuse and pass 'info' upward
    return buildInitCondition(info, rewriter);
  }

  // Case B: L0C is the result of scf::ForOp
  if (auto prevLoop =
          dyn_cast_if_present<scf::ForOp>(info.currentValue.getDefiningOp())) {
    auto resultIndex = cast<OpResult>(info.currentValue).getResultNumber();
    Value prevInit = prevLoop.getInitArgs()[resultIndex];

    BlockArgument tiedBlockArg = prevLoop.getRegionIterArg(resultIndex);

    // Update information.
    info.initTensorOutermostLoop = prevLoop.getOperation();
    info.initTensorIterArgIndex =
        prevLoop.getTiedLoopInit(tiedBlockArg)->getOperandNumber();
    info.currentValue = prevInit;

    // Init condition for current loop is `LowerBound >= UpperBound`
    auto loc = prevLoop.getLoc();
    auto loopNotRun = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, prevLoop.getLowerBound(),
        prevLoop.getUpperBound());

    // Joint condition is `((LowerBound >= UpperBound) && currentCondition)`
    info.currentCondition =
        rewriter.create<arith::AndIOp>(loc, info.currentCondition, loopNotRun);

    return buildInitCondition(info, rewriter);
  }

  // Unable to trace, return failure (init_c = false)
  return failure();
}

template <typename T, typename U>
void MmadL1InfoCollector<T, U>::insertAndUseNewInitTensor(
    InitTensorInfo info, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  bool usedInLoop = info.initTensorOutermostLoop != nullptr;
  // If initTensorOutermostLoop is not defined, it means that there is no
  // loop that uses current init tensor as iter arg. So we can insert the
  // new tensor wherever we like. Otherwise, insert the new init tensor
  // right before the loop.
  if (usedInLoop) {
    rewriter.setInsertionPoint(info.initTensorOutermostLoop);
  }
  Value newInitResult = info.currentValue;
  auto linalgFill = info.currentValue.template getDefiningOp<linalg::FillOp>();
  if (linalgFill) {
    auto newInit = rewriter.clone(
        *(linalgFill.getDpsInitOperand(0)->get().getDefiningOp()));
    newInitResult = newInit->getResults().front();
  }

  if (usedInLoop) {
    // If the init tensor is passed into the for loop as an iter arg,
    // we only need to replace the block argument.
    auto scfForOp = dyn_cast<scf::ForOp>(info.initTensorOutermostLoop);
    assert(scfForOp);
    rewriter.modifyOpInPlace(scfForOp, [&]() {
      scfForOp->setOperand(info.initTensorIterArgIndex, newInitResult);
    });

    return;
  }
  Value oldInit = mmadL0C_;
  op_->replaceUsesOfWith(oldInit, newInitResult);
  mmadL0C_ = newInitResult;
}

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, linalg::MatmulOp> ||
                                      std::is_same_v<T, linalg::BatchMatmulOp>>>
struct MadLikeMapping;

template <>
struct MadLikeMapping<linalg::MatmulOp> {
  using U = typename hivm::MmadL1Op;
};

template <>
struct MadLikeMapping<linalg::BatchMatmulOp> {
  using U = typename hivm::BatchMmadL1Op;
};

/// Rewriting rule that combines Linalg Ops to create hivm mmadl1 like op
///   - linalg::MatmulOp is mapped to hivm::MmadL1Op while
///     linalg::BatchMatmulOp is mapped to hivm::BatchMmadL1Op.
///   - If linalg::TransposeOp is the producer of L1 Tensors, transpose
///     attribute will be added to hivm::MmadL1Op.
///   - Init condition is extracted from the IR.
///   - A new init tensor is created and inserted before the outermost K loop.
template <typename T>
class FuseOpsToMmadL1LikeOp : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  using U = typename MadLikeMapping<T>::U;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    MmadL1InfoCollector<T, U> info(op);
    // Get L0C init condition.
    info.extractInitCondition(rewriter);

    rewriter.replaceOp(info.getSourceMatmulOp(),
                       info.template getReplacementOp<U>(rewriter));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion to HIVM Global MatmulOp
//===----------------------------------------------------------------------===//

template <typename SrcOp>
struct MatmulOpToHIVMMatmulOp : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    // convert
    // linalg::MatmulOp/MatmulTransposeAOp/MatmulTransposeBOp/hfuion::gmm to
    // hivm::MatmulOp
    OpBuilder::InsertionGuard guard(rewriter);
    auto operand1 = op.getOperand(0);
    auto operand2 = op.getOperand(1);
    auto result = op.getOperand(2);
    UnitAttr transposeAAttr{};
    UnitAttr transposeBAttr{};
    if (isa<linalg::MatmulTransposeAOp>(op)) {
      transposeAAttr = rewriter.getUnitAttr();
    } else if (isa<linalg::MatmulTransposeBOp>(op)) {
      transposeBAttr = rewriter.getUnitAttr();
    }

    // backward compatible to some test cases
    if (op.hasPureBufferSemantics()) {
      rewriter.replaceOpWithNewOp<hivm::MatmulOp>(
          op, /*result=*/op->getResultTypes(), /*a=*/operand1, /*b=*/operand2,
          /*c=*/result,
          /*aTranspose=*/transposeAAttr, /*bTranspose=*/transposeBAttr);
      return success();
    }

    // collect hivm.matmul op info
    SmallVector<Value> postVecIns{};
    SmallVector<Value> workspaceIns{};
    Value tilingParams{};
    auto tilingAnnotateOps = utils::getAnnotateOpWithAttr(
        op->getResult(0),
        hacc::stringifyEnum(hacc::KernelArgType::kTilingStruct));
    if (tilingAnnotateOps.has_value()) {
      annotation::MarkOp markOp =
          dyn_cast<annotation::MarkOp>(tilingAnnotateOps.value());
      tilingParams = markOp.getValues().front();
      rewriter.eraseOp(markOp);
    }

    auto workspaceAnnotateOps = utils::getAllAnnotateOpsWithAttr(
        op->getResult(0), hacc::stringifyEnum(hacc::KernelArgType::kWorkspace));
    for (auto *annotateOp : workspaceAnnotateOps) {
      annotation::MarkOp markOp = dyn_cast<annotation::MarkOp>(annotateOp);
      if (markOp.getSrc() != op->getResult(0)) {
        continue;
      }
      for (auto value : markOp.getValues()) {
        Operation *defineOp = value.getDefiningOp();
        rewriter.setInsertionPointAfter(defineOp);
        auto tensorType = cast<TensorType>(value.getType());
        auto shapes = tensorType.getShape();
        if (shapes.size() > 1) {
          ReassociationIndices assocationIndices;
          for (size_t i = 0; i < shapes.size(); i++) {
            assocationIndices.push_back(i);
          }
          value = rewriter.create<tensor::CollapseShapeOp>(
              defineOp->getLoc(), value, assocationIndices);
        }
        workspaceIns.push_back(value);
      }
      rewriter.eraseOp(annotateOp);
    }

    auto postVecAnnotateOps = utils::getAllAnnotateOpsWithAttr(
        op->getResult(0), kPostVectorFuncArgsTagName);
    for (auto *annotateOp : postVecAnnotateOps) {
      annotation::MarkOp markOp = dyn_cast<annotation::MarkOp>(annotateOp);
      if (markOp.getSrc() != op->getResult(0)) {
        continue;
      }
      for (auto value : markOp.getValues()) {
        postVecIns.push_back(value);
      }
      rewriter.eraseOp(annotateOp);
    }

    // collect dummy call op
    SmallVector<std::pair<func::CallOp, func::FuncOp>> dummyOps;
    auto func = op->template getParentOfType<func::FuncOp>();
    auto mod = func->template getParentOfType<ModuleOp>();
    for (auto *userOp : op->getUsers()) {
      if (auto callOp = dyn_cast<func::CallOp>(userOp)) {
        auto callee =
            mod.template lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (!callee)
          continue;
        if (callee->getAttr(hacc::DummyFuncAttr::name)) {
          dummyOps.push_back(std::make_pair(callOp, callee));
        }
      }
    }
    if (!dummyOps.empty()) {
      result = dummyOps.back().first->getOperands().back();
    }

    Operation *newOp = nullptr;
    if (workspaceIns.empty() && postVecIns.empty()) {
      newOp = rewriter.replaceOpWithNewOp<hivm::MatmulOp>(
          op, /*result=*/TypeRange{result.getType()}, /*a=*/operand1,
          /*b=*/operand2, /*c=*/result, tilingParams, /*bias=*/Value{},
          /*descale=*/Value{},
          /*aTranspose=*/transposeAAttr, /*bTranspose=*/transposeBAttr,
          /*descaleMode=*/hivm::DescaleModeAttr{});
    } else {
      if (dummyOps.size() != 1u)
        llvm::report_fatal_error("internal error: dummyOps size is not 1");
      rewriter.setInsertionPointAfter(dummyOps.back().first);
      if constexpr (std::is_same_v<SrcOp, hfusion::GroupMatmulOp>) {
        newOp = rewriter.replaceOpWithNewOp<hivm::MixGroupMatmulOp>(
            dummyOps.back().first, /*result=*/TypeRange{result.getType()},
            /*a=*/op.getOperand(0),
            /*b=*/op.getOperand(1),
            /*tokens_per_expert is at operand 2*/ op.getOperand(2),
            /*result output is at operand 3*/ op.getOperand(3),
            /*postVecFuncIns=*/postVecIns,
            /*postVecFuncOuts*/ SmallVector<Value>{}, workspaceIns,
            tilingParams, /*commParams*/ Value{},
            /*bias=*/Value{},
            /*descale=*/Value{}, /*aTranspose=*/transposeAAttr,
            /*bTranspose=*/transposeBAttr,
            /*descaleMode=*/hivm::DescaleModeAttr{});
      } else {
        newOp = rewriter.replaceOpWithNewOp<hivm::MixMatmulOp>(
            dummyOps.back().first, /*result=*/TypeRange{result.getType()},
            /*a=*/operand1,
            /*b=*/operand2, /*c=*/result, /*postVecFuncIns=*/postVecIns,
            workspaceIns, tilingParams, /*commParams*/ Value{},
            /*bias=*/Value{},
            /*descale=*/Value{}, /*aTranspose=*/transposeAAttr,
            /*bTranspose=*/transposeBAttr,
            /*descaleMode=*/hivm::DescaleModeAttr{});
      }

      rewriter.eraseOp(op);
      rewriter.eraseOp(dummyOps.back().second);
    }

    // add post_vec_func attr
    auto postVAttr = op->getAttr(kPostVectorFuncTagName);
    if (postVAttr) {
      newOp->setAttr(kPostVectorFuncTagName, postVAttr);
    }
    return success();
  }
};

} // namespace

void mlir::populateMatmulPatternsAndLegality(
    RewritePatternSet &patterns, ConversionTarget &target,
    const ConvertHFusionToHIVMOptions &options) {
  target.addIllegalOp<linalg::MatmulOp, linalg::BatchMatmulOp,
                      linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,
                      hfusion::GroupMatmulOp>();
  if (options.mmMapMode == mlir::hfusion::MmMapMode::MacroInstr) {
    patterns.add<FuseOpsToMmadL1LikeOp<linalg::MatmulOp>>(
        patterns.getContext());
    patterns.add<FuseOpsToMmadL1LikeOp<linalg::BatchMatmulOp>>(
        patterns.getContext());
  } else {
    patterns.add<MatmulOpToHIVMMatmulOp<linalg::MatmulOp>>(
        patterns.getContext());
    patterns.add<MatmulOpToHIVMMatmulOp<linalg::MatmulTransposeAOp>>(
        patterns.getContext());
    patterns.add<MatmulOpToHIVMMatmulOp<linalg::MatmulTransposeBOp>>(
        patterns.getContext());
    patterns.add<MatmulOpToHIVMMatmulOp<hfusion::GroupMatmulOp>>(
        patterns.getContext());
  }
}