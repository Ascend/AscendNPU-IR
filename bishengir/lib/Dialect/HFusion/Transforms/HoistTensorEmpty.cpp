//===- HoistTensorEmpty.cpp ---- Hoist Tensor Empty Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTTENSOREMPTY
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
void eraseTriviallyDeadOps(ArrayRef<Operation *> ops) {
  for (auto I = ops.rbegin(), E = ops.rend(); I != E; ++I) {
    Operation *curOp = *I;
    if (isOpTriviallyDead(curOp))
      curOp->erase();
  }
}
} // namespace

LogicalResult CloneNewTensorEmpty(tensor::EmptyOp op, OpBuilder &opBuilder) {
  if (op->getUses().empty()) {
    return failure();
  }
  for (OpOperand &use : op->getUses()) {
    if (use == *(op->use_begin())) {
      continue;
    }
    Operation *useOp = use.getOwner();
    auto operandIdx = use.getOperandNumber();
    Value operand = useOp->getOperand(operandIdx);
    opBuilder.setInsertionPoint(op);
    auto *clonedOp = opBuilder.clone(*op);
    useOp->replaceUsesOfWith(operand, clonedOp->getResult(0));
  }
  return success();
}

LogicalResult applyDistributeTensorEmpty(func::FuncOp funcOp) {
  OpBuilder opBuilder(funcOp.getContext());
  SmallVector<tensor::EmptyOp> emptyOps;
  // collect tensor.empty has more than one use
  funcOp->walk([&](tensor::EmptyOp op) {
    if (op->getUses().empty() || op->hasOneUse()) {
      return WalkResult::skip();
    }
    emptyOps.push_back(op);
    return WalkResult::advance();
  });
  // distribute tensor.empty
  for (auto op : emptyOps) {
    if (failed(CloneNewTensorEmpty(op, opBuilder))) {
      return failure();
    }
  }
  return success();
}

/// This pass hoist tensor empty to func parameters and merge into one parameter
struct HoistTensorEmptyPass
    : public impl::HoistTensorEmptyBase<HoistTensorEmptyPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult addToEmptyList(func::FuncOp funcOp,
                               SmallVector<Operation *, 4> &opsToErase) {
    funcOp.walk(
        [&](tensor::EmptyOp emptyOp) { opsToErase.push_back(emptyOp); });
    return success();
  }

  unsigned getIntegerOrFloatTypeByteSizes(Type type) {
    return getElementTypeOrSelf(type).getIntOrFloatBitWidth() / UINT8_WIDTH;
  }

  std::optional<int64_t> getUnifiedStaticTotalSize(Operation *ops,
                                                   Type unifiedType) {
    auto tensorType = cast<TensorType>(ops->getResult(0).getType());
    auto staticShapes = tensorType.getShape();
    // compute shape scale size
    unsigned tensorTypeOfByteSizes = getIntegerOrFloatTypeByteSizes(tensorType);
    unsigned unifiedTypeOfByteSizes =
        getIntegerOrFloatTypeByteSizes(unifiedType);
    assert(unifiedTypeOfByteSizes % tensorTypeOfByteSizes == 0);
    int64_t shapeSizeScale = unifiedTypeOfByteSizes / tensorTypeOfByteSizes;
    // compute Unified Static Total Size
    std::optional<int64_t> totalStaticSize =
        utils::getStaticTotalSize(staticShapes);
    if (!totalStaticSize.has_value()) {
      return std::nullopt;
    }
    return totalStaticSize.value() * shapeSizeScale;
  }

  std::optional<Value> getUnifiedTotalSize(Operation *ops, Type unifiedType,
                                           OpBuilder &builder, Location loc) {
    builder.setInsertionPointAfter(ops);

    auto tensorType = cast<TensorType>(ops->getResult(0).getType());
    auto shapes = tensorType.getShape();
    // shape is static: create arith.constantIndex op
    if (!ShapedType::isDynamicShape(shapes)) {
      auto totalStaticSize = getUnifiedStaticTotalSize(ops, unifiedType);
      assert(totalStaticSize.has_value());
      if (!totalStaticSize.has_value()) {
        return std::nullopt;
      }
      return builder.create<arith::ConstantIndexOp>(loc,
                                                    totalStaticSize.value());
    }
    // compute shape scale size
    unsigned tensorTypeOfByteSizes = getIntegerOrFloatTypeByteSizes(tensorType);
    unsigned unifiedTypeOfByteSizes =
        getIntegerOrFloatTypeByteSizes(unifiedType);
    assert(unifiedTypeOfByteSizes % tensorTypeOfByteSizes == 0);
    int64_t shapeSizeScale = unifiedTypeOfByteSizes / tensorTypeOfByteSizes;
    auto scaleConstantOp =
        builder.create<arith::ConstantIndexOp>(loc, shapeSizeScale);
    // compute dynamic total size
    auto emptyOps = dyn_cast<tensor::EmptyOp>(ops);
    Value totalSize = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (size_t i = 0; i < shapes.size(); i++) {
      Value operand;
      if (shapes[i] == ShapedType::kDynamic) {
        operand = emptyOps.getDynamicSize(i);
      } else {
        operand = builder.create<arith::ConstantIndexOp>(loc, shapes[i]);
      }
      totalSize = builder.create<arith::MulIOp>(loc, totalSize.getType(),
                                                totalSize, operand);
    }
    return builder.create<arith::MulIOp>(loc, totalSize.getType(), totalSize,
                                         scaleConstantOp);
  }

  SmallVector<func::CallOp> getCallOps(ModuleOp module, StringRef symName) {
    SmallVector<func::CallOp> callOps;
    module->walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == symName) {
        callOps.push_back(callOp);
      }
    });
    return callOps;
  }

  /// Compute Workspace Shape Size needed by following op build
  int64_t getWorkSpaceSize(SmallVector<Operation *, 4> &opsToErase, Type type) {
    int64_t workspaceSize = 0;
    for (Operation *op : opsToErase) {
      std::optional<int64_t> totalStaticSize =
          getUnifiedStaticTotalSize(op, type);
      if (!totalStaticSize.has_value()) {
        return ShapedType::kDynamic;
      }
      workspaceSize = workspaceSize + totalStaticSize.value();
    }
    return workspaceSize;
  }

  Type getUnifyElementTensorType(SmallVector<Operation *, 4> &opsToErase) {
    auto unifiedType =
        cast<TensorType>(opsToErase.front()->getResult(0).getType())
            .getElementType();
    for (Operation *op : opsToErase) {
      auto currOpType = cast<TensorType>(op->getResult(0).getType());
      if (unifiedType != currOpType.getElementType()) {
        op->emitError("element type of all tensor empty op should be same!");
      }
    }
    return unifiedType;
  }

  BlockArgument addWorkspaceFuncArgument(func::FuncOp funcOp,
                                         int64_t workspaceSize,
                                         Type unifiedType) {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> callOps = getCallOps(module, funcOp.getSymName());
    auto endIdx = funcOp.getNumArguments() - 1;
    auto loc = funcOp.getArgument(endIdx).getLoc();
    // use memref type to solve expand_shape bufferize error.
    MemRefType argumentType = MemRefType::get({workspaceSize}, unifiedType);
    funcOp.insertArgument(
        endIdx + 1, argumentType,
        DictionaryAttr::get(
            &getContext(),
            SmallVector<NamedAttribute>{hacc::createHACCKernelArgAttr(
                &getContext(), hacc::KernelArgType::kWorkspace)}),
        loc);
    return funcOp.getArgument(endIdx + 1);
  }

  memref::ReinterpretCastOp generateSubViewOp(OpBuilder &opBuilder,
                                              Type unifiedType, Value argument,
                                              Value offsets, Value totalSize,
                                              Location loc) {
    Operation *totalSizeDefOp = totalSize.getDefiningOp();
    IntegerAttr ones = opBuilder.getIndexAttr(1);
    OpFoldResult size = totalSize;
    int64_t shapeSize = ShapedType::kDynamic;
    // static total size
    if (auto constantOp = dyn_cast<arith::ConstantIndexOp>(totalSizeDefOp)) {
      shapeSize = constantOp.value();
      size = opBuilder.getIndexAttr(shapeSize);
    }
    MemRefType type = MemRefType::get({shapeSize}, unifiedType);
    auto subviewOp = opBuilder.create<memref::SubViewOp>(
        loc, argument, SmallVector<OpFoldResult>{offsets},
        SmallVector<OpFoldResult>{size}, SmallVector<OpFoldResult>{ones});
    return opBuilder.create<memref::ReinterpretCastOp>(
        loc, type, subviewOp, SmallVector<OpFoldResult>{offsets},
        SmallVector<OpFoldResult>{size}, SmallVector<OpFoldResult>{ones});
  }

  tensor::ExpandShapeOp generateExpandOp(OpBuilder &opBuilder, TensorType type,
                                         Value sliceOp, Location loc) {
    ReassociationIndices assocationIndices;
    for (size_t i = 0; i < type.getShape().size(); i++) {
      assocationIndices.push_back(i);
    }
    return opBuilder.create<tensor::ExpandShapeOp>(
        loc, type, sliceOp,
        SmallVector<ReassociationIndices>{assocationIndices});
  }

  void
  recordOpsForInferWorkspaceShapeFunc(Value offset,
                                      SmallVector<Operation *, 4> &recordOps) {
    Operation *ops = offset.getDefiningOp();
    if (ops == nullptr) {
      return;
    }
    recordOps.insert(recordOps.begin(), ops);
    for (auto operand : ops->getOperands()) {
      recordOpsForInferWorkspaceShapeFunc(operand, recordOps);
    }
  }

  Value mergeToWorkspace(func::FuncOp funcOp,
                         SmallVector<Operation *, 4> &opsToErase,
                         SmallVector<Operation *, 4> &recordOps) {
    if (opsToErase.empty()) {
      return nullptr;
    }

    OpBuilder opBuilder(funcOp->getContext());
    Block *firstBlock = &(funcOp.getBlocks().front());
    assert(firstBlock != nullptr);
    Operation *firstOperation = &(firstBlock->front());
    assert(firstOperation != nullptr);
    opBuilder.setInsertionPoint(firstOperation);
    Type unifiedType = getUnifyElementTensorType(opsToErase);
    // compute workspace size
    int64_t workspaceSize = getWorkSpaceSize(opsToErase, unifiedType);
    // add func param
    BlockArgument workSpaceFuncArg =
        addWorkspaceFuncArgument(funcOp, workspaceSize, unifiedType);
    // replace users of tensor.empty's result by using workSpaceFuncArg
    SmallVector<int64_t> strides{1};
    Value offsets =
        opBuilder.create<arith::ConstantIndexOp>(firstOperation->getLoc(), 0);
    for (Operation *op : opsToErase) {
      opBuilder.setInsertionPointAfter(op);
      Location loc = op->getLoc();

      std::optional<Value> totalSize =
          getUnifiedTotalSize(op, unifiedType, opBuilder, loc);
      assert(totalSize.has_value() && "Failed to get unified total size");
      // subview + reinterpret_cast + to tensor
      auto subview = generateSubViewOp(opBuilder, unifiedType, workSpaceFuncArg,
                                       offsets, totalSize.value(), loc);
      auto toTensor = opBuilder.create<bufferization::ToTensorOp>(
          loc, subview->getResult(0), true, true);
      Value workspaceTensor = toTensor->getResult(0);
      // expand
      auto resultType = op->getResult(0).getType();
      assert(isa<TensorType>(resultType));
      auto resultTensorType = cast<TensorType>(resultType);
      if (resultTensorType.getShape().size() > 1) {
        auto expand =
            generateExpandOp(opBuilder, resultTensorType, workspaceTensor, loc);
        op->getResult(0).replaceAllUsesWith(expand);
      } else {
        op->getResult(0).replaceAllUsesWith(workspaceTensor);
      }
      offsets = opBuilder.create<arith::AddIOp>(loc, offsets.getType(), offsets,
                                                totalSize.value());
    }

    // record workspace shape compute implements
    recordOpsForInferWorkspaceShapeFunc(offsets, recordOps);

    return workSpaceFuncArg;
  }

  func::FuncOp
  generateInferShapeWorkspaceFuncImpl(func::FuncOp parFunc,
                                      SmallVector<Operation *, 4> &recordOps,
                                      const std::string &funcName) {
    OpBuilder curBuilder(parFunc.getContext());
    OpBuilder::InsertionGuard insGuard(curBuilder);
    curBuilder.setInsertionPoint(parFunc);
    Value retValue = recordOps.back()->getResult(0);

    // Create function prototype
    FunctionType funcTy =
        FunctionType::get(parFunc.getContext(),
                          /*inputs=*/parFunc.getFunctionType().getInputs(),
                          /*results=*/
                          TypeRange(ValueRange(retValue)));
    auto newFunc = curBuilder.create<func::FuncOp>(parFunc.getLoc(),
                                                   /*name=*/funcName,
                                                   /*type=*/funcTy);
    // Create function body
    Block *entryBB = newFunc.addEntryBlock();
    curBuilder.setInsertionPointToStart(entryBB);

    // Clone operations and replace usages
    IRMapping curMap;
    for (auto [oldIn, newIn] :
         llvm::zip(parFunc.getArguments(), entryBB->getArguments())) {
      curMap.map(oldIn, newIn);
    }

    SetVector<Operation *> newOps;
    for (Operation *op : recordOps)
      newOps.insert(curBuilder.clone(*op, curMap));

    SetVector<Value> outs;
    assert(curMap.getValueMap().contains(retValue));
    outs.insert(curMap.getValueMap().at(retValue));

    curBuilder.create<func::ReturnOp>(parFunc->getLoc(),
                                      ValueRange(outs.getArrayRef()));
    eraseTriviallyDeadOps(newOps.getArrayRef());
    return newFunc;
  }

  void addAttrsForInferWorkspaceFunc(func::FuncOp inferShapeFunc) {
    hacc::utils::setHost(inferShapeFunc);
    hacc::utils::setHostFuncType(
        inferShapeFunc, hacc::HostFuncType::kInferWorkspaceShapeFunction);
  }

  func::FuncOp
  generateInferShapeWorkspaceFunc(func::FuncOp parFunc,
                                  SmallVector<Operation *, 4> &recordOps) {
    OpBuilder opBuilder(parFunc.getContext());
    // outline infershape_workspace_impl && set hacc.host attr
    std::string inferShapeFuncName = hacc::constructHostFunctionName(
        parFunc.getSymName().str(),
        hacc::HostFuncType::kInferWorkspaceShapeFunction);
    func::FuncOp inferShapeFunc = generateInferShapeWorkspaceFuncImpl(
        parFunc, recordOps, inferShapeFuncName);

    // set parFunc && infershape_workspace_func attr
    addAttrsForInferWorkspaceFunc(inferShapeFunc);
    parFunc->setAttr(hacc::InferWorkspaceShapeFunctionAttr::name,
                     hacc::InferWorkspaceShapeFunctionAttr::get(
                         opBuilder.getContext(), inferShapeFunc.getSymName()));
    return inferShapeFunc;
  }

  // infershape func don't contains linalg'ops && call op
  LogicalResult checkInferShapeFunc(func::FuncOp inferFunc) {
    auto res = inferFunc->walk([&](Operation *ops) {
      if (isa<linalg::LinalgOp>(ops) || isa<func::CallOp>(ops)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res == WalkResult::interrupt()) {
      return failure();
    }
    return success();
  }

  SmallVector<Value> getOpValueWithIdxs(Operation *op, BitVector &keepedIdxs) {
    SmallVector<Value> values;
    assert(op->getNumOperands() <= keepedIdxs.size());
    for (size_t i = 0; i < op->getNumOperands(); i++) {
      if (keepedIdxs[i]) {
        values.push_back(op->getOperand(i));
      }
    }
    return values;
  }

  void fixCallSites(func::FuncOp funcOp, func::FuncOp inferShapeFuncOp,
                    Value workspaceArg) {
    if (workspaceArg == nullptr) {
      return;
    }
    // erase infershape func unused argument
    OpBuilder opBuilder(inferShapeFuncOp);
    inferShapeFuncOp.setAllArgAttrs(funcOp.getAllArgAttrs());
    auto argTypeAttr = opBuilder.getStringAttr(hacc::KernelArgTypeAttr::name);
    auto inputAttr = hacc::KernelArgTypeAttr::get(opBuilder.getContext(),
                                                  hacc::KernelArgType::kInput);
    NamedAttribute namedAttr(argTypeAttr, inputAttr);
    BitVector keepdIdxs =
        hfusion::eraseFuncArgsExceptAttr(inferShapeFuncOp, namedAttr);
    // fix call site
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> callOps = getCallOps(module, funcOp.getSymName());
    MemRefType argumentType = cast<MemRefType>(workspaceArg.getType());
    auto shapes = argumentType.getShape();
    assert(shapes.size() == 1);
    for (auto callOp : callOps) {
      OpBuilder opBuilderCallOp(callOp);
      memref::AllocOp newAllocOp;
      if (ShapedType::isDynamicShape(shapes)) {
        SmallVector<Value> keepOperands = getOpValueWithIdxs(callOp, keepdIdxs);
        auto callInferShape = opBuilderCallOp.create<func::CallOp>(
            callOp.getLoc(), inferShapeFuncOp, keepOperands);
        auto workspaceSize = callInferShape->getResult(0);
        newAllocOp = opBuilderCallOp.create<memref::AllocOp>(
            callOp->getLoc(), argumentType, SmallVector<Value>{workspaceSize});
      } else {
        newAllocOp = opBuilderCallOp.create<memref::AllocOp>(callOp->getLoc(),
                                                             argumentType);
      }
      callOp->insertOperands(callOp->getNumOperands(), ValueRange{newAllocOp});
    }
  }

  LogicalResult handleFunc(func::FuncOp funcOp) {
    // apply distribute tensor empty: solve cse conflict
    if (failed(applyDistributeTensorEmpty(funcOp))) {
      return funcOp->emitError("Failed to distributed tensor.empty.");
    }
    // merge to workspace
    SmallVector<Operation *, 4> opsToErase;
    SmallVector<Operation *, 4> recordInferShapeImpl;
    if (failed(addToEmptyList(funcOp, opsToErase))) {
      return failure();
    }
    if (opsToErase.empty()) {
      return success();
    }
    Value workspaceArg =
        mergeToWorkspace(funcOp, opsToErase, recordInferShapeImpl);
    // generate infershape workspace func && check
    func::FuncOp inferShapeFunc =
        generateInferShapeWorkspaceFunc(funcOp, recordInferShapeImpl);
    if (failed(checkInferShapeFunc(inferShapeFunc))) {
      return inferShapeFunc->emitError(
          "infer shape func contains don't supported ops");
    }
    // fix call site
    fixCallSites(funcOp, inferShapeFunc, workspaceArg);
    // erase unused op
    for (Operation *op : opsToErase) {
      assert(
          op->use_empty() &&
          "there are still users of tensor empty and so it cannot be erased");
      op->erase();
    }
    return success();
  }
};

void HoistTensorEmptyPass::runOnOperation() {
  auto module = getOperation();
  auto res = module.walk([&](func::FuncOp funcOp) {
    if (!hacc::utils::isDeviceEntry(funcOp)) {
      return WalkResult::skip();
    }
    auto fusionKindAttr = funcOp->getAttrOfType<hfusion::FusionKindAttr>(
        hfusion::FusionKindAttr::name);
    if (!fusionKindAttr) {
      return WalkResult::skip();
    }
    if (fusionKindAttr.getFusionKind() != hfusion::FusionKind::ShallowCV &&
        fusionKindAttr.getFusionKind() != hfusion::FusionKind::ShallowVV &&
        fusionKindAttr.getFusionKind() != hfusion::FusionKind::MixCV) {
      return WalkResult::skip();
    }
    if (failed(handleFunc(funcOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res == WalkResult::interrupt()) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hfusion::createHoistTensorEmptyPass() {
  return std::make_unique<HoistTensorEmptyPass>();
}
