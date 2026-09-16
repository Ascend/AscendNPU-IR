//===- LowerMesh.cpp - Lower Mesh dialect collectives ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "bishengir/Conversion/LowerMesh/LowerMesh.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Mesh/Util.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERMESH
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace impl;

namespace {
// From hccl_types.h:
enum DataType : uint64_t {
  HCCL_DATA_TYPE_INT8 = 0,
  HCCL_DATA_TYPE_INT16 = 1,
  HCCL_DATA_TYPE_INT32 = 2,
  HCCL_DATA_TYPE_FP16 = 3,
  HCCL_DATA_TYPE_FP32 = 4,
  HCCL_DATA_TYPE_INT64 = 5,
  HCCL_DATA_TYPE_UINT64 = 6,
  HCCL_DATA_TYPE_UINT8 = 7,
  HCCL_DATA_TYPE_UINT16 = 8,
  HCCL_DATA_TYPE_UINT32 = 9,
  HCCL_DATA_TYPE_FP64 = 10,
  HCCL_DATA_TYPE_BFP16 = 11,
  HCCL_DATA_TYPE_INT128 = 12
};
enum ReduceType : uint64_t {
  HCCL_REDUCE_SUM = 0,
  HCCL_REDUCE_PROD = 1,
  HCCL_REDUCE_MAX = 2,
  HCCL_REDUCE_MIN = 3,
  HCCL_REDUCE_UNDEF = 1024
};

const char *targetLib;
DenseMap<OperationName, FunctionType> collectiveSignatureMap;

class MeshLoweringPass : public LowerMeshBase<MeshLoweringPass> {
public:
  explicit MeshLoweringPass(const LowerMeshOptions &options) : Base(options) {}

  void runOnOperation() override;
};

template <typename CollectiveOp>
struct LowerMesh : public OpRewritePattern<CollectiveOp> {
  using OpRewritePattern<CollectiveOp>::OpRewritePattern;

  MemRefType createGMMemref(ArrayRef<int64_t> sizes, Type elTy) const {
    return MemRefType::get(
        sizes, elTy, MemRefLayoutAttrInterface{},
        hivm::AddressSpaceAttr::get(elTy.getContext(), hivm::AddressSpace::GM));
  }

  Value getReduceType(mesh::ReductionKind reductionKind,
                      PatternRewriter &rewriter, Location loc) const {
    uint64_t typeEnum;
    switch (reductionKind) {
    case mesh::ReductionKind::Sum:
      typeEnum = HCCL_REDUCE_SUM;
      break;
    case mesh::ReductionKind::Product:
      typeEnum = HCCL_REDUCE_PROD;
      break;
    case mesh::ReductionKind::Max:
      typeEnum = HCCL_REDUCE_MAX;
      break;
    case mesh::ReductionKind::Min:
      typeEnum = HCCL_REDUCE_MIN;
      break;
    default:
      return nullptr;
    }
    return rewriter.create<arith::ConstantIntOp>(
        loc, typeEnum, /*width=*/64); // constant width is 64bit
  }

  LogicalResult matchAndRewrite(CollectiveOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto tensorTy = cast<TensorType>(op.getResult().getType());
    // No dynamic shape support as of now
    if (tensorTy.getNumDynamicDims())
      return failure();
    LLVMTypeConverter converter(op->getContext());
    // Since this is host function, we want to ignore the gm qualifier.
    converter.addTypeAttributeConversion(
        [](BaseMemRefType type, hivm::AddressSpaceAttr attr) {
          return Attribute();
        });

    // First get the storage of the result, empty tensor -> memref
    auto resultStorage = rewriter.create<tensor::EmptyOp>(
        loc, tensorTy, /*dynamicDims*/ ValueRange{});
    Type elementTy = tensorTy.getElementType();
    auto resultTy = createGMMemref(tensorTy.getShape(), elementTy);
    auto resultMem = rewriter.create<bufferization::ToMemrefOp>(
        loc, resultTy, resultStorage.getResult());

    // Then get the send buffer
    auto input = op.getInput();
    TensorType inputTensor = input.getType();
    auto inputTy = createGMMemref(inputTensor.getShape(), elementTy);
    auto inputMem =
        rewriter.create<bufferization::ToMemrefOp>(loc, inputTy, input);

    // Get the pointer and size from the memrefs
    auto srcDesc = MemRefDescriptor(
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, converter.convertType(inputTy), inputMem.getResult())
            .getResult(0));
    auto resultDesc = MemRefDescriptor(
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, converter.convertType(resultTy), resultMem.getResult())
            .getResult(0));

    // Get the base pointer and sizes to pass to the collective helper
    Value srcPtr = srcDesc.alignedPtr(rewriter, loc);
    Value srcSize = srcDesc.size(rewriter, loc, 0);
    for (int64_t i = 1; i < inputTy.getRank(); ++i) {
      srcSize = rewriter.create<arith::MulIOp>(
          loc, srcDesc.size(rewriter, loc, i), srcSize);
    }
    Value resultPtr = resultDesc.alignedPtr(rewriter, loc);
    Value resultSize = resultDesc.size(rewriter, loc, 0);
    for (int64_t i = 1; i < inputTy.getRank(); ++i) {
      resultSize = rewriter.create<arith::MulIOp>(
          loc, resultDesc.size(rewriter, loc, i), resultSize);
    }

    // Get the element type
    uint64_t typeEnum;

    if (elementTy.isSignedInteger()) {
      switch (elementTy.getIntOrFloatBitWidth()) {
      case 8: // HCCL data type of 8 bits
        typeEnum = HCCL_DATA_TYPE_INT8;
        break;
      case 16: // HCCL data type of 16 bits
        typeEnum = HCCL_DATA_TYPE_INT16;
        break;
      case 32: // HCCL data type of 32 bits
        typeEnum = HCCL_DATA_TYPE_INT32;
        break;
      case 64: // HCCL data type of 64 bits
        typeEnum = HCCL_DATA_TYPE_INT64;
        break;
      case 128: // HCCL data type of 128 bits
        typeEnum = HCCL_DATA_TYPE_INT128;
        break;
      default:
        llvm::report_fatal_error("Unsupported integer type for HCCL",
                                 /*gen_crash_diag*/ false);
      }
    } else if (elementTy.isSignlessInteger()) {
      switch (elementTy.getIntOrFloatBitWidth()) {
      case 8: // HCCL data type of 8 bits
        typeEnum = HCCL_DATA_TYPE_UINT8;
        break;
      case 16: // HCCL data type of 16 bits
        typeEnum = HCCL_DATA_TYPE_UINT16;
        break;
      case 32: // HCCL data type of 32 bits
        typeEnum = HCCL_DATA_TYPE_UINT32;
        break;
      case 64: // HCCL data type of 64 bits
        typeEnum = HCCL_DATA_TYPE_UINT64;
        break;
      default:
        llvm_unreachable("Unsupported unsigned type");
      }
    } else if (elementTy.isF16())
      typeEnum = HCCL_DATA_TYPE_FP16;
    else if (elementTy.isF32())
      typeEnum = HCCL_DATA_TYPE_FP32;
    else if (elementTy.isF64())
      typeEnum = HCCL_DATA_TYPE_FP64;
    else if (elementTy.isBF16())
      typeEnum = HCCL_DATA_TYPE_BFP16;
    else
      llvm_unreachable("Unsupported type for CCL operation");

    op->getName();
    Value typeVal =
        rewriter.create<arith::ConstantIntOp>(loc, typeEnum, /*width=*/64);

    SmallVector<Type, 6> argTys = {resultPtr.getType(), resultSize.getType(),
                                   srcPtr.getType(), srcSize.getType(),
                                   typeVal.getType()};
    SmallVector<Value, 6> argVals = {resultPtr, resultSize, srcPtr, srcSize,
                                     typeVal};

    // Currently collectives does not care about mesh dimensions, it will
    // be performed on all ranks
    std::string funcName = "_mlir_";
    funcName += targetLib;

    Operation *operation = op.getOperation();
    if (isa<mesh::AllGatherOp>(operation))
      funcName += "allgather";
    else if (auto allReduce = dyn_cast<mesh::AllReduceOp>(operation)) {
      Value reduceTypeVal =
          getReduceType(allReduce.getReduction(), rewriter, loc);
      if (!reduceTypeVal) {
        operation->emitOpError("Unsupported reduce kind");
        return failure();
      }
      argVals.push_back(reduceTypeVal);
      argTys.push_back(reduceTypeVal.getType());
      funcName += "allreduce";
    } else if (auto reduceScatter =
                   dyn_cast<mesh::ReduceScatterOp>(operation)) {
      Value reduceTypeVal =
          getReduceType(reduceScatter.getReduction(), rewriter, loc);
      if (!reduceTypeVal) {
        operation->emitOpError("Unsupported reduce kind");
        return failure();
      }
      argVals.push_back(reduceTypeVal);
      argTys.push_back(reduceTypeVal.getType());
      funcName += "reducescatter";
    } else if (isa<mesh::AllToAllOp>(operation))
      funcName += "all2all";
    else {
      op->emitOpError("Currently unsupported collective");
      return failure();
    }

    // Create call to custom function to perform allgather
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    func::FuncOp func =
        bishengir::getCustomFunction(funcName, moduleOp, loc, rewriter, argTys);
    rewriter.create<func::CallOp>(loc, func, argVals);

    // Finally convert result back into tensor
    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
        op, resultMem.getResult(), /*restrict*/ true);
    return success();
  }
};

/// Essentially DCE the mesh op if nothing is using it for now. Will need to
/// be a function to initialize a mesh topology later.
struct MeshCleanup : public OpRewritePattern<mesh::MeshOp> {
  using OpRewritePattern<mesh::MeshOp>::OpRewritePattern;

  LogicalResult match(mesh::MeshOp op) const final {
    auto uses =
        SymbolTable::getSymbolUses(op.getSymNameAttr(), op->getParentOp());
    if (!uses.has_value() || uses->empty())
      return success();

    return failure();
  }

  void rewrite(mesh::MeshOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};
} // namespace

void MeshLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  targetLib = this->target.getValue() == bishengir::mesh::TargetLib::HCCL
                  ? "hccl_"
                  : "lccl_";
  patterns.add<LowerMesh<mesh::AllGatherOp>, LowerMesh<mesh::AllReduceOp>,
               LowerMesh<mesh::ReduceScatterOp>, LowerMesh<mesh::AllToAllOp>,
               MeshCleanup>(&getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass>
mlir::createMeshLoweringPass(const LowerMeshOptions &options) {
  return std::make_unique<MeshLoweringPass>(options);
}
