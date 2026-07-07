//===- MeshShardingInterfaceImpl.cpp - Impl. of MeshShardingInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Sharding Interface for the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Tensor/Transforms/MeshShardingInterfaceImpl.h"

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
template <typename OpT>
struct TensorShardingInterface : public mesh::ShardingInterface::ExternalModel<
                                     TensorShardingInterface<OpT>, OpT> {
  LogicalResult getResultTensorType(Operation *op,
                                    RankedTensorType &result) const {
    if (op->getNumResults() != 1)
      return failure();
    result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    return success(result);
  }

  /// Currently only handles fully parallel operations
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    RankedTensorType tensorTy;
    if (getResultTensorType(op, tensorTy).failed()) {
      op->emitError("Sharding interface expecting single result");
      return {};
    }

    SmallVector<utils::IteratorType> retVal(tensorTy.getRank(),
                                            utils::IteratorType::parallel);
    return retVal;
  }

  /// Currently only handles fully parallel operations
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    RankedTensorType tensorTy;
    if (getResultTensorType(op, tensorTy).failed()) {
      op->emitError("Sharding interface expecting single result");
      return {};
    }
    // TODO: Add operand maps to the list
    SmallVector<AffineMap> retVal;
    auto resultMap =
        AffineMap::getMultiDimIdentityMap(tensorTy.getRank(), op->getContext());
    retVal.push_back(resultMap);
    return retVal;
  }

  /// For use with mesh-spmdize, currently one works with fully parallel
  /// operations.
  LogicalResult spmdize(Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<mesh::MeshShardingAttr> operandShardings,
                        ArrayRef<mesh::MeshShardingAttr> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTable,
                        OpBuilder &builder) const {
    if (resultShardings.size() != 1)
      llvm_unreachable(
          "Tensor ops should not have more than one result sharding");
    Operation *newOp = builder.clone(*op, spmdizationMap);
    newOp->getResult(0).setType(
        shardType(newOp->getResult(0).getType(),
                  mesh::getMesh(op, resultShardings[0].getMesh(), symbolTable),
                  resultShardings[0]));
    return success();
  }
};
} // namespace

/// Only handles tensor.empty at the moment
void bishengir::tensor::registerMeshShardingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::tensor::TensorDialect *dialect) {
        DialectRegistry registry;
        registry.insert<mlir::tensor::TensorDialect>();
        ctx->appendDialectRegistry(registry);
        for (StringRef name : registry.getDialectNames()) {
          ctx->getOrLoadDialect(name);
        }
        mlir::tensor::EmptyOp::attachInterface<
            TensorShardingInterface<mlir::tensor::EmptyOp>>(*ctx);
      });
}
