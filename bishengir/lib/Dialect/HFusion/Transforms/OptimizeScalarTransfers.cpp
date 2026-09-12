//===- OptimizeScalarTransfers.cpp - Optimize scalar transfers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#define DEBUG_TYPE "hfusion-optimize-scalar-transfers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")



namespace mlir {
#define GEN_PASS_DEF_OPTIMIZESCALARTRANSFERS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

using Producers = DenseSet<tensor::ExtractOp>;
using Consumers = DenseSet<Operation *>;
using ProducedMarks = DenseSet<Operation *>;
using ConsumedMarks = DenseSet<Operation *>;
using TopSort = SetVector<Operation *>;
using ReplacementMap = DenseMap<Operation *, Value>;
using Worklist = llvm::SmallSetVector<Operation *, 16>;

const int maxTransformationSize = 400;

// returns true iff the given type is `tensor<1xelementType>
static bool isSingletonTensorType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         tensorType.getDimSize(0) == 1;
}

static bool isScalarToVectorTransfer(tensor::InsertOp insertOp) {
  return isSingletonTensorType(insertOp.getResult().getType());
}

// Returns true iff the scalar was extracted from tenos
// with type tensor<T> or tensor<1xT>
static bool isVectorToScalarTransfer(tensor::ExtractOp extractOp) {
  RankedTensorType type = extractOp.getTensor().getType();
  return type.getRank() == 0 || isSingletonTensorType(type);
}

static bool isConsumer(Operation *op) {
  if (auto insert = dyn_cast<tensor::InsertOp>(op)) {
    return isScalarToVectorTransfer(insert);
  }

  // Every linalg.fillOp is potential
  // vector<->scalar transfer
  if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      if (auto type = dyn_cast<RankedTensorType>(fillOp.output().getType())) {
        return type.getRank() == 1 &&
          !ShapedType::isDynamic(type.getDimSize(0));
      }
      return false;
  }

  return false;
}

static std::pair<Producers, Consumers>
findProducersAndConsumers(func::FuncOp func) {
  Consumers consumers;
  Producers producers;

  func.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (auto extract = dyn_cast<tensor::ExtractOp>(op)) {
        if (isVectorToScalarTransfer(extract)) {
          producers.insert(extract);
        }
      } else if (isConsumer(op)) {
          consumers.insert(op);
      }
  });

  LLVM_DEBUG(
    llvm::dbgs() << "Consumers set: \n";
    llvm::for_each(consumers, [&](Operation *op){
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    llvm::for_each(producers, [&](tensor::ExtractOp insertOp){
      insertOp.getOperation()->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  );
  return std::make_pair(producers, consumers);
}

static bool filterOps(mlir::Operation *op) {
  unsigned numOperands = op->getNumOperands();

  // These operations cant be pulled
  // into Vector context, since
  // the prescision between Vector
  // and Scalar unit differs
  if (isa<arith::DivFOp>(op) ||
      isa<arith::RemFOp>(op) ||
      isa<math::SqrtOp>(op))
    return false;

  if (isa<tensor::ExtractOp>(op) ||
      isa<tensor::InsertOp>(op))
    return true;

  if (isa<linalg::FillOp>(op))
    return true;

  if(!isPure(op))
    return false;

  if (numOperands != 1 && numOperands != 2)
    return false;

  if (op->getNumResults() != 1)
    return false;

  auto isIntOrFloat = [](Type type){
    return type.isIntOrFloat();
  };

  if (!llvm::all_of(op->getOperandTypes(), isIntOrFloat))
    return false;

  if (!isIntOrFloat(op->getResult(0).getType()))
    return false;

  return true;
}


// TODO: remove duplication with markProducedScalars
static ConsumedMarks markConsumedScalars(TopSort topSort, Consumers &consumers, Producers &producers) {
  LDBG("Markup of the consumed scalars started\n");

  ConsumedMarks marks;

  // Initial markup
  llvm::for_each(consumers, [&](Operation *op) {
      marks.insert(op);
  });


  auto isConsumed = [&](Operation *op) {
    bool isProducer = false;
    if (auto extractOp = dyn_cast_if_present<tensor::ExtractOp>(op)) {
      isProducer = producers.contains(extractOp);
    }
    return marks.contains(op) && !isProducer;
  };


  auto meet = [&](Operation *op) {
    // Iff the operation was already mark up (tensor::InsertOp)
    // or compute the mark by uses of the current Op
    return marks.contains(op) ||
      (!op->use_empty() && llvm::all_of(op->getUsers(),
        [&](Operation *user) {
          return isConsumed(user);
      }));
  };

  auto mark = [&](Operation *op) {
    marks.insert(op);
  };

  for (Operation *op : llvm::reverse(topSort)) {
    LDBG("Visiting operation: " << *op);
    LDBG("Meet: " << meet(op));
    LDBG("Filter: " << filterOps(op));
    if (meet(op) && filterOps(op))
      mark(op);
  }

  return marks;
}


static ProducedMarks markProducedScalars(TopSort topSort, Producers &producers, Consumers &consumers) {
  LDBG("Markup of the produced scalars started\n");
  ProducedMarks marks;

  // Initial markup
  llvm::for_each(producers, [&](tensor::ExtractOp extractOp) {
      marks.insert(extractOp);
  });

  auto isProduced = [&](Operation *op) {
    bool isConsumer = false;
    if (auto insertOp = dyn_cast_if_present<tensor::InsertOp>(op)) {
      isConsumer = consumers.contains(insertOp);
    }
    return marks.contains(op) && !isConsumer;
  };

  auto meet = [&](Operation *op) {
    return llvm::any_of(op->getOperands(), [&](Value operand) {
        if (Operation *def = operand.getDefiningOp()) {
          return isProduced(def);
        }
        // Currently BlockArgument considered as non-produced
        return false;
    });
  };

  auto mark = [&](Operation *op) {
    marks.insert(op);
  };

  for (Operation *op : topSort) {
    LDBG("Visition operation: " << *op);
    LDBG("Meet: " << meet(op));
    LDBG("Filter: " << filterOps(op));
    if (meet(op) && filterOps(op))
      mark(op);
  }

  return marks;
}

static SetVector<Operation *>
buildTransformationOrder(TopSort topSort, ProducedMarks &produced, ConsumedMarks &consumed) {
  SetVector<Operation *> order;
  for (Operation *op : topSort) {
    bool isTransfer =
      llvm::any_of(op->getOperands(),[&](Value operand){
        if (Operation *def = operand.getDefiningOp()) {
          return produced.contains(def) && consumed.contains(def);
        }
        return true;
      }) || isa<tensor::ExtractOp>(op);

    if (produced.contains(op) && consumed.contains(op)) {
      if (!isTransfer) {
        produced.erase(op);
        consumed.erase(op);
      } else {
        order.insert(op);
      }
    }
  }

  return order;
}



static void
liftExtractOp(tensor::ExtractOp extractOp, IRRewriter &rewriter,
              IRMapping &map,
              DenseSet<Operation *> &defferedRemove) {
  LDBG("Start lifting extractOp: " << *extractOp.getOperation());

  rewriter.setInsertionPoint(extractOp);
  auto tensor = extractOp.getTensor();
  auto tensorType = tensor.getType();
  auto loc = tensor.getLoc();

  if (isSingletonTensorType(tensorType)) {
    defferedRemove.insert(extractOp);
    map.map(extractOp.getResult(), tensor);
  } else {
    auto resultType = RankedTensorType::get({1},
      tensor.getType().getElementType());

    // Empty reassociation is needed to expand rank 0 -> rank 1
    SmallVector<ReassociationIndices> reassociation;
    auto expandShape = rewriter.create<tensor::ExpandShapeOp>(
        loc,
        resultType,
        tensor,
        reassociation
    );

    LDBG("The replacement for operation is: "
        << *expandShape.getOperation());
    map.map(extractOp.getResult(),
        expandShape.getResult());
    return;
  }
}

static void
liftInsertOp(tensor::InsertOp insertOp, IRRewriter &rewriter,
             IRMapping &map,
             DenseSet<Operation *> &defferedRemove) {
  LDBG("Convering insert to NoOP: "
      << *insertOp.getOperation());

  Value scalar = insertOp.getScalar();
  assert(map.contains(scalar)
      && "broken transformation order, the insert argument wasn't mapped");
  Value scalarReplacement = map.lookup(scalar);

  if (Operation *def = scalar.getDefiningOp()) {
    LDBG("The operation is scheduled for removal: " << *def);
    defferedRemove.insert(def);
  }

  rewriter.replaceOp(insertOp, scalarReplacement);
  return;
}

static void
liftFillOp(linalg::FillOp fillOp, IRRewriter &rewriter,
    IRMapping &map, DenseSet<Operation *> &defferedRemove) {
  LDBG("Started lifting : " << *fillOp);
  rewriter.setInsertionPointAfter(fillOp);
  Value fillVal = fillOp.value();
  assert(map.contains(fillVal)
      && "broken transformation order, the fill argument wasn't mapped");

  Value output = fillOp.output();
  assert(isa<RankedTensorType>(output.getType())
      && "fill over memref should not be marked as consumed");

  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  assert(outputType.getRank() == 1
      && "the init of linalg.fill should be first rank tensor");
  int64_t n = outputType.getDimSize(0);

  SmallVector<int64_t> newShape{n, 1};
  auto empty = rewriter.create<tensor::EmptyOp>(
      fillOp.getLoc(),
      newShape,
      outputType.getElementType()
  );

  auto broadcastOp = rewriter.create<mlir::linalg::BroadcastOp>(
      fillOp.getLoc(),
      map.lookup(fillVal),
      empty,
      SmallVector<int64_t>{0}
  );

  auto collapsedType = RankedTensorType::get(
      {n},
      outputType.getElementType()
  );

  SmallVector<ReassociationIndices> reassociation = {{0, 1}};
  Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
      fillOp.getLoc(),
      collapsedType,
      broadcastOp->getResult(0),
      reassociation
  );

  if (Operation *def = fillVal.getDefiningOp()) {
    defferedRemove.insert(def);
  }

  rewriter.replaceOp(fillOp, collapsed);
}

static Value
createTensorAdaptor(Value operand, IRRewriter &rewriter) {
  Type type = operand.getType();
  assert(type.isIntOrFloat());
  auto loc = operand.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(operand);

  RankedTensorType tensorType =
    RankedTensorType::get({1}, type);

  auto empty = rewriter.create<tensor::EmptyOp>(
    loc,
    tensorType.getShape(),
    type
  );

  Value zero =
    rewriter.create<arith::ConstantIndexOp>(loc, 0);

  auto insertOp = rewriter.create<tensor::InsertOp>(
      loc,
      operand,
      empty,
      ValueRange(zero)
  );

  return insertOp.getResult();
}

static Operation *
liftOp(Operation *op, IRRewriter &rewriter, IRMapping &map) {
  LDBG("Started lifting of: " << *op);
  Type type = op->getResult(0).getType();
  assert(type.isIntOrFloat());
  rewriter.setInsertionPointAfter(op);

  SmallVector<Value> operands;

  for (Value operand : op->getOperands()) {
    if (!map.contains(operand)) {
      Value adaptor =
        createTensorAdaptor(operand, rewriter);
      LDBG("Adaptor created: "
          << *adaptor.getDefiningOp());
      map.map(operand, adaptor);
    }
  }

  assert(llvm::all_of(op->getOperands(), [&](Value operand) {
    return map.contains(operand);
  }));


  Operation *clonedOp = rewriter.clone(*op, map);
  RankedTensorType tensorType =
    RankedTensorType::get({1}, type);
  SmallVector<Type, 1> newResultTypes{tensorType};

  rewriter.modifyOpInPlace(clonedOp, [&] {
    for (auto [result, newType] :
         llvm::zip_equal(clonedOp->getResults(), newResultTypes)) {
      result.setType(newType);
    }
  });

  for (auto [originalVal, newVal] :
      llvm::zip_equal(clonedOp->getResults(), op->getResults())) {
    map.map(originalVal, newVal);
  }

  LDBG("replacement created: " << *clonedOp);
  return clonedOp;
}




static void
eliminateProducerConsumerTransfers(TopSort topSort, SetVector<Operation *> ops) {
  IRMapping replacementMap;
  DenseSet<Operation *> defferedRemove;

  if (ops.empty())
    return;

  IRRewriter rewriter(ops.front()->getContext());

  for (Operation *op : ops) {
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
      liftExtractOp(extractOp, rewriter, replacementMap, defferedRemove);
    } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
      liftInsertOp(insertOp, rewriter, replacementMap, defferedRemove);
    } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
      liftFillOp(fillOp, rewriter, replacementMap, defferedRemove);
    } else {
      liftOp(op, rewriter, replacementMap);
    }
  }

  for (Operation *currentOp : llvm::reverse(topSort)) {
    if (!(defferedRemove.contains(currentOp) && ops.contains(currentOp)))
      continue;

    LDBG("started removal of: " << *currentOp);
    assert(currentOp->getUsers().empty() && "tries to remove operation with uses");

    llvm::for_each(currentOp->getOperands(),[&](Value user) {
        Operation *def = user.getDefiningOp();
        if (def && ops.contains(def)) {
          LDBG("Operation is scheduled for removal: " << *def);
          defferedRemove.insert(def);
        }
    });

    rewriter.eraseOp(currentOp);
  }
  return;
}

static bool isAIV(func::FuncOp funcOp) {
  if (auto mixMode = funcOp->getAttrOfType<StringAttr>("mix_mode")) {
    return mixMode.getValue() == "aiv";
  }
  return false;
}

static void
debugProducersAndConsumersMarkUp(ProducedMarks &producedMarks,
    ConsumedMarks &consumedMarks,
    SetVector<Operation *> hasBothMarks) {

  llvm::dbgs() << "Operation that marked as produced\n";
  for (Operation *op : producedMarks) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Operations that marked as consumed\n";
  for (Operation *op : consumedMarks) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Operation that have both marks\n";

  for (Operation *op : hasBothMarks) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
}

static void
optimizeScalarTransfers(func::FuncOp funcOp) {
  llvm::SetVector<mlir::Operation *> operations;
  funcOp.walk([&](Operation *op){
    if (funcOp.getOperation() != op) {
      operations.insert(op);
    }
  });

  TopSort topSort =
    mlir::topologicalSort(operations);

  auto [producers, consumers] = findProducersAndConsumers(funcOp);
  auto consumed = markConsumedScalars(topSort, consumers, producers);
  auto produced = markProducedScalars(topSort, producers, consumers);
  auto transformationOrder = buildTransformationOrder(topSort, produced, consumed);

  LLVM_DEBUG(
    debugProducersAndConsumersMarkUp(
      produced,
      consumed,
      transformationOrder
    )
  );

  // FIXME: the large number of subsequent vector
  // operations can cause an errors in the
  // AutoVectorization pipeline
  if (transformationOrder.size() > maxTransformationSize) {
    emitWarning(funcOp.getLoc(),
        "[hfusion-optimize-scalar-transfers]: "
        "cant transform large sequence "
        "of scalar operations\n");
    return;
  }
  eliminateProducerConsumerTransfers(topSort, transformationOrder);
}

struct OptimizeScalarTransfersPass
    : public impl::OptimizeScalarTransfersBase<OptimizeScalarTransfersPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!isAIV(funcOp)) {
      LDBG("The following func is not AIV, so pass is skipped"
          << funcOp.getSymName());
      return;
    }

    LDBG("started the transfers optimizations");
    optimizeScalarTransfers(funcOp);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createOptimizeScalarTransfersPass() {
  return std::make_unique<OptimizeScalarTransfersPass>();
}
