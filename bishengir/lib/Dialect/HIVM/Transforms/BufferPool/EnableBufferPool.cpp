//===- EnableBufferPool.cpp - Independent load submission and slot binding ===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Transforms/BufferPoolUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_ENABLEBUFFERPOOL
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::buffer_pool;

namespace {
struct Stream {
  LoadInfo load;
  TaskInfo task;
  size_t pool = 0;
  int64_t stage = 1;
  int64_t delay = 0;
};

struct SlotPool {
  SmallVector<size_t> members;
  int64_t size = 1;
  size_t issuedState;
  size_t freeState;
  size_t consumedState;
};

struct LoopPlan {
  Operation *loop;
  Block *body;
  SmallVector<Stream, 4> streams;
  SmallVector<SlotPool, 4> pools;
  DenseMap<Operation *, size_t> loadScopes;
  DenseMap<Operation *, SmallVector<size_t>> reads;
  DenseMap<Operation *, SmallVector<size_t>> releases;
  DenseMap<Operation *, SmallVector<size_t>> bindingStarts;
};

static bool canRematerialize(Value value, const LoadInfo &load,
                             DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  Operation *def = value.getDefiningOp();
  if (!def || !load.loop->isAncestor(def) || value == load.root)
    return true;
  if (load.scope->isAncestor(def))
    return true;
  if (def->getNumRegions() || !isMemoryEffectFree(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return canRematerialize(operand, load, visited);
  });
}

static LogicalResult analyzeLoop(LoopPlan &plan, scope::ScopeOp firstScope,
                                 Operation *firstTransfer) {
  auto loop = cast<LoopLikeOpInterface>(plan.loop);
  DenseMap<int64_t, size_t> ids;
  for (Operation &operation : plan.body->without_terminator()) {
    auto scope = dyn_cast<scope::ScopeOp>(&operation);
    if (!scope || !scope->hasAttr(LoadPreloadAttr::name))
      continue;
    Operation *transfer =
        scope == firstScope ? firstTransfer : findLoadTransfer(scope);
    if (!transfer)
      continue;
    auto load = analyzeLoad(scope, transfer, /*planned=*/true);
    if (failed(load))
      return failure();
    auto task = analyzeTask(*load);
    if (failed(task) || !loop.isDefinedOutsideOfLoop(task->tripStart) ||
        !loop.isDefinedOutsideOfLoop(task->tripBound))
      return scope.emitError("cannot recover a fixed independent-load task "
                             "range and counter update");
    auto stage = scope->getAttrOfType<IntegerAttr>(LoadPreloadAttr::name);
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(scope->getRegions(), captures);
    DenseSet<Value> visited;
    for (Value capture : captures)
      if (!canRematerialize(capture, *load, visited))
        return scope.emitError("load address depends on a non-rematerializable "
                               "value in its scheduling loop");
    int64_t size = load->originalSlots;
    size_t poolIndex = plan.pools.size();
    if (auto id = scope->getAttrOfType<IntegerAttr>(MemPoolIdAttr::name)) {
      // PlanMemory assigns one storage entry and address table to each pool.
      poolIndex = ids.try_emplace(id.getInt(), poolIndex).first->second;
    }
    if (poolIndex == plan.pools.size()) {
      SlotPool pool;
      pool.size = size;
      plan.pools.push_back(std::move(pool));
    }
    size_t index = plan.streams.size();
    plan.pools[poolIndex].members.push_back(index);
    plan.loadScopes[scope] = index;
    Stream stream;
    stream.load = std::move(*load);
    stream.task = *task;
    stream.pool = poolIndex;
    stream.stage = stage.getInt();
    plan.streams.push_back(std::move(stream));
  }
  for (auto [i, stream] : llvm::enumerate(plan.streams)) {
    scf::IfOp guard = stream.task.guard;
    Operation *first = nullptr;
    Operation *last = nullptr;
    for (Operation *reader : stream.load.readers) {
      if (guard) {
        if (!guard.getThenRegion().isAncestor(reader->getParentRegion()))
          return stream.load.scope.emitError(
              "all readers must belong to the task's consumption branch");
      }
      Operation *top = reader;
      Block *body = guard ? guard.thenBlock() : plan.body;
      while (top->getBlock() != body)
        top = top->getParentOp();
      if (!guard && isa<scf::IfOp>(top))
        return stream.load.scope.emitError(
            "conditional consumption requires an identifiable task guard");
      if (!first || top->isBeforeInBlock(first))
        first = top;
      if (!last || last->isBeforeInBlock(top))
        last = top;
      Value root = stream.load.root;
      if (llvm::none_of(reader->getOperands(), [root](Value value) {
            return getBufferRoot(value) == root;
          }))
        return reader->emitError("cannot bind a buffer forwarded through a "
                                 "control-flow result");
      plan.reads[reader].push_back(i);
    }
    plan.releases[last].push_back(i);
    plan.bindingStarts[first].push_back(i);
  }
  for (auto [i, pool] : llvm::enumerate(plan.pools)) {
    int64_t maxStage = 1;
    for (size_t member : pool.members)
      maxStage = std::max(maxStage, plan.streams[member].stage);
    for (size_t member : pool.members) {
      Stream &stream = plan.streams[member];
      stream.delay = maxStage - stream.stage;
      if (stream.delay && !stream.task.guard)
        return stream.load.scope.emitError(
            "staged pooled loads require the computation schedule to be "
            "materialized before EnableBufferPool");
    }
    pool.issuedState = plan.streams.size() + 3 * i;
    pool.freeState = pool.issuedState + 1;
    pool.consumedState = pool.issuedState + 2;
  }
  return success();
}

class EnableBufferPool {
public:
  explicit EnableBufferPool(LoopPlan &plan)
      : plan(plan), builder(plan.loop), loc(plan.loop->getLoc()) {}
  void run();

private:
  Value integer(int64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  }
  Value toIndex(Value value) {
    if (value.getType().isIndex())
      return value;
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                              value);
  }
  Value add(Value lhs, Value rhs) {
    return builder.create<arith::AddIOp>(loc, lhs, rhs);
  }
  Value subtract(Value lhs, Value rhs) {
    return builder.create<arith::SubIOp>(loc, lhs, rhs);
  }
  Value bit(Value slot) {
    return builder.create<arith::ShLIOp>(loc, integer(1), slot);
  }
  Value compare(arith::CmpIPredicate predicate, Value lhs, Value rhs) {
    return builder.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
  }
  Value both(Value lhs, Value rhs) {
    return builder.create<arith::AndIOp>(loc, lhs, rhs);
  }
  bool precedes(size_t lhs, size_t rhs) const {
    int64_t a = plan.streams[lhs].stage;
    int64_t b = plan.streams[rhs].stage;
    return a != b ? a > b : lhs < rhs;
  }
  Value rematerialize(Value value, const LoadInfo &load, IRMapping &local);
  Value bindView(Value value, DenseMap<Value, Value> &roots, IRMapping &local);
  void cloneWithBindings(Operation &operation);
  void emitLoad(size_t stream, SmallVectorImpl<Value> &state);
  void release(Operation *operation, SmallVectorImpl<Value> &state);
  bool hasReleases(Operation *operation);
  void rewriteBlock(Block &block, SmallVectorImpl<Value> &state);
  void rewriteIf(scf::IfOp op, SmallVectorImpl<Value> &state);
  void rewriteScope(scope::ScopeOp op, SmallVectorImpl<Value> &state);
  SmallVector<Value> rewriteWhile(ValueRange initial);

  LoopPlan &plan;
  OpBuilder builder;
  Location loc;
  IRMapping mapping;
  SmallVector<Value> totals;
  SmallVector<Value> computeBindings;
};

Value EnableBufferPool::rematerialize(Value value, const LoadInfo &load,
                                      IRMapping &local) {
  if (Value mapped = local.lookupOrNull(value))
    return mapped;
  Operation *def = value.getDefiningOp();
  if (!def || !load.loop->isAncestor(def))
    return mapping.lookupOrDefault(value);
  // Captures were validated before any rewriting. Recompute pure address
  // expressions with the Load's own counter, including views CSE'd out of
  // scope.
  for (Value operand : def->getOperands())
    local.map(operand, rematerialize(operand, load, local));
  builder.clone(*def, local);
  return local.lookup(value);
}

Value EnableBufferPool::bindView(Value value, DenseMap<Value, Value> &roots,
                                 IRMapping &local) {
  if (auto it = roots.find(value); it != roots.end())
    return it->second;
  if (!roots.contains(getBufferRoot(value)))
    return mapping.lookupOrDefault(value);
  if (Value cached = local.lookupOrNull(value))
    return cached;
  Operation *view = value.getDefiningOp();
  assert(view && "validated root/view chain");
  IRMapping operands;
  for (Value operand : view->getOperands())
    operands.map(operand, bindView(operand, roots, local));
  Operation *copy = builder.clone(*view, operands);
  local.map(view->getResults(), copy->getResults());
  return local.lookup(value);
}

void EnableBufferPool::cloneWithBindings(Operation &operation) {
  Operation *copy = builder.clone(operation, mapping);
  for (auto &[reader, streams] : plan.reads) {
    if (reader != &operation && !operation.isAncestor(reader))
      continue;
    Operation *newReader = mapping.lookup(reader);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(newReader);
    DenseMap<Value, Value> roots;
    for (size_t stream : streams) {
      Value root = plan.streams[stream].load.root;
      roots[root] = computeBindings[stream];
    }
    IRMapping views;
    for (auto [i, operand] : llvm::enumerate(reader->getOperands()))
      newReader->setOperand(i, bindView(operand, roots, views));
  }
  if (auto root = dyn_cast<PointerCastOp>(copy))
    for (const Stream &stream : plan.streams)
      if (stream.load.root.getDefiningOp() == &operation)
        root->setAttr(ExperimentalBufferSlotsAttr::name, builder.getUnitAttr());
}

void EnableBufferPool::emitLoad(size_t stream, SmallVectorImpl<Value> &state) {
  Stream &current = plan.streams[stream];
  SlotPool &pool = plan.pools[current.pool];
  Value ordinal = state[stream];
  Value condition = compare(arith::CmpIPredicate::slt, ordinal, totals[stream]);
  Value key = add(ordinal, integer(current.delay));
  for (size_t other : pool.members) {
    if (other == stream)
      continue;
    const Stream &otherStream = plan.streams[other];
    Value otherKey = add(state[other], integer(otherStream.delay));
    Value ordered = compare(precedes(stream, other) ? arith::CmpIPredicate::sle
                                                    : arith::CmpIPredicate::slt,
                            key, otherKey);
    // For equal task ranges and no greater delay, other.done implies ordered
    // whenever the current task is in range.
    bool sameRange = current.task.tripStart == otherStream.task.tripStart &&
                     current.task.tripBound == otherStream.task.tripBound &&
                     current.task.tripStep == otherStream.task.tripStep;
    if (!sameRange || current.delay > otherStream.delay) {
      Value done =
          compare(arith::CmpIPredicate::sge, state[other], totals[other]);
      ordered = builder.create<arith::OrIOp>(loc, done, ordered);
    }
    condition = both(condition, ordered);
  }
  Value issued = state[pool.issuedState];
  Value slot = builder.create<arith::RemUIOp>(loc, issued, integer(pool.size));
  Value flag = bit(slot);
  Value available = both(state[pool.freeState], flag);
  condition =
      both(condition, compare(arith::CmpIPredicate::ne, available, integer(0)));
  SmallVector<Value> previous{ordinal, issued, state[pool.freeState]};
  auto issue = builder.create<scf::IfOp>(loc, TypeRange(ValueRange(previous)),
                                         condition, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(issue.thenBlock());
  IRMapping local;
  Value source = mapping.lookupOrDefault(current.load.root);
  Value buffer =
      builder.create<PoolSlotOp>(loc, source.getType(), source, slot);
  local.map(current.load.root, buffer);
  Value nextCounter = add(
      toIndex(mapping.lookupOrDefault(current.task.start)),
      builder.create<arith::MulIOp>(loc, ordinal, integer(current.task.step)));
  Type counterType = current.task.counter.getType();
  if (!counterType.isIndex())
    nextCounter =
        builder.create<arith::IndexCastOp>(loc, counterType, nextCounter);
  local.map(current.task.counter, nextCounter);
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(current.load.scope->getRegions(), captures);
  for (Value capture : captures)
    local.map(capture, rematerialize(capture, current.load, local));
  for (Operation &op : current.load.scope.getBody()->without_terminator())
    builder.clone(op, local);
  auto ready = builder.create<annotation::MarkOp>(loc, buffer);
  ready->setAttr(ExperimentalPoolReadyAttr::name, builder.getUnitAttr());
  Value newFree =
      builder.create<arith::XOrIOp>(loc, state[pool.freeState], flag);
  builder.create<scf::YieldOp>(
      loc,
      ValueRange{add(ordinal, integer(1)), add(issued, integer(1)), newFree});
  builder.setInsertionPointToStart(issue.elseBlock());
  builder.create<scf::YieldOp>(loc, previous);
  builder.setInsertionPointAfter(issue);
  state[stream] = issue.getResult(0);
  state[pool.issuedState] = issue.getResult(1);
  state[pool.freeState] = issue.getResult(2);
}

void EnableBufferPool::release(Operation *operation,
                               SmallVectorImpl<Value> &state) {
  auto it = plan.releases.find(operation);
  if (it == plan.releases.end())
    return;
  for (size_t stream : it->second) {
    size_t index = plan.pools[plan.streams[stream].pool].freeState;
    Value binding = computeBindings[stream];
    auto release = builder.create<annotation::MarkOp>(loc, binding);
    release->setAttr(ExperimentalPoolReleaseAttr::name, builder.getUnitAttr());
    Value slot = binding.getDefiningOp<PoolSlotOp>().getSlot();
    state[index] = builder.create<arith::OrIOp>(loc, state[index], bit(slot));
  }
}

bool EnableBufferPool::hasReleases(Operation *operation) {
  return llvm::any_of(plan.releases, [&](const auto &entry) {
    return operation->isAncestor(entry.first);
  });
}

void EnableBufferPool::rewriteIf(scf::IfOp op, SmallVectorImpl<Value> &state) {
  SmallVector<Type> types(op.getResultTypes());
  llvm::append_range(types, ValueRange(state).getTypes());
  auto next = builder.create<scf::IfOp>(
      op.getLoc(), types, mapping.lookupOrDefault(op.getCondition()), true);
  next->setAttrs(op->getAttrs());
  auto rewriteRegion = [&](Block *old, Block *block) {
    builder.setInsertionPointToStart(block);
    SmallVector<Value> branchState(state.begin(), state.end());
    SmallVector<Value> yields;
    if (old) {
      rewriteBlock(*old, branchState);
      for (Value value : old->getTerminator()->getOperands())
        yields.push_back(mapping.lookupOrDefault(value));
    }
    llvm::append_range(yields, branchState);
    builder.create<scf::YieldOp>(loc, yields);
  };
  rewriteRegion(op.thenBlock(), next.thenBlock());
  rewriteRegion(op.getElseRegion().empty() ? nullptr : op.elseBlock(),
                next.elseBlock());
  mapping.map(op.getResults(),
              next.getResults().take_front(op.getNumResults()));
  auto results = next.getResults().drop_front(op.getNumResults());
  state.assign(results.begin(), results.end());
  builder.setInsertionPointAfter(next);
}

void EnableBufferPool::rewriteScope(scope::ScopeOp op,
                                    SmallVectorImpl<Value> &state) {
  SmallVector<Type> types(op.getResultTypes());
  llvm::append_range(types, ValueRange(state).getTypes());
  auto next = builder.create<scope::ScopeOp>(op.getLoc(), types);
  next->setAttrs(op->getAttrs());
  Block *block = builder.createBlock(&next.getRegion());
  builder.setInsertionPointToStart(block);
  rewriteBlock(*op.getBody(), state);
  SmallVector<Value> yields;
  for (Value value : op.getBody()->getTerminator()->getOperands())
    yields.push_back(mapping.lookupOrDefault(value));
  llvm::append_range(yields, state);
  builder.create<scope::ReturnOp>(loc, yields);
  mapping.map(op.getResults(),
              next.getResults().take_front(op.getNumResults()));
  auto results = next.getResults().drop_front(op.getNumResults());
  state.assign(results.begin(), results.end());
  builder.setInsertionPointAfter(next);
}

void EnableBufferPool::rewriteBlock(Block &block,
                                    SmallVectorImpl<Value> &state) {
  for (Operation &op : block.without_terminator()) {
    if (auto it = plan.bindingStarts.find(&op);
        it != plan.bindingStarts.end()) {
      for (size_t stream : it->second) {
        const SlotPool &pool = plan.pools[plan.streams[stream].pool];
        Value root = mapping.lookupOrDefault(plan.streams[stream].load.root);
        Value slot = builder.create<arith::RemUIOp>(
            loc, state[pool.consumedState], integer(pool.size));
        computeBindings[stream] =
            builder.create<PoolSlotOp>(loc, root.getType(), root, slot);
        // The upstream schedule consumes tasks in issue order. Reserve one
        // slot per input, including multiple inputs of the same computation.
        state[pool.consumedState] = add(state[pool.consumedState], integer(1));
      }
    }
    if (auto it = plan.loadScopes.find(&op); it != plan.loadScopes.end()) {
      emitLoad(it->second, state);
    } else if (auto guard = dyn_cast<scf::IfOp>(&op);
               guard && hasReleases(&op)) {
      rewriteIf(guard, state);
    } else if (auto scope = dyn_cast<scope::ScopeOp>(&op);
               scope && hasReleases(&op)) {
      rewriteScope(scope, state);
    } else {
      cloneWithBindings(op);
    }
    release(&op, state);
  }
  release(block.getTerminator(), state);
}

// Append state to both while channels: inits/before/yield and
// results/after/condition. The scheduling region updates it; the other region
// forwards it unchanged, including the final before visit on loop exit.
SmallVector<Value> EnableBufferPool::rewriteWhile(ValueRange initial) {
  auto loop = cast<scf::WhileOp>(plan.loop);
  bool inBefore = plan.body == &loop.getBefore().front();
  unsigned originalCount = loop.getNumResults();
  SmallVector<Value> inits(loop.getInits());
  llvm::append_range(inits, initial);
  SmallVector<Type> resultTypes(loop.getResultTypes());
  llvm::append_range(resultTypes, initial.getTypes());
  auto next = builder.create<scf::WhileOp>(loc, resultTypes, inits);
  next->setAttrs(loop->getAttrs());
  next.getBefore().getBlocks().clear();
  next.getAfter().getBlocks().clear();

  Block *old = plan.body;
  Region &region = inBefore ? next.getBefore() : next.getAfter();
  SmallVector<Type> types(old->getArgumentTypes());
  llvm::append_range(types, initial.getTypes());
  SmallVector<Location> locations(types.size(), loc);
  Block *body = builder.createBlock(&region, region.end(), types, locations);
  mapping.map(old->getArguments(),
              body->getArguments().take_front(old->getNumArguments()));
  SmallVector<Value> state(body->getArguments().take_back(initial.size()));
  rewriteBlock(*old, state);
  SmallVector<Value> operands;
  for (Value value : old->getTerminator()->getOperands())
    operands.push_back(mapping.lookupOrDefault(value));
  llvm::append_range(operands, state);
  if (inBefore)
    builder.create<scf::ConditionOp>(loc, operands.front(),
                                     ValueRange(operands).drop_front());
  else
    builder.create<scf::YieldOp>(loc, operands);

  // Move the other region so any pending plan for it keeps valid block,
  // operation and value references. Its original channel can have a different
  // arity or types from the scheduling region.
  Region &other = inBefore ? next.getAfter() : next.getBefore();
  other.takeBody(inBefore ? loop.getAfter() : loop.getBefore());
  SmallVector<Value> forwarded;
  for (Type type : initial.getTypes())
    forwarded.push_back(other.front().addArgument(type, loc));
  Operation *terminator = other.front().getTerminator();
  terminator->insertOperands(terminator->getNumOperands(), forwarded);

  loop.replaceAllUsesWith(next.getResults().take_front(originalCount));
  loop.erase();
  plan.loop = next;
  plan.body = body;
  builder.setInsertionPointAfter(next);
  auto results = next.getResults().drop_front(originalCount);
  return SmallVector<Value>(results.begin(), results.end());
}

void EnableBufferPool::run() {
  computeBindings.resize(plan.streams.size());
  // Keep independent issue and consumption cursors across repeated entries
  // from enclosing loops. Within the scheduling loop both are carried in SSA;
  // task ordinals and the software free state belong to each entry.
  SmallVector<Value> cursors;
  auto function = plan.loop->getParentOfType<func::FuncOp>();
  auto cursorType = MemRefType::get({1}, builder.getI64Type());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&function.getBody().front());
    Value zero =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    Value index = integer(0);
    for (size_t i = 0; i < 2 * plan.pools.size(); ++i) {
      Value cursor = builder.create<memref::AllocaOp>(loc, cursorType);
      builder.create<memref::StoreOp>(loc, zero, cursor, index);
      cursors.push_back(cursor);
    }
  }
  SmallVector<Value> initial;
  for (const Stream &stream : plan.streams) {
    Value distance = subtract(toIndex(stream.task.tripBound),
                              toIndex(stream.task.tripStart));
    distance = builder.create<arith::MaxSIOp>(loc, distance, integer(0));
    totals.push_back(builder.create<arith::CeilDivSIOp>(
        loc, distance, integer(stream.task.tripStep)));
    initial.push_back(integer(0));
  }
  for (auto [i, pool] : llvm::enumerate(plan.pools)) {
    initial.push_back(toIndex(
        builder.create<memref::LoadOp>(loc, cursors[2 * i], integer(0))));
    initial.push_back(integer((int64_t(1) << pool.size) - 1));
    initial.push_back(toIndex(
        builder.create<memref::LoadOp>(loc, cursors[2 * i + 1], integer(0))));
  }
  SmallVector<Value> state;
  if (auto loop = dyn_cast<scf::ForOp>(plan.loop)) {
    SmallVector<Value> inits(loop.getInitArgs());
    llvm::append_range(inits, initial);
    auto next = builder.create<scf::ForOp>(
        loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), inits);
    next->setAttrs(loop->getAttrs());
    if (!next.getBody()->empty())
      next.getBody()->getTerminator()->erase();
    builder.setInsertionPointToStart(next.getBody());
    mapping.map(loop.getInductionVar(), next.getInductionVar());
    unsigned originalCount = loop.getNumRegionIterArgs();
    mapping.map(loop.getRegionIterArgs(),
                next.getRegionIterArgs().take_front(originalCount));
    SmallVector<Value> bodyState(
        next.getRegionIterArgs().drop_front(originalCount));
    rewriteBlock(*plan.body, bodyState);
    SmallVector<Value> yields;
    for (Value value : loop.getYieldedValues())
      yields.push_back(mapping.lookupOrDefault(value));
    llvm::append_range(yields, bodyState);
    builder.create<scf::YieldOp>(loc, yields);
    builder.setInsertionPointAfter(next);
    auto results = next.getResults().drop_front(originalCount);
    state.assign(results.begin(), results.end());
    loop.replaceAllUsesWith(next.getResults().take_front(originalCount));
  } else {
    state = rewriteWhile(initial);
  }
  for (auto [i, storage] : llvm::enumerate(cursors)) {
    const SlotPool &pool = plan.pools[i / 2];
    Value cursor = state[i % 2 ? pool.consumedState : pool.issuedState];
    cursor = builder.create<arith::RemUIOp>(loc, cursor, integer(pool.size));
    cursor =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), cursor);
    builder.create<memref::StoreOp>(loc, cursor, storage, integer(0));
  }
  if (isa<scf::ForOp>(plan.loop))
    plan.loop->erase();
}

static LogicalResult enableFunction(func::FuncOp function) {
  SmallVector<std::unique_ptr<LoopPlan>, 4> plans;
  DenseSet<Block *> seen;
  bool invalid = false;
  function.walk([&](scope::ScopeOp scope) {
    if (!scope->hasAttr(LoadPreloadAttr::name) ||
        seen.contains(scope->getBlock()))
      return;
    Operation *transfer = findLoadTransfer(scope);
    if (!transfer)
      return;
    auto loop = scope->getParentOfType<LoopLikeOpInterface>();
    if (getLoopBodyBlock(loop, scope) != scope->getBlock()) {
      scope.emitError(
          "independent load must be a direct child of its scheduling block");
      invalid = true;
      return;
    }
    seen.insert(scope->getBlock());
    auto plan = std::make_unique<LoopPlan>();
    plan->loop = loop.getOperation();
    plan->body = scope->getBlock();
    if (failed(analyzeLoop(*plan, scope, transfer))) {
      invalid = true;
      return;
    }
    plans.push_back(std::move(plan));
  });
  if (invalid)
    return failure();
  if (plans.empty())
    return success();
  for (auto &plan : plans) {
    // Rebuilding one while region may have replaced the enclosing operation
    // while preserving this region and all of its analyzed accesses.
    plan->loop = plan->body->getParentOp();
    for (Stream &stream : plan->streams)
      stream.load.loop = plan->loop;
    EnableBufferPool(*plan).run();
  }
  // Remove obsolete pure views; actual accesses now derive from PoolSlotOp.
  RewritePatternSet cleanup(function.getContext());
  return applyPatternsGreedily(function, std::move(cleanup));
}

struct EnableBufferPoolPass : impl::EnableBufferPoolBase<EnableBufferPoolPass> {
  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
      if (failed(enableFunction(function)))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createEnableBufferPoolPass() {
  return std::make_unique<EnableBufferPoolPass>();
}
