#include "bishengir/Conversion/ProtonToHIVM/ProtonToHIVM.h"

#include <memory>
#include <string>
#include <tuple>

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMProtonOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"

namespace bishengir::triton::proton::a5 {
#define GEN_PASS_DEF_CONVERTPROTONTOHIVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace bishengir::triton::proton::a5

using namespace mlir;

namespace bishengir::triton::proton::a5 {

namespace {

struct ActiveScope {
  std::uint32_t logicalId;
  std::string name;
};

inline bool operator==(const ActiveScope &lhs, const ActiveScope &rhs) {
  return std::tie(lhs.logicalId, lhs.name) == std::tie(rhs.logicalId, rhs.name);
}

class ProtonScopePlan {
  std::uint32_t scopeCount{0};
  DenseMap<Operation *, std::uint32_t> recordRegionIds;

public:
  static FailureOr<ProtonScopePlan> build(ModuleOp module);

  std::uint32_t getRegionId(mlir::triton::proton::RecordOp record) const;

private:
  LogicalResult analyzeRegion(Region &region,
                              ArrayRef<ActiveScope> inheritedScopes);
  LogicalResult analyzeRecord(mlir::triton::proton::RecordOp record,
                              SmallVectorImpl<ActiveScope> &activeScopes,
                              std::size_t inheritedScopeCount);
};

FailureOr<ProtonScopePlan> ProtonScopePlan::build(ModuleOp module) {
  ProtonScopePlan plan;
  WalkResult result =
      module.walk<WalkOrder::PreOrder>([&](FunctionOpInterface function) {
        if (function.isExternal()) {
          return WalkResult::advance();
        }
        if (function->getParentOfType<FunctionOpInterface>()) {
          return WalkResult::advance();
        }

        SmallVector<ActiveScope> activeScopes;
        if (failed(
                plan.analyzeRegion(function.getFunctionBody(), activeScopes))) {
          return WalkResult::interrupt();
        }

        return WalkResult::skip();
      });
  if (result.wasInterrupted()) {
    return failure();
  }

  WalkResult allRecordsMapped =
      module.walk([&](mlir::triton::proton::RecordOp record) {
        if (!plan.recordRegionIds.contains(record.getOperation())) {
          record->emitError("A5 Proton records must be nested in a function");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (allRecordsMapped.wasInterrupted()) {
    return failure();
  }
  return plan;
}

std::uint32_t
ProtonScopePlan::getRegionId(mlir::triton::proton::RecordOp record) const {
  return recordRegionIds.lookup(record.getOperation());
}

LogicalResult
ProtonScopePlan::analyzeRegion(Region &region,
                               ArrayRef<ActiveScope> inheritedScopes) {
  for (Block &block : region) {
    SmallVector<ActiveScope> activeScopes(inheritedScopes);
    for (Operation &operation : block) {
      if (auto record = dyn_cast<mlir::triton::proton::RecordOp>(&operation)) {
        if (failed(
                analyzeRecord(record, activeScopes, inheritedScopes.size()))) {
          return failure();
        }
        continue;
      }

      if (isa<FunctionOpInterface>(operation)) {
        continue;
      }

      for (Region &nestedRegion : operation.getRegions()) {
        if (failed(analyzeRegion(nestedRegion, activeScopes))) {
          return failure();
        }
      }
    }

    if (!llvm::equal(activeScopes, inheritedScopes)) {
      return block.getParentOp()->emitError(
          "A5 Proton scope must end in the block where it starts");
    }
  }

  return success();
}

// Host tools parses entries by region_id. The same name on non-overlapping
// scopes is valid and receives a distinct id.
LogicalResult
ProtonScopePlan::analyzeRecord(mlir::triton::proton::RecordOp record,
                               SmallVectorImpl<ActiveScope> &activeScopes,
                               std::size_t inheritedScopeCount) {
  StringRef name = record.getName();
  if (record.getIsStart()) {
    if (llvm::any_of(activeScopes, [&](const ActiveScope &scope) {
          return scope.name == name;
        })) {
      return record.emitError("scope name '") << name << "' is already active";
    }
    if (scopeCount >= hivm::proton::kMaxScopeCount) {
      return record.emitError(
          "too many A5 Proton scopes; 10-bit logical region IDs allow at "
          "most 1024 scopes per module");
    }

    uint32_t logicalId = scopeCount++;
    activeScopes.push_back(ActiveScope{logicalId, name.str()});
    recordRegionIds[record.getOperation()] =
        hivm::proton::getStartRegionId(logicalId);
    return success();
  }

  if (activeScopes.size() == inheritedScopeCount) {
    return record.emitError("scope end crosses a block or region boundary");
  }
  if (activeScopes.back().name != name) {
    return record.emitError("scope end '")
           << name << "' does not match active scope '"
           << activeScopes.back().name << "'";
  }

  uint32_t logicalId = activeScopes.back().logicalId;
  activeScopes.pop_back();
  recordRegionIds[record.getOperation()] =
      hivm::proton::getEndRegionId(logicalId);
  return success();
}

class ConvertProtonToHIVMPass
    : public impl::ConvertProtonToHIVMBase<ConvertProtonToHIVMPass> {
public:
  void runOnOperation() override;
};

void ConvertProtonToHIVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  // A5 only. On any other architecture this pass is a no-op.
  if (!hacc::utils::isAscend910_95(module)) {
    return;
  }

  SmallVector<mlir::triton::proton::RecordOp> records;
  module.walk([&](mlir::triton::proton::RecordOp record) {
    records.push_back(record);
  });
  if (records.empty()) {
    return;
  }

  auto plan = ProtonScopePlan::build(module);
  if (failed(plan)) {
    return signalPassFailure();
  }

  for (auto record : records) {
    OpBuilder builder{record};
    builder.create<hivm::ProtonScopeBoundaryOp>(
        record->getLoc(), builder.getI32IntegerAttr(plan->getRegionId(record)));
    record->erase();
  }
}

} // unnamed namespace

std::unique_ptr<mlir::Pass> createConvertProtonToHIVMPass() {
  return std::make_unique<ConvertProtonToHIVMPass>();
}

} // namespace bishengir::triton::proton::a5
