#include "bishengir/Conversion/ProtonToHIVM/ProtonToHIVM.h"

#include <memory>
#include <string>
#include <tuple>

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMProtonOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"

namespace bishengir::triton::proton::a5 {
#define GEN_PASS_DEF_CONVERTPROTONTOHIVM
#define GEN_PASS_DEF_GETPROTONMETADATA
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

struct ScopeDescriptor {
  std::string name;
  std::int32_t parentId;
};

class ProtonScopePlan {
  SmallVector<ScopeDescriptor> scopes;
  DenseMap<Operation *, std::uint32_t> recordRegionIds;

public:
  static FailureOr<ProtonScopePlan> build(ModuleOp module);

  std::uint32_t getRegionId(mlir::triton::proton::RecordOp record) const;
  ArrayRef<ScopeDescriptor> getScopes() const;

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

ArrayRef<ScopeDescriptor> ProtonScopePlan::getScopes() const { return scopes; }

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
    if (scopes.size() >= hivm::proton::kMaxScopeCount) {
      return record.emitError(
          "too many A5 Proton scopes; 10-bit logical region IDs allow at "
          "most 1024 scopes per module");
    }

    uint32_t logicalId = static_cast<std::uint32_t>(scopes.size());
    std::int32_t parentId = -1;
    if (!activeScopes.empty()) {
      parentId = static_cast<std::int32_t>(activeScopes.back().logicalId);
    }
    scopes.push_back(ScopeDescriptor{name.str(), parentId});

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

void attachMetadata(ModuleOp module, ArrayRef<ScopeDescriptor> scopes) {
  Builder builder{module.getContext()};
  SmallVector<Attribute> scopeAttrs;
  scopeAttrs.reserve(scopes.size());

  for (auto [logicalId, scope] : llvm::enumerate(scopes)) {
    uint32_t id = static_cast<uint32_t>(logicalId);
    scopeAttrs.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("name", builder.getStringAttr(scope.name)),
         builder.getNamedAttr("parent_id",
                              builder.getI32IntegerAttr(scope.parentId)),
         builder.getNamedAttr(
             "start_region_id",
             builder.getI32IntegerAttr(hivm::proton::getStartRegionId(id))),
         builder.getNamedAttr(
             "end_region_id",
             builder.getI32IntegerAttr(hivm::proton::getEndRegionId(id)))}));
  }

  module->setAttr(
      "proton.metadata",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("format_version", builder.getI32IntegerAttr(1)),
           builder.getNamedAttr("scopes", builder.getArrayAttr(scopeAttrs))}));
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
    Value regionId = builder.create<arith::ConstantIntOp>(
        record.getLoc(), static_cast<int64_t>(plan->getRegionId(record)), 32);
    auto boundary = builder.create<hivm::CustomOp>(
        record.getLoc(), hivm::CustomOp::kBuiltinProtonScopeBoundaryName,
        ValueRange{regionId}, ValueRange{});
    boundary.setCoreType(hivm::TCoreType::CUBE_OR_VECTOR);
    boundary.setPipe(hivm::PIPE::PIPE_S);
    boundary.setVFMode(hivm::VFMode::SIMD);
    boundary.setInlineMode(hivm::InlineMode::AlwaysInline);
    record->erase();
  }

  attachMetadata(module, plan->getScopes());
}

class GetProtonMetadataPass
    : public impl::GetProtonMetadataBase<GetProtonMetadataPass> {
public:
  explicit GetProtonMetadataPass(const GetProtonMetadataOptions &options);

  void runOnOperation() override;
};

GetProtonMetadataPass::GetProtonMetadataPass(
    const GetProtonMetadataOptions &options)
    : impl::GetProtonMetadataBase<GetProtonMetadataPass>(options) {}

void GetProtonMetadataPass::runOnOperation() {
  auto module = getOperation();
  auto proton = module->getAttrOfType<DictionaryAttr>("proton.metadata");
  if (!proton || protonMetadataOutput == "") {
    return;
  }

  auto formatVersion = proton.getAs<IntegerAttr>("format_version");
  auto scopes = proton.getAs<ArrayAttr>("scopes");
  if (!formatVersion || !scopes) {
    module.emitError("malformed Proton metadata"); /// TODO: raiify
    return signalPassFailure();
  }

  llvm::json::Array jsonScopes;
  for (auto attr : scopes) {
    auto scope = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!scope) {
      module->emitError("malformed Proton scope metadata");
      return signalPassFailure();
    }

    auto name = scope.getAs<StringAttr>("name");
    auto parentId = scope.getAs<IntegerAttr>("parent_id");
    auto startRegionId = scope.getAs<IntegerAttr>("start_region_id");
    auto endRegionId = scope.getAs<IntegerAttr>("end_region_id");
    if (!name || !parentId || !startRegionId || !endRegionId) {
      module->emitError("malformed Proton scope metadata");
      return signalPassFailure();
    }

    jsonScopes.push_back(
        llvm::json::Object{/// TODO: remove explicit types in aggregates
                           {"name", name.getValue()},
                           {"parent_id", parentId.getInt()},
                           {"start_region_id", startRegionId.getInt()},
                           {"end_region_id", endRegionId.getInt()}});
  }

  llvm::json::Value json{llvm::json::Object{
      {"proton", llvm::json::Object{{"format_version", formatVersion.getInt()},
                                    {"scopes", std::move(jsonScopes)}}}}};

  if (protonMetadataOutput == "--") {
    llvm::outs() << json << "\n";
    return;
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(protonMetadataOutput, ec);
  if (ec) {
    module->emitError("Failed to open Proton metadata output: ")
        << ec.message();
    return signalPassFailure();
  }

  os << json << "\n";
  os.close();
  if (os.has_error()) {
    module->emitError("Failed to write Proton metadata: ")
        << os.error().message();
    signalPassFailure();
  }
}

} // unnamed namespace

std::unique_ptr<mlir::Pass> createConvertProtonToHIVMPass() {
  return std::make_unique<ConvertProtonToHIVMPass>();
}

std::unique_ptr<mlir::Pass>
createGetProtonMetadataPass(const GetProtonMetadataOptions &options) {
  return std::make_unique<GetProtonMetadataPass>(options);
}

} // namespace bishengir::triton::proton::a5
