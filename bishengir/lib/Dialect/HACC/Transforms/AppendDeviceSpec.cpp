//===- AppendDeviceSpec.cpp --------------------------------------*- C++-*-===//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpecUtils.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace hacc {
#define GEN_PASS_DEF_APPENDTARGETDEVICESPEC
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"
} // namespace hacc
} // namespace mlir

using namespace mlir;
using namespace mlir::hacc;

namespace {
struct AppendDeviceSpec
    : public mlir::hacc::impl::AppendTargetDeviceSpecBase<AppendDeviceSpec> {
  using AppendDeviceSpecBase =
      mlir::hacc::impl::AppendTargetDeviceSpecBase<AppendDeviceSpec>;

public:
  explicit AppendDeviceSpec(const AppendTargetDeviceSpecOptions &options)
      : AppendDeviceSpecBase(options) {}

  void runOnOperation() override;
};

/// The core counts that end up in the device spec.
struct CoreCounts {
  int32_t aiCore;
  int32_t cubeCore;
  int32_t vectorCore;
};

/// A value supplied on the command line wins over the target's default; -1
/// means the option was not specified.
CoreCounts resolveCoreCounts(const TargetSpec &spec, int32_t customAIC,
                             int32_t customAIV) {
  return {customAIC == -1 ? spec.AiCoreCount : customAIC,
          customAIC == -1 ? spec.CubeCoreCount : customAIC,
          customAIV == -1 ? spec.VectorCoreCount : customAIV};
}

LogicalResult validateCoreCounts(const TargetSpec &spec, CoreCounts counts,
                                 Location loc) {
  if (counts.aiCore > 0 && counts.cubeCore > 0 && counts.vectorCore > 0)
    return success();

  return emitError(loc) << "invalid core counts for target '"
                        << stringifyTargetDeviceEnum(spec.device)
                        << "': AI_CORE_COUNT=" << counts.aiCore
                        << ", CUBE_CORE_COUNT=" << counts.cubeCore
                        << ", VECTOR_CORE_COUNT=" << counts.vectorCore
                        << "; core counts must be strictly positive, pass "
                           "--custom-aic-number / --custom-aiv-number for "
                           "targets that have no default core counts";
}

HACCTargetDeviceSpecInterface buildNPUTargetSpecAttr(MLIRContext *context,
                                                     const TargetSpec &spec,
                                                     CoreCounts counts,
                                                     Location loc) {
  ImplicitLocOpBuilder builder(loc, context);
  SmallVector<DataLayoutEntryInterface> entries;
  const bool isRegBase = hacc::utils::isRegBasedArch(spec.device);
  for (uint32_t i = 0; i <= getMaxEnumValForDeviceSpec(); i++) {
    auto specEntry = static_cast<DeviceSpec>(i);

    // MINIMAL_D_CACHE_SIZE / MAXIMUM_D_CACHE_SIZE / ARCH are regbase-only
    // (introduced by A5). The external hivmc shipped with CANN for membase
    // targets cannot parse them, so skip them on membase arch.
    if (!isRegBase && (specEntry == DeviceSpec::MINIMAL_D_CACHE_SIZE ||
                       specEntry == DeviceSpec::MAXIMUM_D_CACHE_SIZE ||
                       specEntry == DeviceSpec::ARCH)) {
      continue;
    }

    Attribute specValue;
    switch (specEntry) {
    case DeviceSpec::AI_CORE_COUNT:
      specValue = builder.getI32IntegerAttr(counts.aiCore);
      break;
    case DeviceSpec::CUBE_CORE_COUNT:
      specValue = builder.getI32IntegerAttr(counts.cubeCore);
      break;
    case DeviceSpec::VECTOR_CORE_COUNT:
      specValue = builder.getI32IntegerAttr(counts.vectorCore);
      break;
    default:
      specValue = spec.getSpecEntry(specEntry, builder);
      break;
    }

    entries.push_back(DataLayoutEntryAttr::get(
        builder.getStringAttr(stringifyEnum(specEntry)), specValue));
  }
  return cast<HACCTargetDeviceSpecInterface>(
      hacc::TargetDeviceSpecAttr::get(context, entries));
}

} // namespace

void AppendDeviceSpec::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  auto isKnownTarget = [](TargetDevice d) -> bool {
    return d != TargetDevice::Unknown;
  };

  TargetDevice targetFromOption = target;
  TargetDevice targetFromIR = TargetDevice::Unknown;
  if (auto targetAttr = moduleOp->getAttrOfType<TargetAttr>(TargetAttr::name))
    targetFromIR = symbolizeTargetDeviceEnum(targetAttr.getTarget());

  // If the target device was not set by hand or by option, do nothing.
  if (!isKnownTarget(targetFromOption) && !isKnownTarget(targetFromIR)) {
    return;
  }

  // Prefer option if there exist one
  TargetDevice finalTarget =
      isKnownTarget(targetFromOption) ? targetFromOption : targetFromIR;
  auto maybeSpec = lookupTargetSpec(finalTarget);
  if (!maybeSpec.has_value() ||
      maybeSpec.value()->device == TargetDevice::Unknown)
    llvm::report_fatal_error("Unknown target device");
  const TargetSpec *spec = *maybeSpec;

  // Override warn
  if (isKnownTarget(targetFromOption) && isKnownTarget(targetFromIR) &&
      targetFromOption != targetFromIR) {
    moduleOp.emitWarning() << "Overwriting the target by the pass option...";
  }

  // If data layout for NPU has already been populated... overwrite it
  if (hacc::utils::getNPUTargetSpec(moduleOp)) {
    moduleOp.emitWarning() << "Overwriting the device spec...";
  }

  const CoreCounts counts =
      resolveCoreCounts(*spec, customAICNumber, customAIVNumber);

  if (failed(validateCoreCounts(*spec, counts, moduleOp->getLoc()))) {
    return signalPassFailure();
  }

  MLIRContext *ctx = &getContext();
  auto specAttr =
      buildNPUTargetSpecAttr(ctx, *spec, counts, moduleOp->getLoc());

  hacc::utils::setTargetDevice(moduleOp, finalTarget);
  hacc::utils::setNPUTargetSpec(moduleOp, specAttr);

  if (hacc::utils::isMemBasedArch(finalTarget)) {
    auto hivmcVersion = llvm::VersionTuple(0, 0, 0);
    moduleOp->setAttr(hacc::HIVMCVersionAttr::name,
                      hacc::HIVMCVersionAttr::get(ctx, hivmcVersion));
    moduleOp->setAttr(hacc::HIVMCCompatiblePrintAttr::name,
                      BoolAttr::get(ctx, false));
  }
}

std::unique_ptr<Pass> mlir::hacc::createAppendDeviceSpecPass(
    const AppendTargetDeviceSpecOptions &options) {
  return std::make_unique<AppendDeviceSpec>(options);
}
