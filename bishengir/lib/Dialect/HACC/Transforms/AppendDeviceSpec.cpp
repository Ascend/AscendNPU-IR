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
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

/// The generated target spec declaration
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.cpp.inc"

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

HACCTargetDeviceSpecInterface
getNPUTargetSpecAttr(MLIRContext *context, TargetDevice target, Location loc) {
  auto maybeSpec = getTargetSpec(target);
  if (!maybeSpec.has_value() ||
      maybeSpec.value()->device == TargetDevice::Unknown)
    llvm::report_fatal_error("Unknown target device");

  ImplicitLocOpBuilder builder(loc, context);
  SmallVector<DataLayoutEntryInterface> entries;
  for (uint32_t i = 0; i <= getMaxEnumValForDeviceSpec(); i++) {
    auto specEntry = static_cast<DeviceSpec>(i);
    entries.push_back(DataLayoutEntryAttr::get(
        builder.getStringAttr(stringifyEnum(specEntry)),
        maybeSpec.value()->getSpecEntry(specEntry, builder)));
  }
  return cast<HACCTargetDeviceSpecInterface>(
      hacc::TargetDeviceSpecAttr::get(context, entries));
}

} // namespace

void AppendDeviceSpec::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  auto targetIsUnknown = [](TargetDevice d) -> bool {
    return d == TargetDevice::Unknown;
  };

  TargetDevice targetFromOption = target;
  TargetDevice targetFromIR = TargetDevice::Unknown;
  if (auto targetAttr = moduleOp->getAttrOfType<TargetAttr>(TargetAttr::name))
    targetFromIR = symbolizeTargetDeviceEnum(targetAttr.getTarget());

  // If the target device was not set by hand or by option, do nothing.
  if (targetIsUnknown(targetFromOption) && targetIsUnknown(targetFromIR))
    return;

  // Prefer option if there exist one
  TargetDevice finalTarget = (targetFromOption != TargetDevice::Unknown)
                                 ? targetFromOption
                                 : targetFromIR;

  // Override warn
  if (!targetIsUnknown(targetFromOption) && !targetIsUnknown(targetFromIR) &&
      targetFromOption != targetFromIR)
    moduleOp.emitWarning() << "Overwriting the target by the pass option...";

  // If data layout for NPU has already been populated... overwrite it
  auto maybeSpec = utils::getNPUTargetSpec(moduleOp);
  if (maybeSpec.has_value())
    moduleOp.emitWarning() << "Overwriting the device spec...";

  MLIRContext *ctx = &getContext();
  auto targetSpec = getNPUTargetSpecAttr(ctx, finalTarget, moduleOp->getLoc());
  utils::setNPUTargetSpec(moduleOp, targetSpec);

  llvm::VersionTuple hivmcVersion;
  if (hivmcVersion.tryParse(HIVMCVersion))
    hivmcVersion = llvm::VersionTuple(0, 0, 0);
  moduleOp->setAttr(hacc::HIVMCVersionAttr::name,
                    hacc::HIVMCVersionAttr::get(ctx, hivmcVersion));
  moduleOp->setAttr(hacc::HIVMCCompatiblePrintAttr::name,
                    BoolAttr::get(ctx, false));

  // Remove attr if exists
  moduleOp->removeAttr(TargetAttr::name);
}

std::unique_ptr<Pass> mlir::hacc::createAppendDeviceSpecPass(
    const AppendTargetDeviceSpecOptions &options) {
  return std::make_unique<AppendDeviceSpec>(options);
}
