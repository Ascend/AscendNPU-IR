//===- AppendDeviceSpec.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    llvm_unreachable("Unknown target device");

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
  auto maybeSpec = hacc::utils::getNPUTargetSpec(moduleOp);
  if (maybeSpec.has_value())
    moduleOp.emitWarning() << "Overwriting the device spec...";

  MLIRContext *ctx = &getContext();
  hacc::utils::setTargetDevice(moduleOp, finalTarget);
  auto targetSpec = getNPUTargetSpecAttr(ctx, finalTarget, moduleOp->getLoc());
  hacc::utils::setNPUTargetSpec(moduleOp, targetSpec);
}

std::unique_ptr<Pass> mlir::hacc::createAppendDeviceSpecPass(
    const AppendTargetDeviceSpecOptions &options) {
  return std::make_unique<AppendDeviceSpec>(options);
}
