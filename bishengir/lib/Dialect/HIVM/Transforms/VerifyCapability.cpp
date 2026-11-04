//===- VerifyCapability.cpp - HIVM target capability check -------*- C++
//-*-===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeCapabilityQuery.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_VERIFYCAPABILITY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace bishengir::npu;

namespace {

struct VerifyCapabilityPass
    : public impl::VerifyCapabilityBase<VerifyCapabilityPass> {
  using impl::VerifyCapabilityBase<VerifyCapabilityPass>::VerifyCapabilityBase;

  void emit(Operation *op, const Twine &message) {
    if (strict)
      op->emitOpError(message);
    else
      op->emitRemark(message);
  }

  bool isStrict() const { return strict; }

  void runOnOperation() override;
};

StringRef capabilityResultNote(NativeSupport result) {
  switch (result) {
  case NativeSupport::Native:
    return "native means this execution schema has a confirmed support row";
  case NativeSupport::Unknown:
    return "unknown means this execution schema has no confirmed native row; "
           "it does not mean the hardware rejects the operation";
  case NativeSupport::Unsupported:
    return "unsupported means the complete profile has no native row for this "
           "request";
  case NativeSupport::InvalidRequest:
    return "invalid-request means the request does not match the execution "
           "schema";
  }
  return "unrecognized capability result";
}

StringRef resultName(NativeSupport result) {
  switch (result) {
  case NativeSupport::Native:
    return "native";
  case NativeSupport::Unknown:
    return "unknown";
  case NativeSupport::Unsupported:
    return "unsupported";
  case NativeSupport::InvalidRequest:
    return "invalid-request";
  }
  return "unrecognized";
}

std::string formatRequest(const CapabilityRequest &request) {
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleaveComma(
      request.values, os, [&](const CapabilityFieldValue &field) {
        if (field.field.kind == CapabilityRequestFieldKind::Slot) {
          StringRef slot = field.field.position == 0   ? "src"
                           : field.field.position == 1 ? "dst"
                                                       : "";
          if (slot.empty())
            os << "slot" << field.field.position;
          else
            os << slot;
          os << "."
             << stringifyCapabilitySlotPropertyKind(
                    static_cast<CapabilitySlotPropertyKind>(
                        field.field.identity));
          os << " = ";
          if (const auto *type = std::get_if<Type>(&field.value))
            type->print(os);
          else
            os << "<value>";
          return;
        }
        os << stringifyCapabilityParameterKind(
                  static_cast<CapabilityParameterKind>(field.field.identity))
           << " = ";
        const auto *value = std::get_if<CapabilityValue>(&field.value);
        if (!value) {
          os << "<value>";
          return;
        }
        if (std::optional<StringRef> spelling =
                stringifyCapabilityValue(*value))
          os << *spelling;
        else
          os << "<invalid>";
      });
  return text;
}

std::optional<ProfileCompleteness>
fixpipeCoverage(NPUArchitectureKind architecture) {
  const ArchitectureCapabilityProfileDesc *profile =
      lookupArchitectureCapabilityProfile(architecture);
  if (!profile)
    return std::nullopt;
  for (const ArchitectureCapabilityCoverageDesc &coverage :
       profile->capabilities) {
    if (coverage.capability == CapabilityKind::Fixpipe)
      return coverage.completeness;
  }
  return std::nullopt;
}

std::string formatCheck(hacc::TargetDevice device,
                        NPUArchitectureKind architecture,
                        const CapabilityRequest &request,
                        NativeSupport result) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "result is " << resultName(result) << "; target is "
     << hacc::stringifyTargetDeviceEnum(device) << " ("
     << stringifyNPUArchitecture(architecture) << "); coverage is ";
  if (std::optional<ProfileCompleteness> coverage =
          fixpipeCoverage(architecture))
    os << stringifyProfileCompleteness(*coverage);
  else
    os << "none";
  os << "; request is " << formatRequest(request) << "; "
     << capabilityResultNote(result);
  return text;
}

/// One op's capability check. The pass runs every registered checker.
class CapabilityChecker {
public:
  virtual ~CapabilityChecker() = default;
  virtual void run(Operation *root, hivm::ModuleTargetAnalysis &targets,
                   VerifyCapabilityPass &pass) = 0;
};

class FixpipeCapabilityChecker : public CapabilityChecker {
public:
  void run(Operation *root, hivm::ModuleTargetAnalysis &targets,
           VerifyCapabilityPass &pass) override {
    root->walk([&](hivm::FixpipeOp op) { check(op, targets, pass); });
  }

private:
  void check(hivm::FixpipeOp op, hivm::ModuleTargetAnalysis &targets,
             VerifyCapabilityPass &pass) {
    std::optional<hivm::CachedModuleTarget> target = targets.lookup(op);
    if (!target) {
      pass.emit(op, "no #hacc.target on this operation or an enclosing "
                    "module; this check does not assume Ascend910B");
      return;
    }
    if (!hivm::getOptionalHIVMAddressSpace(op.getSrc().getType()) ||
        !hivm::getOptionalHIVMAddressSpace(op.getDst().getType())) {
      pass.emit(op, "result is unknown; src or dst memory scope is not on "
                    "this operation and was not set for this query");
      return;
    }
    std::optional<FixpipeCapabilityRequestBuilder> builder =
        FixpipeCapabilityRequestBuilder::from(op);
    if (!builder) {
      pass.emit(op, "the Fixpipe execution request could not be built; this "
                    "is not a hardware result");
      return;
    }
    CapabilityRequestStorage storage = builder->build();
    NativeSupport result =
        queryNativeSupport(target->architecture, storage.getRequest());
    if (pass.isStrict() && result == NativeSupport::Native)
      return;
    std::string message = formatCheck(target->device, target->architecture,
                                      storage.getRequest(), result);
    if (!pass.isStrict())
      message += "; report mode does not prove that the operation is legal";
    pass.emit(op, message);
  }
};

void VerifyCapabilityPass::runOnOperation() {
  hivm::ModuleTargetAnalysis &targets =
      getAnalysis<hivm::ModuleTargetAnalysis>();

  SmallVector<std::unique_ptr<CapabilityChecker>> checkers;
  checkers.push_back(std::make_unique<FixpipeCapabilityChecker>());
  Operation *root = getOperation();
  for (auto &checker : checkers)
    checker->run(root, targets, *this);

  markAnalysesPreserved<hivm::ModuleTargetAnalysis>();
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createVerifyCapabilityPass() {
  return std::make_unique<VerifyCapabilityPass>();
}
