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
#include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/CubeLoad/CubeLoadCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeCapabilityQuery.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Target/NPU/NPUArchitecture.h"

#include "llvm/ADT/STLExtras.h"
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

template <typename Catalogue>
StringRef resultName(typename Catalogue::Support result) {
  switch (result) {
  case Catalogue::Support::Native:
    return "native";
  case Catalogue::Support::Unknown:
    return "unknown";
  case Catalogue::Support::Unsupported:
    return "unsupported";
  case Catalogue::Support::InvalidRequest:
    return "invalid-request";
  }
  return "unrecognized";
}

template <typename Catalogue>
StringRef capabilityResultNote(typename Catalogue::Support result) {
  switch (result) {
  case Catalogue::Support::Native:
    return "native means this execution schema has a confirmed support row";
  case Catalogue::Support::Unknown:
    return "unknown means this execution schema has no confirmed native row; "
           "it does not mean the hardware rejects the operation";
  case Catalogue::Support::Unsupported:
    return "unsupported means the complete profile has no native row for this "
           "request";
  case Catalogue::Support::InvalidRequest:
    return "invalid-request means the request does not match the execution "
           "schema";
  }
  return "unrecognized capability result";
}

template <typename Catalogue>
std::string formatRequest(const typename Catalogue::Request &request) {
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleaveComma(
      request.values, os, [&](const typename Catalogue::FieldValue &field) {
        if (field.field.kind == Catalogue::FieldKind::Slot) {
          StringRef slot = field.field.position == 0   ? "src"
                           : field.field.position == 1 ? "dst"
                                                       : "";
          if (slot.empty())
            os << "slot" << field.field.position;
          else
            os << slot;
          os << "."
             << Catalogue::property(
                    static_cast<typename Catalogue::PropertyKind>(
                        field.field.identity));
          os << " = ";
          if (const auto *type = std::get_if<Type>(&field.value))
            type->print(os);
          else
            os << "<value>";
          return;
        }
        os << Catalogue::parameter(
                  static_cast<typename Catalogue::ParameterKind>(
                      field.field.identity))
           << " = ";
        const auto *value =
            std::get_if<typename Catalogue::StoredValue>(&field.value);
        if (!value) {
          os << "<value>";
          return;
        }
        if (std::optional<StringRef> spelling = Catalogue::value(*value))
          os << *spelling;
        else
          os << "<invalid>";
      });
  return text;
}

template <typename Catalogue>
std::optional<typename Catalogue::Completeness>
coverageOf(typename Catalogue::Architecture architecture,
           typename Catalogue::Request request) {
  const auto *profile = Catalogue::profile(architecture);
  if (!profile)
    return std::nullopt;
  for (const auto &coverage : profile->capabilities) {
    if (coverage.capability == request.capability)
      return coverage.completeness;
  }
  return std::nullopt;
}

template <typename Catalogue>
std::string formatCheck(hacc::TargetDevice device,
                        typename Catalogue::Architecture architecture,
                        const typename Catalogue::Request &request,
                        typename Catalogue::Support result) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "result is " << resultName<Catalogue>(result) << "; target is "
     << hacc::stringifyTargetDeviceEnum(device) << " ("
     << stringifyNPUArchitecture(architecture) << "); coverage is ";
  if (std::optional<typename Catalogue::Completeness> coverage =
          coverageOf<Catalogue>(architecture, request))
    os << Catalogue::completeness(*coverage);
  else
    os << "none";
  os << "; request is " << formatRequest<Catalogue>(request) << "; "
     << capabilityResultNote<Catalogue>(result);
  return text;
}

template <typename Catalogue>
void checkCatalogue(Operation *root, hivm::ModuleTargetAnalysis &targets,
                    VerifyCapabilityPass &pass) {
  Catalogue::visit(
      root, [&](Operation *op, bool bindingFailed, bool missingAddressSpace,
                std::optional<typename Catalogue::RequestStorage> request) {
        std::optional<hivm::CachedModuleTarget> target = targets.lookup(op);
        if (!target) {
          pass.emit(op, "no #hacc.target on this operation or an enclosing "
                        "module; this check does not assume Ascend910B");
          return;
        }
        if (bindingFailed) {
          pass.emit(op,
                    "the execution request could not be built; this is not a "
                    "hardware result");
          return;
        }
        if (missingAddressSpace || !request) {
          pass.emit(op, "result is unknown; src or dst memory scope is not on "
                        "this operation and was not set for this query");
          return;
        }
        typename Catalogue::Support result =
            Catalogue::query(target->architecture, request->getRequest());
        if (pass.isStrict() && result == Catalogue::Support::Native)
          return;
        std::string message =
            formatCheck<Catalogue>(target->device, target->architecture,
                                   request->getRequest(), result);
        if (!pass.isStrict())
          message += "; report mode does not prove that the operation is legal";
        pass.emit(op, message);
      });
}

void VerifyCapabilityPass::runOnOperation() {
  hivm::ModuleTargetAnalysis &targets =
      getAnalysis<hivm::ModuleTargetAnalysis>();
  Operation *root = getOperation();
  // Each catalogue is one generated library. Every binding with Verify=1 in
  // that library is walked here. A new op does not need a hand-written
  // checker. A new catalogue namespace adds one line.
  checkCatalogue<bishengir::npu::CapabilityCatalogue>(root, targets, *this);
  checkCatalogue<bishengir::npu::mmad::CapabilityCatalogue>(root, targets,
                                                            *this);
  checkCatalogue<bishengir::npu::cubeload::CapabilityCatalogue>(root, targets,
                                                                *this);
  markAnalysesPreserved<hivm::ModuleTargetAnalysis>();
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createVerifyCapabilityPass() {
  return std::make_unique<VerifyCapabilityPass>();
}
