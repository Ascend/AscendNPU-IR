//===- PassMetadataGen.cpp - HIVM pass metadata generation ------*- C++ -*-===//
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
//
// Emits pass-argument lists from HIVMPass records. Each list is wrapped
// in its own ifdef so one .inc can be included once per list:
//
//   GEN_MIX_KERNEL_PASS_ARGS
//   GEN_PURE_AIV_KERNEL_PASS_ARGS
//   GEN_PURE_AIC_KERNEL_PASS_ARGS
//   GEN_CLASSIFY_KERNEL_PASS_ARGS
//
// A pass is listed for each kernel kind in its set. The pipeline skips the
// pass when the module core type maps to a kind that is not in that set.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

namespace {

struct PassMetadata {
  std::string argument;
  bool mix = false;
  bool pureAiv = false;
  bool pureAic = false;
  bool classifyKernel = false;
};

void parseKernelKinds(const Record &def, PassMetadata &metadata) {
  auto kinds = def.getValueAsListOfDefs("appliedKernelKinds");
  if (kinds.empty())
    PrintFatalError(def.getLoc(), "HIVMPass kernelKinds must not be empty");
  for (const Record *kind : kinds) {
    StringRef name = kind->getName();
    bool *seen = nullptr;
    if (name == "MixKernel")
      seen = &metadata.mix;
    else if (name == "PureAivKernel")
      seen = &metadata.pureAiv;
    else if (name == "PureAicKernel")
      seen = &metadata.pureAic;
    else
      PrintFatalError(def.getLoc(),
                      "unknown HIVMPass kernel kind '" + name + "'");
    if (*seen)
      PrintFatalError(def.getLoc(),
                      "duplicate HIVMPass kernel kind '" + name + "'");
    *seen = true;
  }
}

void emitList(raw_ostream &os, StringRef macro, ArrayRef<PassMetadata> passes,
              bool (*predicate)(const PassMetadata &)) {
  os << "#ifdef " << macro << "\n";
  for (const PassMetadata &pass : passes) {
    if (predicate(pass))
      os << "\"" << pass.argument << "\",\n";
  }
  os << "#endif // " << macro << "\n\n";
}

bool hasMix(const PassMetadata &pass) { return pass.mix; }
bool hasPureAiv(const PassMetadata &pass) { return pass.pureAiv; }
bool hasPureAic(const PassMetadata &pass) { return pass.pureAic; }
bool classifies(const PassMetadata &pass) { return pass.classifyKernel; }

bool emitPassMetadata(const RecordKeeper &records, raw_ostream &os) {
  std::vector<PassMetadata> passes;
  for (const Record *def : records.getAllDerivedDefinitions("PassBase")) {
    // Passes.td is the HIVM pass list. A concrete pass that uses mlir Pass
    // or InterfacePass has no kernel metadata, so reject it here.
    if (!def->isSubClassOf("HIVMPass"))
      PrintFatalError(def->getLoc(), "pass '" + def->getName() +
                                         "' must derive from HIVMPass");
    PassMetadata metadata;
    metadata.argument = def->getValueAsString("argument").str();
    parseKernelKinds(*def, metadata);
    metadata.classifyKernel = def->getValueAsBit("appliedClassifyKernel");
    if (metadata.argument.empty())
      PrintFatalError(def->getLoc(),
                      "HIVMPass metadata requires a pass argument");
    passes.push_back(std::move(metadata));
  }

  llvm::sort(passes, [](const PassMetadata &lhs, const PassMetadata &rhs) {
    return lhs.argument < rhs.argument;
  });

  os << "// Generated from HIVMPass records. Do not edit.\n";
  emitList(os, "GEN_MIX_KERNEL_PASS_ARGS", passes, hasMix);
  emitList(os, "GEN_PURE_AIV_KERNEL_PASS_ARGS", passes, hasPureAiv);
  emitList(os, "GEN_PURE_AIC_KERNEL_PASS_ARGS", passes, hasPureAic);
  emitList(os, "GEN_CLASSIFY_KERNEL_PASS_ARGS", passes, classifies);
  return false;
}

static mlir::GenRegistration genPassMetadata(
    "gen-bishengir-pass-metadata",
    "Generate kernel-kind and classifyKernel lists for HIVMPass",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitPassMetadata(records, os);
    });

} // namespace
