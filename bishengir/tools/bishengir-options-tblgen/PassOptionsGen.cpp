//===- PassOptionsGen.cpp - Pass and pipeline option generation -*- C++ -*-===//
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
//
// PassOptionsGen generates pass and pipeline's option registration.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/ConfigOptions/Options.h"

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace bishengir::tblgen;

static std::string getPassGroupVarName(StringRef passGroup) {
  return "GEN_" + passGroup.upper() + "_OPTION_REGISTRATION";
}

static void genPassOption(const ConfigOption &option, raw_ostream &OS,
                          bool isPipeline) {
  // Pipeline options are declared within the derived classes of
  // `PassPipelineOptions`.
  if (isPipeline)
    OS << "PassOptions::";
  else
    OS << "Pass::";

  if (option.isListOption())
    OS << "ListOption<";
  else
    OS << "Option<";

  OS << option.getType();
  OS << ">\n";
  // Variable
  OS << "" << option.getCppVariableName() << "";
  OS << "{\n";
  OS << "  *this\n";
  // Argument
  OS << "  ,\"" << option.getArgument() << "\"\n";
  // Description
  OS << "  ,llvm::cl::desc(\"" << option.getDescription() << "\")\n";
  // Default value
  if (std::optional<StringRef> defaultValue = option.getDefaultValue())
    OS << "  ,llvm::cl::init(" << defaultValue << ")\n";
  // Additional flags
  if (std::optional<StringRef> additionalFlags = option.getAdditionalFlags())
    OS << "    ," << additionalFlags << "\n";
  OS << "};\n";
}

static void genPassOptionsForGroup(StringRef group,
                                   ArrayRef<ConfigOption> options,
                                   llvm::raw_ostream &OS, bool isPipeline) {
  const std::string passGroupVarName = getPassGroupVarName(group);
  OS << "#ifdef " << passGroupVarName << "\n";

  for (const auto &option : options)
    genPassOption(option, OS, isPipeline);

  OS << "#undef " << passGroupVarName << "\n";
  OS << "#endif // " << passGroupVarName << "\n\n";
}

static void genPassOptionsForGroups(
    const StringMap<SmallVector<ConfigOption>> &passGroup2Options,
    raw_ostream &OS, bool isPipeline) {
  for (const auto &iter : passGroup2Options) {
    genPassOptionsForGroup(iter.getKey(), iter.getValue(), OS, isPipeline);
  }
}

/// Main entry to emit target spec decls.
static bool emitPassOptions(const RecordKeeper &records, raw_ostream &OS,
                            bool isPipeline = false) {
  std::vector<Record *> specs =
      records.getAllDerivedDefinitions(kOptionClassName);
  StringMap<SmallVector<ConfigOption>> passGroup2Options;
  for (const Record *R : specs) {
    ConfigOption cfgOpt(R);
    if (!cfgOpt.shouldEmitPassOption())
      continue;
    std::vector<StringRef> groups = cfgOpt.getPassGroups();
    passGroup2Options["ALL"].emplace_back(R);
    if (groups.empty())
      continue;

    for (StringRef g : groups)
      passGroup2Options[g].emplace_back(R);
  }
  genPassOptionsForGroups(passGroup2Options, OS, isPipeline);
  return false;
}

// Registers the generator to bishengir-options-tblgen.
static mlir::GenRegistration
    genPassOptions("gen-pass-options", "Generate options for pass",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitPassOptions(records, os);
                   });

static mlir::GenRegistration genPassPipelineOptions(
    "gen-pass-pipeline-options", "Generate options for pass pipeline",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitPassOptions(records, os, /*isPipeline=*/true);
    });
