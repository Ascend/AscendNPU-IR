//===- ConfigsToOptionsConversionGen.cpp ------------------------*- C++ -*-===//
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
// ConfigsToOptionsConversionGen generates the code to set up pipeline options
// from compile config or vice versa.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/ConfigOptions/Options.h"

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace bishengir::tblgen;

namespace {

std::string getPassGroupVarName(StringRef passGroup) {
  return "GEN_" + passGroup.upper() + "_OPTION_SETUP";
}

void genConfigToOptions(const ConfigOption &option, raw_ostream &OS) {
  OS << llvm::formatv(R"(
options.{0} = config.get{1}();
)",
                      option.getCppVariableName(),
                      option.getCapitalizedCppVariableName());
}

void genConfigsAndOptionsConversionForGroup(StringRef group,
                                            ArrayRef<ConfigOption> options,
                                            raw_ostream &OS) {
  const std::string passGroupVarName = getPassGroupVarName(group);
  OS << "#ifdef " << passGroupVarName << "\n";

  for (const auto &option : options)
    genConfigToOptions(option, OS);

  OS << "#undef " << passGroupVarName << "\n";
  OS << "#endif // " << passGroupVarName << "\n\n";
}

void genConfigsAndOptionsConversionForGroups(
    const StringMap<SmallVector<ConfigOption>> &passGroup2Options,
    raw_ostream &OS) {
  for (const auto &iter : passGroup2Options)
    genConfigsAndOptionsConversionForGroup(iter.getKey(), iter.getValue(), OS);
}

/// Main entry to generate conversion between compile configs and pipeline
/// options.
bool emitConfigsAndOptionsConversion(const RecordKeeper &records,
                                     raw_ostream &OS) {
  std::vector<Record *> specs =
      records.getAllDerivedDefinitions(kOptionClassName);
  StringMap<SmallVector<ConfigOption>> passGroup2Options;
  for (const Record *R : specs) {
    ConfigOption cfgOpt(R);
    std::vector<StringRef> groups = cfgOpt.getPassGroups();
    if (groups.empty())
      continue;

    for (StringRef g : groups)
      passGroup2Options[g].emplace_back(R);
  }
  genConfigsAndOptionsConversionForGroups(passGroup2Options, OS);
  return false;
}

} // namespace

// Registers the generator to bishengir-options-tblgen.
static mlir::GenRegistration genConfigsAndOptionsConversion(
    "gen-configs-and-options-conversion",
    "Generate conversions between config and pass pipeline options",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitConfigsAndOptionsConversion(records, os);
    });
