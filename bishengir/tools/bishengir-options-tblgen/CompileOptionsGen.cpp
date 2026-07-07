//===- CompileOptionsGen.cpp - Compile tool's config generation -*- C++ -*-===//
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
// CompileOptionsGen generates static initializers to compile tool's command
// line options.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/ConfigOptions/Options.h"

#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace bishengir::tblgen;

namespace {

bool isSharedWithDownstreamToolchain(const ConfigOption &option) {
  if (const RecordVal *sharedOption =
          option.getDef()->getValue("isSharedWithDownstreamToolchain")) {
  if (option.getDef()->getValue("isSharedWithDownstreamToolchain")) {
    return option.getDef()->getValueAsBit("isSharedWithDownstreamToolchain");
  }
  return false;
}

std::string getCLOptionClass(const ConfigOption &option) {
  std::string result = "llvm::cl::";
  if (option.isListOption())
    result += "list<";
  else
    result += "opt<";
  result += option.getType();

  // External storage
  if (!option.isListOption()) {
    result += ", /*ExternalStorage=*/";
    result += (option.getExternalStorage() ? "true" : "false");
  } else {
    // For list options, we need to specify the data storage type
    if (option.getExternalStorage())
  if (!option.isListOption()) {
    result += ", /*ExternalStorage=*/";
    result += option.shouldUseCLIExternalStorage() ? "true" : "false";
  } else {
    // List options need to specify the externally stored data type when they
    // register against external storage.
    if (option.shouldUseCLIExternalStorage())
      result += ", " + getContainerType(option);
  }

  result += "> ";
  return result;
}

bool emitCompileOption(const ConfigOption &option, raw_ostream &OS) {
  OS << "static ";
  OS << getCLOptionClass(option);
  // Variable Name
  OS << option.getCppVariableName();
  OS << "(\n";
  // Argument
  OS << "    \"" << option.getArgument() << "\"\n";
  // Description
  OS << "    ,llvm::cl::desc(\"" << option.getDescription() << "\")\n";
  // External storage location
  if (option.getExternalStorage()) {
    if (std::optional<StringRef> externalStorageLocation =
            option.getExternalStorageLocation()) {
      OS << "    ,llvm::cl::location(" << externalStorageLocation << ")\n";
    } else {
      PrintFatalError(option.getDef(),
                      "Has external storage but location was not specified");
      return true;
    }
  if (std::optional<StringRef> externalStorageLocation =
          option.getCLIExternalStorageLocation()) {
    OS << "    ,llvm::cl::location(" << externalStorageLocation << ")\n";
  } else if (option.shouldUseCLIExternalStorage()) {
    PrintFatalError(option.getDef(),
                    "External storage location was not specified");
    return true;
  }
  // Default value
  if (!option.isListOption())
    if (std::optional<StringRef> defaultValue = option.getDefaultValue())
      OS << "    ,llvm::cl::init(" << defaultValue << ")\n";
  // Categories
  for (const auto &cat : option.getOptionCategories())
    OS << "    ,llvm::cl::cat(" << cat << ")\n";

  // Check if the option is shared with HIVM compile tool
  if (isSharedWithDownstreamToolchain(option))
    OS << "    ,llvm::cl::cat(sharedWithDownstreamToolchainCategory)\n";

  // Additional flags
  if (std::optional<StringRef> additionalFlags = option.getAdditionalFlags())
    OS << "    ," << additionalFlags << "\n";

  // We implicitly set the MisFlag to comma separated for list options
  if (option.isListOption())
    OS << "    ,llvm::cl::MiscFlags::CommaSeparated\n";

  OS << ");\n";
  return false;
}

bool emitCollectConfigs(const RecordKeeper &records, raw_ostream &OS) {
  std::vector<Record *> specs =
      records.getAllDerivedDefinitions(kOptionClassName);
  OS << "#ifdef GEN_OPTION_COLLECTION\n";
  for (const Record *R : specs) {
    ConfigOption cfgOpt(R);
    if (!isSharedWithDownstreamToolchain(cfgOpt))
      continue;

    // Don't support list option for now
    if (cfgOpt.isListOption())
      continue;

    OS << ""
       << formatv(
              R"(
if (optStr.str() == "{0}") {
  const auto &optPtr = static_cast<{1}*>(opt);
  optValue = option_handler::handleOpt(*optPtr);
}
)",
              cfgOpt.getArgument(), getCLOptionClass(cfgOpt));
  }
  OS << "#undef GEN_OPTION_COLLECTION\n";
  OS << "#endif // GEN_OPTION_COLLECTION\n";
  return false;
}

/// Main entry to emit compile tool's command line option registration.
bool emitCompileOptions(const RecordKeeper &records, raw_ostream &OS) {
  std::vector<Record *> specs =
      records.getAllDerivedDefinitions(kOptionClassName);
  OS << "#ifdef GEN_OPTION_REGISTRATIONS\n";

  for (const Record *R : specs) {
    ConfigOption cfgOpt(R);
    if (!cfgOpt.getEmitOptionRegistration())
    if (!cfgOpt.shouldRegisterCLIOption())
      continue;

    if (emitCompileOption(cfgOpt, OS))
      return true;
  }

  OS << "#undef GEN_OPTION_REGISTRATIONS\n";
  OS << "#endif // GEN_OPTION_REGISTRATIONS\n";
  return false;
}

} // namespace

// Registers the generator to bishengir-options-tblgen.
static mlir::GenRegistration genCompileOptions(
    "gen-compile-options",
    "Generate command line options registration for bishengir compile tool",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitCompileOptions(records, os) || emitCollectConfigs(records, os);
    });
