//===- Options.cpp - Option related classes -------------------------------===//
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

#include "bishengir/Tools/ConfigOptions/Options.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace bishengir::tblgen;

namespace {
bool getBitField(const llvm::Record *def, StringRef fieldName) {
  return def->getValueAsBit(fieldName);
}

std::string lowerLeadingAcronym(StringRef input) {
  if (input.empty())
    return "";

  std::string result = input.str();
  if (!std::isalpha(static_cast<unsigned char>(result.front())))
    return result;

  if (result.size() == 1 || !std::isupper(static_cast<unsigned char>(result[1]))) {
    result.front() = std::tolower(static_cast<unsigned char>(result.front()));
    return result;
  }

  size_t acronymEnd = 1;
  while (acronymEnd < result.size() &&
         std::isupper(static_cast<unsigned char>(result[acronymEnd]))) {
    bool isLastUppercaseBeforeWord =
        acronymEnd + 1 < result.size() &&
        std::islower(static_cast<unsigned char>(result[acronymEnd + 1]));
    if (isLastUppercaseBeforeWord)
      break;
    ++acronymEnd;
  }

  for (size_t i = 0; i < acronymEnd; ++i)
    result[i] = std::tolower(static_cast<unsigned char>(result[i]));
  return result;
}

std::string convertToDashSnakeFromCamelCase(StringRef input) {
  auto snakeCaseStr = llvm::convertToSnakeFromCamelCase(input);
  SmallVector<StringRef> rawParts;
  SmallVector<std::string> normalizedParts;
  StringRef(snakeCaseStr).split(rawParts, '_', /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
  for (StringRef part : rawParts) {
    if (!normalizedParts.empty() && !normalizedParts.back().empty() &&
        std::isdigit(normalizedParts.back().back()) && part.size() <= 2 &&
        llvm::all_of(part, [](char c) { return std::islower(c); })) {
      normalizedParts.back() += part.str();
      continue;
    }
    normalizedParts.push_back(part.str());
  }

  snakeCaseStr.clear();
  for (size_t i = 0; i < normalizedParts.size(); ++i) {
    if (i > 0)
      snakeCaseStr += '_';
    snakeCaseStr += normalizedParts[i];
  }
  std::replace(snakeCaseStr.begin(), snakeCaseStr.end(), '_', '-');
  return snakeCaseStr;
}

std::string upperFirstLetter(StringRef input) {
  if (input.empty())
    return "";

  std::string result = input.str();
  if (std::isalpha(static_cast<unsigned char>(result[0])))
    result[0] = std::toupper(static_cast<unsigned char>(result[0]));
  return result;
}

std::string resolveExplicitStorageLocation(const llvm::Record *def) {
  StringRef explicitExternalStorageLocation =
      def->getValueAsString(OptionFields::kExternalStorageLocation);
  if (!explicitExternalStorageLocation.empty())
    return explicitExternalStorageLocation.str();

  StringRef externalStateObject =
      def->getValueAsString(OptionFields::kExternalStateObject);
  StringRef externalStateField =
      def->getValueAsString(OptionFields::kExternalStateField);
  if (!externalStateObject.empty() || !externalStateField.empty()) {
    if (externalStateObject.empty() || externalStateField.empty())
      return "";
    return (externalStateObject + "." + externalStateField).str();
  }

  return "";
}

} // namespace

namespace bishengir::tblgen {

//===----------------------------------------------------------------------===//
// ConfigOptions
//===----------------------------------------------------------------------===//

ConfigOption::ConfigOption(const llvm::Record *def) : def(def) {
  StringRef defName = def->getName();
  this->cppName = lowerLeadingAcronym(defName);
  this->capitalizedCppName = upperFirstLetter(defName);
  this->argument = convertToDashSnakeFromCamelCase(defName);
  this->externalStorageLocation = resolveExplicitStorageLocation(def);
  if (this->externalStorageLocation.empty())
    this->externalStorageLocation =
        this->cppName + (def->isSubClassOf(OptionFields::kListOption) ? "Flags"
                                                                      : "Flag");
}

StringRef ConfigOption::getCppVariableName() const { return this->cppName; }

StringRef ConfigOption::getCapitalizedCppVariableName() const {
  return this->capitalizedCppName;
}

StringRef ConfigOption::getArgument() const { return this->argument; }

StringRef ConfigOption::getType() const {
  return def->getValueAsString(OptionFields::kType);
}

std::optional<StringRef> ConfigOption::getDefaultValue() const {
  StringRef defaultVal = def->getValueAsString(OptionFields::kDefaultValue);
  return defaultVal.empty() ? std::optional<StringRef>() : defaultVal;
}

StringRef ConfigOption::getDescription() const {
  return def->getValueAsString(OptionFields::kDescription);
}

std::optional<StringRef> ConfigOption::getAdditionalFlags() const {
  StringRef additionalFlags =
      def->getValueAsString(OptionFields::kAdditionalOptFlags);
  return additionalFlags.empty() ? std::optional<StringRef>() : additionalFlags;
}

std::optional<StringRef> ConfigOption::getExternalStorageLocation() const {
  return this->externalStorageLocation;
}

bool ConfigOption::shouldUseCLIExternalStorage() const {
  return getBitField(def, OptionFields::kExternalStorage);
}

std::optional<StringRef> ConfigOption::getCLIExternalStorageLocation() const {
  if (!shouldUseCLIExternalStorage())
    return std::nullopt;
  return getExternalStorageLocation();
}

std::vector<StringRef> ConfigOption::getPassGroups() const {
  return def->getValueAsListOfStrings(OptionFields::kPassGroup);
}

std::vector<StringRef> ConfigOption::getOptionScopes() const {
  return def->getValueAsListOfStrings(OptionFields::kOptionScope);
}

std::vector<mlir::StringRef> ConfigOption::getOptionCategories() const {
  return def->getValueAsListOfStrings(OptionFields::kCompileOptionCategories);
}

std::optional<StringRef> ConfigOption::getCompileConfigName() const {
  return def->getValueAsOptionalString(OptionFields::kCompileConfigName);
}

bool ConfigOption::shouldRegisterCLIOption() const {
  return !isInternalStateOption() &&
         getBitField(def, OptionFields::kEmitOptionRegistration);
}

bool ConfigOption::shouldEmitConfigField() const {
  return !isExternalStateOption();
}

bool ConfigOption::shouldEmitGetterSetter() const {
  return shouldEmitConfigField() &&
         getBitField(def, OptionFields::kEmitGetterSetter);
}

bool ConfigOption::shouldEmitPassOption() const {
  return shouldRegisterCLIOption() && shouldEmitGetterSetter() &&
         !isA3OnlyOption();
}

bool ConfigOption::isListOption() const {
  return def->isSubClassOf(OptionFields::kListOption);
}

bool ConfigOption::isInternalStateOption() const {
  return def->isSubClassOf(OptionFields::kInternalStateOption);
}

bool ConfigOption::isA3OnlyOption() const {
  return llvm::is_contained(getOptionScopes(), StringRef("A3_ONLY"));
}

bool ConfigOption::isA5OnlyOption() const {
  return llvm::is_contained(getOptionScopes(), StringRef("A5_ONLY"));
}

bool ConfigOption::isSharedOption() const {
  return !isA3OnlyOption() && !isA5OnlyOption();
}

bool ConfigOption::isExternalStateOption() const {
  return def->isSubClassOf(OptionFields::kExternalStateOption);
}

std::string getContainerType(const ConfigOption &opt) {
  // By default use vector to hold list data
  if (opt.isListOption())
    return "std::vector<" + opt.getType().str() + ">";

  return opt.getType().str();
}

} // namespace bishengir::tblgen
