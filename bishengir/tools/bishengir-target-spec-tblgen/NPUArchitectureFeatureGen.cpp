//===- NPUArchitectureFeatureGen.cpp - NPU feature generation -------------===//
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

#include "NPUArchitectureGen.h"

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <algorithm>
#include <utility>

using namespace llvm;

namespace {

static bool isValidCppIdentifier(StringRef name) {
  if (name.empty() || (!llvm::isAlpha(name.front()) && name.front() != '_'))
    return false;
  for (char character : name.drop_front())
    if (!llvm::isAlnum(character) && character != '_')
      return false;
  // Feature type and accessor names are emitted verbatim, unlike prefixed
  // enumerators. Reject language keywords as well as malformed identifiers.
  return !StringSwitch<bool>(name)
              .Cases("alignas", "alignof", "and", "and_eq", "asm", true)
              .Cases("auto", "bitand", "bitor", "bool", "break", true)
              .Cases("case", "catch", "char", "char8_t", "char16_t", true)
              .Cases("char32_t", "class", "compl", "concept", "const", true)
              .Cases("consteval", "constexpr", "constinit", "const_cast", true)
              .Cases("continue", "co_await", "co_return", "co_yield", true)
              .Cases("decltype", "default", "delete", "do", "double", true)
              .Cases("dynamic_cast", "else", "enum", "explicit", "export", true)
              .Cases("extern", "false", "float", "for", "friend", true)
              .Cases("goto", "if", "inline", "int", "long", "mutable", true)
              .Cases("namespace", "new", "noexcept", "not", "not_eq", true)
              .Cases("nullptr", "operator", "or", "or_eq", "private", true)
              .Cases("protected", "public", "register", "reinterpret_cast",
                     true)
              .Cases("requires", "return", "short", "signed", "sizeof", true)
              .Cases("static", "static_assert", "static_cast", "struct", true)
              .Cases("switch", "template", "this", "thread_local", "throw",
                     true)
              .Cases("true", "try", "typedef", "typeid", "typename", true)
              .Cases("union", "unsigned", "using", "virtual", "void", true)
              .Cases("volatile", "wchar_t", "while", "xor", "xor_eq", true)
              .Default(false);
}

static std::string getSpelling(const Record *record) {
  StringRef spelling = record->getValueAsString("Name");
  if (!spelling.empty())
    return spelling.str();
  std::string name =
      convertToSnakeFromCamelCase(record->getValueAsString("CppName"));
  std::replace(name.begin(), name.end(), '_', '-');
  return name;
}

struct EnumFeature {
  const Record *record;
  std::string cppName;
  std::string spelling;
  std::string accessor;
  std::string fieldName;
  SmallVector<const Record *> values;
};

static SmallVector<EnumFeature>
collectEnumFeatures(const RecordKeeper &records) {
  auto featureValues =
      records.getAllDerivedDefinitions("EnumArchitectureFeatureValue");
  SmallVector<EnumFeature> features;
  for (const Record *record :
       records.getAllDerivedDefinitions("EnumArchitectureFeature")) {
    std::string cppName = record->getValueAsString("CppName").str();
    StringRef accessor = record->getValueAsString("Accessor");
    // Do not call CamelCase conversion on an empty name. The validator below
    // diagnoses it, before any code can be emitted.
    std::string fieldName =
        cppName.empty()
            ? std::string()
            : convertToCamelFromSnakeCase(convertToSnakeFromCamelCase(cppName),
                                          /*capitalizeFirst=*/false);
    EnumFeature feature{record,
                        cppName,
                        getSpelling(record),
                        accessor.empty() ? "get" + cppName : accessor.str(),
                        fieldName,
                        {}};
    for (const Record *value : featureValues)
      if (value->getValueAsDef("Feature") == record)
        feature.values.push_back(value);
    features.push_back(std::move(feature));
  }
  return features;
}

static bool validateFeatures(const RecordKeeper &records,
                             ArrayRef<EnumFeature> features) {
  bool hadError = false;
  auto error = [&](const Record *record, const Twine &message) {
    PrintError(record->getLoc(), message);
    hadError = true;
  };
  auto validateName = [&](const Record *record, StringRef name,
                          StringRef kind) {
    if (name.empty())
      error(record, kind + " must not be empty");
    else if (!isValidCppIdentifier(name))
      error(record, "invalid " + kind + ": " + name);
  };

  for (const Record *feature :
       records.getAllDerivedDefinitions("ArchitectureFeature"))
    if (!feature->isSubClassOf("EnumArchitectureFeature"))
      error(feature,
            "unsupported architecture feature kind: " + feature->getName());
  for (const Record *setting :
       records.getAllDerivedDefinitions("ArchitectureFeatureSetting"))
    if (!setting->isSubClassOf("EnumArchitectureFeatureValue"))
      error(setting, "unsupported architecture feature setting kind");

  StringSet<> featureSpellings;
  StringSet<> featureCppNames;
  StringSet<> featureAccessors;
  StringSet<> featureFieldNames;
  // Check collisions across generated symbols, not just within each category.
  // The architecture identity API shares this namespace with the feature API.
  StringSet<> generatedNames;
  for (StringRef name :
       {"NPUArchitectureKind", "stringifyNPUArchitecture",
        "symbolizeNPUArchitecture", "ArchitectureFeatureProfile",
        "lookupArchitectureFeatureProfile"})
    generatedNames.insert(name);
  auto reserveName = [&](const Record *record, StringRef name) {
    if (!generatedNames.insert(name).second)
      error(record,
            "duplicate generated architecture feature C++ identifier: " + name);
  };

  for (const EnumFeature &feature : features) {
    const Record *record = feature.record;
    validateName(record, feature.cppName, "architecture feature C++ name");
    validateName(record, feature.accessor, "architecture feature accessor");
    if (!featureSpellings.insert(feature.spelling).second)
      error(record,
            "duplicate architecture feature spelling: " + feature.spelling);
    if (!featureCppNames.insert(feature.cppName).second)
      error(record,
            "duplicate architecture feature C++ name: " + feature.cppName);
    if (!featureAccessors.insert(feature.accessor).second)
      error(record,
            "duplicate architecture feature accessor: " + feature.accessor);
    if (feature.fieldName == "architecture")
      error(record, "architecture feature profile field 'architecture' is "
                    "reserved");
    else if (!featureFieldNames.insert(feature.fieldName).second)
      error(record, "duplicate architecture feature profile field: " +
                        feature.fieldName);
    validateName(record, feature.fieldName,
                 "architecture feature profile field");
    reserveName(record, feature.cppName);
    reserveName(record, "stringify" + feature.cppName);
    reserveName(record, "symbolize" + feature.cppName);
    reserveName(record, feature.accessor);

    StringSet<> valueSpellings;
    StringSet<> valueCppNames;
    if (feature.values.empty())
      error(record,
            "architecture enum feature has no values: " + feature.spelling);
    for (const Record *value : feature.values) {
      StringRef cppName = value->getValueAsString("CppName");
      std::string spelling = getSpelling(value);
      validateName(value, cppName, "architecture feature value C++ name");
      if (cppName == "Unknown")
        error(value,
              "architecture feature value C++ name 'Unknown' is reserved "
              "for kUnknown");
      if (spelling == "unknown")
        error(value,
              "architecture feature value spelling 'unknown' is reserved");
      if (!valueCppNames.insert(cppName).second)
        error(value, "duplicate C++ value for architecture feature '" +
                         feature.spelling + "': " + cppName);
      if (!valueSpellings.insert(spelling).second)
        error(value, "duplicate value spelling for architecture feature '" +
                         feature.spelling + "': " + spelling);
    }
  }
  return !hadError;
}

static bool validateProfiles(const RecordKeeper &records,
                             ArrayRef<EnumFeature> features) {
  bool hadError = false;
  auto error = [&](const Record *record, const Twine &message) {
    PrintError(record->getLoc(), message);
    hadError = true;
  };
  DenseMap<const Record *, const Record *> profilesByArchitecture;
  for (const Record *profile :
       records.getAllDerivedDefinitions("ArchitectureFeatureProfile")) {
    const Record *architecture = profile->getValueAsDef("Architecture");
    StringRef architectureName = architecture->getValueAsString("Name");
    if (!profilesByArchitecture.try_emplace(architecture, profile).second)
      error(profile,
            "duplicate feature profile for architecture: " + architectureName);

    DenseMap<const Record *, const Record *> selectedFeatures;
    for (const Record *setting : profile->getValueAsListOfDefs("Features")) {
      // The feature validator diagnoses unsupported setting kinds, including
      // unused records. Do not read enum-specific fields from them here.
      if (!setting->isSubClassOf("EnumArchitectureFeatureValue"))
        continue;
      const Record *feature = setting->getValueAsDef("Feature");
      if (!selectedFeatures.try_emplace(feature, setting).second)
        error(profile, "multiple values for architecture feature '" +
                           getSpelling(feature) +
                           "' in architecture: " + architectureName);
    }
    for (const EnumFeature &feature : features)
      if (!selectedFeatures.contains(feature.record))
        error(profile, "missing value for architecture feature '" +
                           feature.spelling +
                           "' in architecture: " + architectureName);
  }
  for (const Record *architecture :
       records.getAllDerivedDefinitions("NPUArchitecture"))
    if (!profilesByArchitecture.contains(architecture))
      error(architecture, "missing feature profile for architecture: " +
                              architecture->getValueAsString("Name"));
  return !hadError;
}

static void emitFeatureEnum(const EnumFeature &feature, raw_ostream &os) {
  StringRef cppName = feature.cppName;
  os << "enum class " << cppName << " {\n";
  for (const Record *value : feature.values)
    os << "  k" << value->getValueAsString("CppName") << ",\n";
  os << "  kUnknown,\n};\n\ninline ::llvm::StringRef stringify" << cppName
     << "(" << cppName << " value) {\n  switch (value) {\n";
  for (const Record *value : feature.values) {
    os << "  case " << cppName << "::k" << value->getValueAsString("CppName")
       << ": return \"";
    os.write_escaped(getSpelling(value));
    os << "\";\n";
  }
  os << "  case " << cppName << R"(::kUnknown: return "unknown";
  }
  llvm_unreachable("unknown architecture feature value");
}

inline )"
     << cppName << " symbolize" << cppName << R"((::llvm::StringRef name) {
  return ::llvm::StringSwitch<::bishengir::npu::)"
     << cppName << ">(name)\n";
  for (const Record *value : feature.values) {
    os << "      .Case(\"";
    os.write_escaped(getSpelling(value));
    os << "\", " << cppName << "::k" << value->getValueAsString("CppName")
       << ")\n";
  }
  os << "      .Default(" << cppName << "::kUnknown);\n}\n\n";
}

static bool emitNPUArchitectureFeatureDecls(const RecordKeeper &records,
                                            raw_ostream &os) {
  SmallVector<EnumFeature> features = collectEnumFeatures(records);
  bool validFeatures = validateFeatures(records, features);
  bool validProfiles = validateProfiles(records, features);
  if (!validFeatures || !validProfiles)
    return true;

  emitSourceFileHeader("NPU Architecture Feature Declarations", os, records);
  os << R"(
#ifndef BISHENGIR_NPU_ARCHITECTURE_FEATURES_H
#define BISHENGIR_NPU_ARCHITECTURE_FEATURES_H

#include "bishengir/Target/NPU/NPUArchitecture.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace bishengir {
namespace npu {

)";
  for (const EnumFeature &feature : features)
    emitFeatureEnum(feature, os);

  os << R"(struct ArchitectureFeatureProfile {
  NPUArchitectureKind architecture;
)";
  for (const EnumFeature &feature : features)
    os << "  ::bishengir::npu::" << feature.cppName << " " << feature.fieldName
       << ";\n";
  os << R"(};

inline const ArchitectureFeatureProfile *
lookupArchitectureFeatureProfile(NPUArchitectureKind architecture) {
)";
  auto profiles =
      records.getAllDerivedDefinitions("ArchitectureFeatureProfile");
  for (const Record *profile : profiles) {
    const Record *architecture = profile->getValueAsDef("Architecture");
    std::string enumerator = getNPUArchitectureEnumerator(architecture);
    os << "  static constexpr ArchitectureFeatureProfile " << enumerator
       << "Profile{\n      NPUArchitectureKind::" << enumerator;
    DenseMap<const Record *, const Record *> settings;
    for (const Record *setting : profile->getValueAsListOfDefs("Features"))
      settings[setting->getValueAsDef("Feature")] = setting;
    for (const EnumFeature &feature : features)
      os << ",\n      " << feature.cppName << "::k"
         << settings.lookup(feature.record)->getValueAsString("CppName");
    os << "};\n";
  }
  os << "  switch (architecture) {\n";
  for (const Record *profile : profiles) {
    std::string enumerator =
        getNPUArchitectureEnumerator(profile->getValueAsDef("Architecture"));
    os << "  case NPUArchitectureKind::" << enumerator << ": return &"
       << enumerator << "Profile;\n";
  }
  os << R"(  case NPUArchitectureKind::kUnknown: return nullptr;
  }
  return nullptr;
}

)";
  for (const EnumFeature &feature : features) {
    os << "inline " << feature.cppName << " " << feature.accessor
       << "(NPUArchitectureKind architecture) {\n"
       << "  const ArchitectureFeatureProfile *profile =\n"
       << "      lookupArchitectureFeatureProfile(architecture);\n"
       << "  return profile ? profile->" << feature.fieldName << " : "
       << feature.cppName << "::kUnknown;\n}\n\n";
  }
  os << R"(} // namespace npu
} // namespace bishengir

#endif // BISHENGIR_NPU_ARCHITECTURE_FEATURES_H
)";
  return false;
}

static mlir::GenRegistration genNPUArchitectureFeatureDecls(
    "gen-npu-architecture-feature-decls",
    "Generate NPU architecture feature declarations and queries",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitNPUArchitectureFeatureDecls(records, os);
    });

} // namespace
