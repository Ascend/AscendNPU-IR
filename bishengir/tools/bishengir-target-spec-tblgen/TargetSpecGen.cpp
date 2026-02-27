//===- TargetSpecGen.cpp - Target Device Spec generator -------------------===//
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
// TargetSpecGen generates target device specifications.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {
static const char *const startOfHeaderGuard = R"(
#ifndef BISHENGIR_TARGET_SPEC_H
#define BISHENGIR_TARGET_SPEC_H
)";

static const char *const endOfHeaderGuard = R"(
#endif // BISHENGIR_TARGET_SPEC_H
)";

static const char *const startOfNameSpace = R"(
namespace mlir {
namespace hacc {
)";

static const char *const endOfNameSpace = R"(
} // namespace hacc
} // namespace mlir
)";

static const char *const wrapAttrDefs = R"(
template<typename T>
Attribute wrap(T rawVal, OpBuilder& builder) {
  llvm_unreachable("Unimplemented");
}

template<>
Attribute wrap(int rawVal, OpBuilder& builder) {
  return builder.getI32IntegerAttr(rawVal);
}
)";

static const char *const getTargetSpecDef = R"(
inline std::optional<const TargetSpec *> getTargetSpec(TargetDevice target) {
  for (const auto &spec : specs) {
    if (spec.device == target) return &spec;
  }
  return std::nullopt;
}
)";

static const char *const getSpecEntryDecls = R"(
  Attribute getSpecEntry(DeviceSpec spec, OpBuilder& builder) const;
)";

static const char *const symbolizeEnumDeclStr = R"(
{0} symbolize{0}Enum(::llvm::StringRef);
)";

static const char *const stringifyEnumDeclStr = R"(
::llvm::StringRef stringify{0}Enum({0});
)";
} // namespace

//===----------------------------------------------------------------------===//
// Target Specification Declaration
//===----------------------------------------------------------------------===//

/// Get the \c RecordVal from the input record.
///
/// For example, given the record:
/// ```tablegen
/// def Ascend910B1Spec {   // TargetSpec Ascend910bBaseSpec
///  string Name = "Ascend910B1";
///  int AiCoreCount = 24;
/// }
/// ```
/// Will return:
///  {string Name = "Ascend910B1";},
///  {int AiCoreCount = 24;},
static SmallVector<RecordVal>
getSpecSuperClassEntries(const Record *derivedClassRecord) {
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  SmallVector<Record *> superClasses;
#else
  SmallVector<const Record *> superClasses;
#endif
  derivedClassRecord->getDirectSuperClasses(superClasses);
  auto *superClass = superClasses.front();
  SmallVector<RecordVal> result = llvm::to_vector(superClass->getValues());
  return result;
}

/// Main entry to emit target spec decls.
static bool emitTargetSpecDecls(const llvm::RecordKeeper &records,
                                llvm::raw_ostream &OS) {
  auto specs = records.getAllDerivedDefinitions("TargetSpec");
  llvm::emitSourceFileHeader("Target Spec Declarations", OS, records);
  if (specs.empty())
    return false;

  // Emit start of namespace and header guard.
  OS << startOfHeaderGuard;
  OS << startOfNameSpace;

  // Generate TargetDevice enum class.
  OS << "enum class TargetDevice {\n";
  for (auto *spec : specs) {
    OS << "  " << spec->getValueAsString("Name") << ",\n";
  }
  OS << "  Unknown\n";
  OS << "};\n\n";

  // Generate function declarations to convert between string and TargetDevice
  // type.
  OS << formatv(symbolizeEnumDeclStr, "TargetDevice");
  OS << formatv(stringifyEnumDeclStr, "TargetDevice");
  OS << "\n";

  auto superClassEntry = getSpecSuperClassEntries(specs.front());
  // Generate TargetSpec struct declaration.
  OS << "struct TargetSpec {\n";
  OS << "  "
     << "TargetDevice device;\n";
  for (const RecordVal &specRecord : superClassEntry) {
    if (specRecord.getName() == "Name")
      continue;

    // Here, we directly use the tablegen type as the c++ type. So only
    // `int` and `string` type will be valid for now.
    // TODO: find another way to map basic spec type to c++ type.
    if (!specRecord.isTemplateArg()) {
      auto recordKind = specRecord.getType()->getRecTyKind();
      if (recordKind != RecTy::IntRecTyKind &&
          recordKind != RecTy::StringRecTyKind) {
        PrintError(specRecord.getLoc(),
                   Twine("Unsupported spec type to map to c++ type: ") +
                       specRecord.getPrintType());
        return true;
      }
      OS << "  " << specRecord.getPrintType() << " " << specRecord.getName()
         << ";\n";
    }
  }
  OS << "\n";
  // Generate class function declaration.
  OS << "public:";
  OS << getSpecEntryDecls;
  OS << "};\n\n";

  // Emit end of namespace and header guard.
  OS << endOfNameSpace;
  OS << endOfHeaderGuard;
  return false;
}

//===----------------------------------------------------------------------===//
// Target Specification Definitions
//===----------------------------------------------------------------------===//

static std::string emitSpecEntryBodyCase(const std::string &deviceSpecCase,
                                         const std::string &deviceSpecVar) {
  return formatv(R"(
  if (specEntry == DeviceSpec::{0}) {
    return wrap({1}, builder);
  })",
                 deviceSpecCase, deviceSpecVar);
}

/// Emit a function to map \c DeviceSpec entry to MLIR Attribute.
///
/// For example, given the record:
///  {int AiCoreCount = 24;},
///
/// Generate a function like:
/// ```cpp
/// Attribute TargetSpec::getSpecEntry(DeviceSpec specEntry,
///                                     OpBuilder& builder) const {
///   if (specEntry == DeviceSpec::AI_CORE_COUNT) {
///    return wrap(AiCoreCount, builder);
///   }
///   return Attribute();
/// }
/// ```
static void emitGetSpecEntryFnDef(llvm::raw_ostream &OS,
                                  ArrayRef<RecordVal> targetSpecClassRecord) {
  OS << R"(
Attribute TargetSpec::getSpecEntry(DeviceSpec specEntry, OpBuilder& builder) const {
)";

  for (const RecordVal &specRecord : targetSpecClassRecord) {
    if (specRecord.getName() == "Name")
      continue;

    if (specRecord.isTemplateArg())
      continue;

    // The `DeviceSpec` enum class is defined in `HACCAttrs.td`, the enum case
    // is defined as snake case.
    // We require the record name defined in the `TargetSpec.td` as a camel
    // case, so we map it to snake case.
    auto recordName = specRecord.getName().str();
    auto recordSpecEnumName = convertToSnakeFromCamelCase(recordName);
    llvm::transform(recordSpecEnumName, recordSpecEnumName.begin(),
                    llvm::toUpper);
    OS << emitSpecEntryBodyCase(recordSpecEnumName, recordName);
  }
  OS << "\n";
  OS << "  return Attribute();";
  OS << "\n}\n\n";
}

/// Emit a function to map string to \c DeviceTarget enum.
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
static void emitStrToSymFnForDeviceTarget(const std::vector<Record *> &records,
                                          raw_ostream &OS) {
#else
static void emitStrToSymFnForDeviceTarget(const std::vector<const Record *> &records,
                                          raw_ostream &OS) {
#endif
  const auto *enumName = "TargetDevice";
  OS << formatv("{0} symbolize{0}Enum(::llvm::StringRef str){{\n", enumName);
  OS << formatv("  return ::llvm::StringSwitch<{0}>(str)\n", enumName);
  for (auto [idx, record] : llvm::enumerate(records)) {
    auto deviceName = record->getValueAsString("Name");
    OS << formatv("      .Case(\"{1}\", {0}::{2})\n", enumName, deviceName,
                  deviceName);
  }
  OS << "      .Default(TargetDevice::Unknown);\n";
  OS << "}\n\n";
}

/// Emit a function to map \c DeviceTarget enum to string.
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
static void emitSymToStrFnForDeviceTarget(const std::vector<Record *> &records,
                                          raw_ostream &OS) {
#else
static void emitSymToStrFnForDeviceTarget(const std::vector<const Record *> &records,
                                          raw_ostream &OS) {
#endif
  const auto *enumName = "TargetDevice";
  OS << formatv("::llvm::StringRef stringify{0}Enum({0} val){{\n", enumName);
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  OS << formatv("  switch (val) {{\n", enumName);
#else
  OS << "  switch (val) {\n";
#endif
  for (auto [idx, record] : llvm::enumerate(records)) {
    auto deviceName = record->getValueAsString("Name");
    OS << formatv("    case {0}::{1}: return \"{2}\";\n", enumName, deviceName,
                  deviceName);
  }
  OS << "  }\n";
  OS << "  return \"\";\n";
  OS << "}\n\n";
}

/// Main entry to emit target spec defs.
static bool emitTargetSpecDefs(const llvm::RecordKeeper &records,
                               llvm::raw_ostream &OS) {
  auto specs = records.getAllDerivedDefinitions("TargetSpec");
  llvm::emitSourceFileHeader("Target Spec Declarations", OS, records);

  if (specs.empty())
    return false;

  OS << "#include \"NPUTargetSpec.h.inc\" \n\n";

  // Emit start of namespace.
  OS << startOfNameSpace;

  // Emit static const array to hold all the spec entries
  auto superClassEntry = getSpecSuperClassEntries(specs.front());
  OS << "static const TargetSpec specs[] = {\n";
  for (auto *spec : specs) {
    OS << "  {\n";
    for (const RecordVal &specRecord : superClassEntry) {
      if (specRecord.getName() == "Name") {
        OS << "    TargetDevice::" << spec->getValueAsString("Name");
        OS << ",\n";
        continue;
      }

      if (specRecord.isTemplateArg())
        continue;

      switch (specRecord.getType()->getRecTyKind()) {
      case RecTy::IntRecTyKind:
        OS << "    " << spec->getValueAsInt(specRecord.getName());
        break;
      case RecTy::StringRecTyKind:
        OS << "    \"" << spec->getValueAsString(specRecord.getName()) << "\"";
        break;
      default:
        PrintError(specRecord.getLoc(), Twine("Unsupported spec type: ") +
                                            specRecord.getPrintType());
        return true;
      }
      OS << ",\n";
    }

    OS << "  },\n";
  }
  OS << "};\n\n";

  // Emit a function to get the spec by target.
  OS << getTargetSpecDef;

  // Emit functions to convert spec to attributes.
  OS << wrapAttrDefs;
  emitGetSpecEntryFnDef(OS, superClassEntry);

  // Emit functions to convert between string device target and enum.
  emitStrToSymFnForDeviceTarget(specs, OS);
  emitSymToStrFnForDeviceTarget(specs, OS);

  // Emit end of namespace.
  OS << endOfNameSpace;
  return false;
}

// Registers the generator to bishengir-target-spec-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-target-spec-decls", "Generate target spec declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitTargetSpecDecls(records, os);
                 });

static mlir::GenRegistration
    genEnumDefs("gen-target-spec-defs", "Generate target spec definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitTargetSpecDefs(records, os);
                });
