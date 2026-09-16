//===- TargetSpecGen.cpp - Target Device Spec generator -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

constexpr StringLiteral kName = "Name";
constexpr StringLiteral kArch = "Arch";

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

template<>
Attribute wrap(std::string rawVal, OpBuilder& builder) {
  return builder.getStringAttr(rawVal);
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
std::string stringify{0}Enum({0} dev);
)";

static const char *const getArcDeclStr = R"(
std::string getArch({0} dev);
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
  SmallVector<Record *> superClasses;
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
  // Required by some var decls
  OS << "#include <string>\n";
  OS << "#include <array>\n";
  OS << startOfNameSpace;

  // Generate TargetDevice enum class.
  OS << "enum class TargetDevice {\n";
  for (auto *spec : specs) {
    OS << "  " << spec->getValueAsString(kName) << ",\n";
  }
  OS << "  Unknown\n";
  OS << "};\n\n";

  OS << formatv("constexpr std::array<TargetDevice, {0}> targetDevices = {{\n",
                specs.size());
  {
    auto it = specs.begin();
    OS << "  TargetDevice::" << (*it)->getValueAsString(kName);
    for (++it; it != specs.end(); ++it) {
      OS << ",\n  TargetDevice::" << (*it)->getValueAsString(kName);
    }
  }
  OS << "\n};\n\n";

  // Generate a function declaration to convert string to TargetDevice type.
  OS << formatv(symbolizeEnumDeclStr, "TargetDevice");
  OS << "\n";

  // Generate a function declaration to convert TargetDevice enum to string.
  OS << formatv(stringifyEnumDeclStr, "TargetDevice");
  OS << "\n";

  // Generate a function declaration to get the arch string by device target
  // enum.
  OS << formatv(getArcDeclStr, "TargetDevice");
  OS << "\n";

  auto superClassEntry = getSpecSuperClassEntries(specs.front());
  // Generate TargetSpec struct declaration.
  OS << "struct TargetSpec {\n";
  OS << "  "
     << "TargetDevice device;\n";
  for (const RecordVal &specRecord : superClassEntry) {
    auto name = specRecord.getName();
    auto printType = specRecord.getPrintType();
    if (name == kName)
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
                       printType);
        return true;
      }
      // Add std namespace for convenience
      if (recordKind == RecTy::StringRecTyKind)
        printType = "std::" + printType;
      OS << "  " << printType << " " << name << ";\n";
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
    if (specRecord.getName() == kName)
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
static void emitStrToSymFnForDeviceTarget(const std::vector<Record *> &records,
                                          raw_ostream &OS) {
  const auto *enumName = "TargetDevice";
  OS << formatv("{0} symbolize{0}Enum(::llvm::StringRef str){{\n", enumName);
  OS << formatv("  return ::llvm::StringSwitch<{0}>(str)\n", enumName);
  for (auto [idx, record] : llvm::enumerate(records)) {
    auto deviceName = record->getValueAsString(kName);

    OS << formatv("      .Case(\"{1}\", {0}::{2})\n", enumName, deviceName,
                  deviceName);
  }
  OS << "      .Default(TargetDevice::Unknown);\n";
  OS << "}\n\n";
}

/// Emit a function to map \c DeviceTarget enum to string.
static void emitSymToStrFnForDeviceTarget(const std::vector<Record *> &records,
                                          raw_ostream &OS) {
  const auto *enumName = "TargetDevice";
  OS << formatv("std::string stringify{0}Enum({0} dev){{\n", enumName);
  OS << formatv("  switch (dev) {{\n", enumName);
  for (auto [idx, record] : llvm::enumerate(records)) {
    auto deviceName = record->getValueAsString(kName);
    OS << formatv("    case {0}::{1}: return \"{1}\";\n", enumName, deviceName);
  }
  OS << formatv("    case {0}::Unknown: return \"Unknown\";\n", enumName);
  OS << "  }\n";
  OS << formatv("  llvm_unreachable(\"Unknown {0} value\");\n", enumName);
  OS << "}\n\n";
}

/// Emit a function to map \c DeviceTarget enum to the Arch string.
static void emitDeviceTargetToArch(const std::vector<Record *> &records,
                                   raw_ostream &OS) {
  const auto *enumName = "TargetDevice";
  OS << formatv("std::string getArch({0} dev){{\n", enumName);
  OS << formatv("  switch (dev) {{\n", enumName);
  for (auto [idx, record] : llvm::enumerate(records)) {
    auto deviceName = record->getValueAsString(kName);
    auto archString = record->getValueAsOptionalString(kArch);
    if (!archString.has_value())
      continue;
    OS << formatv("    case {0}::{1}: return \"{2}\";\n", enumName, deviceName,
                  archString.value());
  }
  OS << formatv(
      "    case {0}::Unknown: llvm_unreachable(\"Unknown {0} value\");\n",
      enumName);
  OS << "  }\n";
  OS << formatv("  llvm_unreachable(\"Unknown {0} value\");\n", enumName);
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
      if (specRecord.getName() == kName) {
        OS << "    TargetDevice::" << spec->getValueAsString(kName);
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

  // Emit a function to convert string device target to enum.
  emitStrToSymFnForDeviceTarget(specs, OS);

  // Emit a function to convert enum device target to string.
  emitSymToStrFnForDeviceTarget(specs, OS);

  // Emit a function to get the arch string by device target enum.
  emitDeviceTargetToArch(specs, OS);

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
