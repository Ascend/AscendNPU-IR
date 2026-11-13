//===- MmadCapabilityBindingGen.cpp - MMAD capability binding -----------===//
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
// MMAD binding backend. The shared capability generator does not know about
// MMAD's optional bias, format, or K-alignment projections.

#include "NPUArchitectureGen.h"

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <algorithm>

using namespace llvm;

namespace {

constexpr StringLiteral kCppName = "CppName";
constexpr StringLiteral kName = "Name";
constexpr StringLiteral kDomain = "Domain";
constexpr StringLiteral kEnumAttr = "EnumAttr";
constexpr StringLiteral kField = "Field";
constexpr StringLiteral kFields = "Fields";
constexpr StringLiteral kSchema = "Schema";
constexpr StringLiteral kSlot = "Slot";
constexpr StringLiteral kSlots = "Slots";
constexpr StringLiteral kProperty = "Property";
constexpr StringLiteral kProperties = "Properties";
constexpr StringLiteral kParameters = "Parameters";
constexpr StringLiteral kValues = "Values";
constexpr StringLiteral kTypes = "Types";
constexpr StringLiteral kCapability = "Capability";
constexpr StringLiteral kConstraints = "Constraints";
constexpr StringLiteral kCases = "Cases";
constexpr StringLiteral kCapabilities = "Capabilities";
constexpr StringLiteral kArchitecture = "Architecture";
constexpr StringLiteral kFieldKind = "FieldKind";
constexpr StringLiteral kBoolAttr = "BoolAttr";
constexpr StringLiteral kValue = "Value";

static const Record *getEnumDomainRecord(const Record *record) {
  if (record->isSubClassOf("EnumAttrCapabilityDomain"))
    return record;
  return nullptr;
}

static bool isBoolDomain(const Record *record) {
  return record->isSubClassOf("BoolAttrCapabilityDomain");
}

static bool isTypeDomain(const Record *record) {
  return record->isSubClassOf("MLIRTypeCapabilityDomain");
}

// Named records derive each omitted name independently from the TD record.
// In particular, Name derives from the record even when CppName is overridden.
// Only anonymous ODS carrier-domain wrappers use the carrier's stable identity;
// their synthesized TableGen names are never exposed as public API names.
static StringRef getEffectiveCppName(const Record *record) {
  StringRef authoredName = record->getValueAsString(kCppName);
  if (!authoredName.empty())
    return authoredName;
  if (!record->isAnonymous())
    return record->getName();

  if (isBoolDomain(record))
    return "Bool";

  const Record *domain = getEnumDomainRecord(record);
  if (!domain)
    return {};
  const Record *attr = domain->getValueAsDef(kEnumAttr);
  if (!attr->isSubClassOf("AttrDef"))
    return {};
  mlir::tblgen::AttrDef attrDef(attr);
  auto parameters = attrDef.getParameters();
  if (parameters.size() != 1)
    return {};
  return parameters.front().getCppType().rsplit("::").second;
}

static std::string getEffectiveName(const Record *record) {
  StringRef authoredName = record->getValueAsString(kName);
  if (!authoredName.empty())
    return authoredName.str();
  if (!record->isAnonymous())
    return llvm::convertToSnakeFromCamelCase(record->getName());

  if (isBoolDomain(record))
    return "bool";

  const Record *domain = getEnumDomainRecord(record);
  if (!domain)
    return {};
  const Record *attr = domain->getValueAsDef(kEnumAttr);
  if (!attr->isSubClassOf("AttrDef"))
    return {};
  mlir::tblgen::AttrDef attrDef(attr);
  std::optional<StringRef> mnemonic = attrDef.getMnemonic();
  return mnemonic ? mnemonic->str() : std::string();
}

static bool isValidCppIdentifier(StringRef name);

// A catalogue with no CapabilityEmissionScope stays in bishengir::npu and
// keeps the original include guards. A scope puts that catalogue in
// bishengir::npu::<CppNamespace> so another catalogue can link beside it.
struct CapabilityEmissionScope {
  std::string extraNamespace;
  std::string declGuard = "BISHENGIR_TARGET_CAPABILITY_H";
  std::string bindingGuard = "BISHENGIR_TARGET_CAPABILITY_BINDING_H";
};

static bool parseEmissionScope(const RecordKeeper &records,
                               CapabilityEmissionScope &scope) {
  auto defs = records.getAllDerivedDefinitions("CapabilityEmissionScope");
  if (defs.empty())
    return false;
  if (defs.size() != 1) {
    PrintFatalError(defs.front()->getLoc(),
                    "expected at most one CapabilityEmissionScope");
    return true;
  }
  const Record *def = defs.front();
  scope.extraNamespace = def->getValueAsString("CppNamespace").str();
  scope.declGuard = def->getValueAsString("DeclGuard").str();
  scope.bindingGuard = def->getValueAsString("BindingGuard").str();
  if (!isValidCppIdentifier(scope.extraNamespace) || scope.declGuard.empty() ||
      scope.bindingGuard.empty()) {
    PrintFatalError(def->getLoc(),
                    "CapabilityEmissionScope requires an identifier namespace "
                    "and both include guards");
    return true;
  }
  return false;
}

static void emitIncludeGuardOpen(raw_ostream &os, StringRef guard) {
  os << "\n#ifndef " << guard << "\n#define " << guard << "\n";
}

static void emitNamespaceOpen(raw_ostream &os,
                              const CapabilityEmissionScope &scope,
                              bool declareArchitecture) {
  os << "\nnamespace bishengir {\nnamespace npu {\n";
  // The definition header included by the .cpp fragment may already have
  // defined this enum. Only the declaration header forward-declares it, and
  // it stays in bishengir::npu when an extra catalogue namespace is used.
  if (declareArchitecture)
    os << "enum class NPUArchitectureKind;\n";
  if (!scope.extraNamespace.empty())
    os << "namespace " << scope.extraNamespace << " {\n";
  os << "\n";
}

static void emitNamespaceClose(raw_ostream &os,
                               const CapabilityEmissionScope &scope) {
  os << "\n";
  if (!scope.extraNamespace.empty())
    os << "} // namespace " << scope.extraNamespace << "\n";
  os << "} // namespace npu\n} // namespace bishengir\n";
}

static void emitIncludeGuardClose(raw_ostream &os, StringRef guard) {
  os << "\n#endif // " << guard << "\n";
}

static bool isValidCppIdentifier(StringRef name) {
  if (name.empty() || (!llvm::isAlpha(name.front()) && name.front() != '_') ||
      name.contains("__") ||
      (name.size() > 1 && name.front() == '_' && llvm::isUpper(name[1])))
    return false;
  if (!llvm::all_of(name.drop_front(),
                    [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return false;
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
              .Cases("goto", "if", "inline", "int", "long", true)
              .Cases("mutable", "namespace", "new", "noexcept", "not", true)
              .Cases("not_eq", "nullptr", "operator", "or", "or_eq", true)
              .Cases("private", "protected", "public", "register", true)
              .Cases("reinterpret_cast", "requires", "return", "short", true)
              .Cases("signed", "sizeof", "static", "static_assert", true)
              .Cases("static_cast", "struct", "switch", "template", true)
              .Cases("this", "thread_local", "throw", "true", "try", true)
              .Cases("typedef", "typeid", "typename", "union", "unsigned", true)
              .Cases("using", "virtual", "void", "volatile", "wchar_t", true)
              .Cases("while", "xor", "xor_eq", true)
              .Default(false);
}

static bool hasUniqueCppNames(ArrayRef<const Record *> records,
                              StringRef recordKind) {
  StringSet<> names;
  bool hadError = false;
  for (const Record *record : records) {
    bool anonymousCarrier =
        record->isAnonymous() &&
        (isBoolDomain(record) || getEnumDomainRecord(record));
    if (record->isAnonymous() && !anonymousCarrier &&
        (record->getValueAsString(kCppName).empty() ||
         record->getValueAsString(kName).empty())) {
      PrintError(record->getLoc(), Twine("anonymous ") + recordKind +
                                       " requires explicit CppName and Name");
      hadError = true;
      continue;
    }
    StringRef cppName = getEffectiveCppName(record);
    if (cppName.empty()) {
      PrintError(record->getLoc(),
                 Twine(recordKind) + " must have a non-empty CppName");
      hadError = true;
      continue;
    }
    if (!isValidCppIdentifier(cppName)) {
      PrintError(record->getLoc(),
                 Twine("invalid ") + recordKind + " CppName: " + cppName);
      hadError = true;
    }
    if (getEffectiveName(record).empty()) {
      PrintError(record->getLoc(),
                 Twine(recordKind) + " must have a non-empty Name");
      hadError = true;
    }
    if (!names.insert(cppName).second) {
      PrintError(record->getLoc(),
                 Twine("duplicate ") + recordKind + " CppName: " + cppName);
      hadError = true;
    }
  }
  return hadError;
}

static bool hasUniqueNames(ArrayRef<const Record *> records,
                           StringRef recordKind) {
  StringSet<> names;
  bool hadError = false;
  for (const Record *record : records) {
    std::string name = getEffectiveName(record);
    if (name.empty() || names.insert(name).second)
      continue;
    PrintError(record->getLoc(),
               Twine("duplicate ") + recordKind + " Name: " + name);
    hadError = true;
  }
  return hadError;
}

template <typename RecordVector>
static SmallVector<const Record *> asConstRecords(const RecordVector &records) {
  SmallVector<const Record *> result;
  result.reserve(records.size());
  for (const Record *record : records)
    result.push_back(record);
  return result;
}

// Profiles are optional: generic schema files need not include NPU records.
static SmallVector<const Record *> getDefinitions(const RecordKeeper &records,
                                                  StringRef className) {
  if (!records.getClass(className))
    return {};
  return asConstRecords(records.getAllDerivedDefinitions(className));
}

// Independently created anonymous carrier wrappers can share one ODS carrier.
// Deduplicate only those equivalent wrappers. Named domains retain their own
// identities, and conflicting declarations stay visible to validation.
static SmallVector<const Record *>
getCapabilityValueDomains(const RecordKeeper &records) {
  SmallVector<const Record *> result;
  for (const Record *domain :
       getDefinitions(records, "CapabilityValueDomain")) {
    StringRef carrierField = domain->isSubClassOf("EnumAttrCapabilityDomain")
                                 ? kEnumAttr
                                 : kBoolAttr;
    if (domain->isAnonymous() && domain->getValue(carrierField) &&
        llvm::any_of(result, [&](const Record *other) {
          return other->isAnonymous() && other->getValue(carrierField) &&
                 other->getValueAsDef(carrierField) ==
                     domain->getValueAsDef(carrierField) &&
                 getEffectiveCppName(other) == getEffectiveCppName(domain) &&
                 getEffectiveName(other) == getEffectiveName(domain);
        }))
      continue;
    result.push_back(domain);
  }
  return result;
}

static bool isUnset(const Record *record, StringRef field) {
  return isa<UnsetInit>(record->getValueInit(field));
}

static bool isEnumDomain(const Record *domain) {
  return domain->isSubClassOf("EnumAttrCapabilityDomain");
}

static const Record *getDomainCarrierAttr(const Record *domain) {
  return domain->getValueAsDef(kEnumAttr);
}

static const Record *findDomainODSEnum(const RecordKeeper &records,
                                       const Record *domain) {
  if (!isEnumDomain(domain))
    return nullptr;

  const Record *attr = getDomainCarrierAttr(domain);
  if (!attr->isSubClassOf("AttrDef"))
    return nullptr;
  mlir::tblgen::AttrDef attrDef(attr);
  auto parameters = attrDef.getParameters();
  if (parameters.size() != 1)
    return nullptr;

  StringRef parameterType = parameters.front().getCppType();
  const Record *match = nullptr;
  for (const Record *odsEnum :
       records.getAllDerivedDefinitions("EnumAttrInfo")) {
    if (odsEnum->getValueAsString("cppType") != parameterType)
      continue;
    if (match)
      return nullptr;
    match = odsEnum;
  }
  return match;
}

static void appendKeyComponent(std::string &key, StringRef component) {
  key.append(utostr(component.size()));
  key.push_back(':');
  key.append(component);
}

static std::string getSlotFieldKey(const Record *slot, const Record *property) {
  std::string key = "slot:";
  appendKeyComponent(key, slot->getName());
  appendKeyComponent(key, property->getName());
  return key;
}

static std::string getRequestFieldKey(const Record *field) {
  StringRef fieldKind = field->getValueAsString(kFieldKind);
  if (fieldKind == "slot" && field->isSubClassOf("SlotField"))
    return getSlotFieldKey(field->getValueAsDef(kSlot),
                           field->getValueAsDef(kProperty));
  if (fieldKind == "parameter" && field->isSubClassOf("CapabilityParameter"))
    return (Twine("parameter:") + field->getName()).str();
  return (Twine("unknown:") + field->getName()).str();
}

static const Record *getCapabilitySchema(const Record *capability) {
  return capability->getValueAsDef(kSchema);
}

static SmallVector<const Record *>
getCapabilitySlots(const Record *capability) {
  return asConstRecords(
      getCapabilitySchema(capability)->getValueAsListOfDefs(kSlots));
}

static SmallVector<const Record *>
getCapabilityParameters(const Record *capability) {
  return asConstRecords(
      getCapabilitySchema(capability)->getValueAsListOfDefs(kParameters));
}

static int64_t getSlotPosition(const Record *capability, const Record *field) {
  auto slots = getCapabilitySlots(capability);
  const Record *slot = field->getValueAsDef(kSlot);
  auto it = llvm::find(slots, slot);
  if (it == slots.end())
    PrintFatalError(field->getLoc(),
                    "cannot emit a slot field outside the validated schema");
  return std::distance(slots.begin(), it);
}

static StringSet<> getCapabilitySchemaFieldKeys(const Record *capability) {
  StringSet<> keys;
  for (const Record *slot : getCapabilitySlots(capability))
    for (const Record *property : slot->getValueAsListOfDefs(kProperties))
      keys.insert(getSlotFieldKey(slot, property));
  for (const Record *parameter : getCapabilityParameters(capability))
    keys.insert((Twine("parameter:") + parameter->getName()).str());
  return keys;
}

static const Record *getRequestFieldDomain(const Record *field) {
  StringRef fieldKind = field->getValueAsString(kFieldKind);
  if (fieldKind == "slot" && field->isSubClassOf("SlotField")) {
    return field->getValueAsDef(kProperty)->getValueAsDef(kDomain);
  }
  if (fieldKind == "parameter" && field->isSubClassOf("CapabilityParameter"))
    return field->getValueAsDef(kDomain);
  return nullptr;
}

static bool isTypeRequestField(const Record *field) {
  const Record *domain = getRequestFieldDomain(field);
  return domain && isTypeDomain(domain);
}

static StringSet<> getCapabilityTypeFieldKeys(const Record *capability) {
  StringSet<> keys;
  for (const Record *slot : getCapabilitySlots(capability))
    for (const Record *property : slot->getValueAsListOfDefs(kProperties))
      if (isTypeDomain(property->getValueAsDef(kDomain)))
        keys.insert(getSlotFieldKey(slot, property));
  return keys;
}

static std::string getTypePredicate(const Record *type) {
  if (isUnset(type, "predicate"))
    return {};
  return mlir::tblgen::Constraint(type).getConditionTemplate();
}

// Identical predicate spellings are duplicates even when their ODS records
// have different names. This does not attempt to prove logical equivalence
// between arbitrary C++ predicates. Length prefixes keep the key unambiguous.
static std::string getTypeSignatureKey(const Record *signature) {
  SmallVector<std::string> fieldPredicates;
  auto fields = signature->getValueAsListOfDefs(kFields);
  auto types = signature->getValueAsListOfDefs(kTypes);
  for (auto [field, type] : llvm::zip(fields, types)) {
    std::string pair;
    appendKeyComponent(pair, getRequestFieldKey(field));
    appendKeyComponent(pair, getTypePredicate(type));
    fieldPredicates.push_back(std::move(pair));
  }
  llvm::sort(fieldPredicates);
  std::string key;
  for (const std::string &pair : fieldPredicates)
    appendKeyComponent(key, pair);
  return key;
}

static std::string getValueConstraintKey(const Record *constraint) {
  if (constraint->isSubClassOf("AnyValue"))
    return (Twine(getRequestFieldKey(constraint->getValueAsDef(kField))) + "=*")
        .str();

  if (constraint->isSubClassOf("BoolValueConstraint"))
    return (Twine(getRequestFieldKey(constraint->getValueAsDef(kField))) + "=" +
            (constraint->getValueAsBit(kValue) ? "true" : "false"))
        .str();

  SmallVector<int64_t> rawValues;
  for (const Record *value : constraint->getValueAsListOfDefs(kValues))
    rawValues.push_back(value->getValueAsInt("value"));
  llvm::sort(rawValues);
  rawValues.erase(std::unique(rawValues.begin(), rawValues.end()),
                  rawValues.end());

  std::string key = getRequestFieldKey(constraint->getValueAsDef(kField));
  key.push_back('=');
  for (int64_t value : rawValues) {
    key.append(itostr(value));
    key.push_back(',');
  }
  return key;
}

// SupportCase is an unordered disjunction of type signatures conjoined with
// an unordered set of per-field constraints. Normalize both collections so
// duplicate rows are rejected even when authors use different record names or
// list orders.
static std::string getNormalizedSupportCaseKey(const Record *supportCase) {
  SmallVector<std::string> signatureKeys;
  SmallVector<std::string> valueConstraintKeys;
  for (const Record *constraint :
       supportCase->getValueAsListOfDefs(kConstraints)) {
    if (constraint->isSubClassOf("TypeSignature")) {
      // An explicit empty signature and an omitted signature both mean true
      // when the schema has no remaining type fields to constrain.
      if (!constraint->getValueAsListOfDefs(kFields).empty() ||
          !constraint->getValueAsListOfDefs(kTypes).empty())
        signatureKeys.push_back(getTypeSignatureKey(constraint));
    } else if (constraint->isSubClassOf("EnumValueConstraint") ||
               constraint->isSubClassOf("BoolValueConstraint") ||
               constraint->isSubClassOf("AnyValue"))
      valueConstraintKeys.push_back(getValueConstraintKey(constraint));
  }
  llvm::sort(signatureKeys);
  signatureKeys.erase(std::unique(signatureKeys.begin(), signatureKeys.end()),
                      signatureKeys.end());
  llvm::sort(valueConstraintKeys);

  std::string key = supportCase->getValueAsDef(kCapability)->getName().str();
  key.append(";signatures=");
  for (const std::string &signatureKey : signatureKeys) {
    appendKeyComponent(key, signatureKey);
  }
  key.append(";constraints=");
  for (const std::string &constraintKey : valueConstraintKeys) {
    appendKeyComponent(key, constraintKey);
  }
  return key;
}

static bool validateCapabilityRecords(const RecordKeeper &records) {
  auto valueDomains = getCapabilityValueDomains(records);
  auto properties = getDefinitions(records, "CapabilitySlotProperty");
  auto parameters =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityParameter"));
  auto slots =
      asConstRecords(records.getAllDerivedDefinitions("CapabilitySlot"));
  auto capabilities =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityKey"));
  auto supportCases =
      asConstRecords(records.getAllDerivedDefinitions("SupportCase"));
  auto completenesses =
      asConstRecords(records.getAllDerivedDefinitions("ProfileCompleteness"));
  auto coverages = getDefinitions(records, "ArchitectureCapabilityCoverage");
  auto profiles = getDefinitions(records, "ArchitectureCapabilityProfile");

  bool hadError = false;
  hadError |= hasUniqueCppNames(valueDomains, "CapabilityValueDomain");
  hadError |= hasUniqueCppNames(parameters, "CapabilityParameter");
  hadError |= hasUniqueCppNames(slots, "CapabilitySlot");
  hadError |= hasUniqueCppNames(properties, "CapabilitySlotProperty");
  hadError |= hasUniqueCppNames(capabilities, "CapabilityKey");
  hadError |= hasUniqueCppNames(completenesses, "ProfileCompleteness");
  hadError |= hasUniqueNames(valueDomains, "CapabilityValueDomain");
  hadError |= hasUniqueNames(completenesses, "ProfileCompleteness");

  // Check unused declarations too: every property has a domain regardless of
  // whether a schema or support case happens to reference it yet.
  for (const Record *property : properties)
    if (isUnset(property, kDomain))
      PrintFatalError(property->getLoc(),
                      "CapabilitySlotProperty requires a Domain");

  for (const Record *slot : slots) {
    SmallPtrSet<const Record *, 8> seenProperties;
    StringSet<> seenNames;
    for (const Record *property : slot->getValueAsListOfDefs(kProperties)) {
      if (seenProperties.insert(property).second &&
          seenNames.insert(getEffectiveName(property)).second)
        continue;
      PrintError(slot->getLoc(),
                 Twine("duplicate property '") + getEffectiveName(property) +
                     "' on semantic slot '" + getEffectiveName(slot) + "'");
      hadError = true;
    }
  }
  for (const Record *field :
       getDefinitions(records, "CapabilityRequestField")) {
    StringRef kind = field->getValueAsString(kFieldKind);
    if (field->isSubClassOf("SlotField") && kind == "slot") {
      const Record *slot = field->getValueAsDef(kSlot);
      const Record *property = field->getValueAsDef(kProperty);
      if (!llvm::is_contained(slot->getValueAsListOfDefs(kProperties),
                              property)) {
        PrintError(field->getLoc(), Twine("property '") + property->getName() +
                                        "' is not declared on slot '" +
                                        slot->getName() + "'");
        hadError = true;
      }
      continue;
    }
    if (field->isSubClassOf("CapabilityParameter") && kind == "parameter")
      continue;
    PrintError(field->getLoc(),
               "request field must be a SlotField or CapabilityParameter with "
               "its declared field kind");
    hadError = true;
  }

  // Validate every authored signature, including currently unused ones. ODS
  // constraints match the request's mlir::Type directly: no registration,
  // concrete-type builder, or unique type classification is needed.
  SmallPtrSet<const Record *, 16> checkedTypes;
  for (const Record *signature : getDefinitions(records, "TypeSignature")) {
    auto fields = signature->getValueAsListOfDefs(kFields);
    auto types = signature->getValueAsListOfDefs(kTypes);
    if (fields.size() != types.size()) {
      PrintError(signature->getLoc(),
                 "TypeSignature Fields and Types must have equal lengths");
      hadError = true;
    }
    StringSet<> seenFields;
    for (const Record *field : fields) {
      if (!isTypeRequestField(field)) {
        PrintError(field->getLoc(),
                   "TypeSignature must target an MLIRType-domain slot field");
        hadError = true;
      }
      if (!seenFields.insert(getRequestFieldKey(field)).second) {
        PrintError(signature->getLoc(), "duplicate field in TypeSignature");
        hadError = true;
      }
    }
    for (const Record *type : types) {
      if (!checkedTypes.insert(type).second)
        continue;
      std::string conditionTemplate = getTypePredicate(type);
      // CPred adds parentheses even to an empty authored expression.
      if (llvm::all_of(conditionTemplate, [](char c) {
            return llvm::isSpace(c) || c == '(' || c == ')';
          })) {
        PrintError(type->getLoc(),
                   "TypeSignature requires a non-empty ODS Type predicate");
        hadError = true;
        continue;
      }
      mlir::tblgen::FmtContext context;
      context.withSelf("type");
      std::string predicate =
          mlir::tblgen::tgfmt(conditionTemplate, &context).str();
      if (StringRef(predicate).contains("$")) {
        PrintError(type->getLoc(),
                   "TypeSignature predicate may only reference $_self");
        hadError = true;
      }
    }
  }

  // Validate every declaration, including aliases removed from the emitted
  // list.
  for (const Record *domain :
       getDefinitions(records, "CapabilityValueDomain")) {
    unsigned carrierCount = unsigned(isTypeDomain(domain)) +
                            unsigned(isEnumDomain(domain)) +
                            unsigned(isBoolDomain(domain));
    if (carrierCount != 1) {
      PrintError(domain->getLoc(),
                 "capability value domain must have exactly one MLIRType, "
                 "enum-attribute, or BoolAttr carrier");
      hadError = true;
      continue;
    }
    if (isTypeDomain(domain))
      continue;
    if (isEnumDomain(domain)) {
      const Record *attr = getDomainCarrierAttr(domain);
      if (!attr->isSubClassOf("AttrDef")) {
        PrintError(domain->getLoc(),
                   "EnumAttrCapabilityDomain requires an ODS AttrDef");
        hadError = true;
        continue;
      }
      mlir::tblgen::AttrDef attrDef(attr);
      auto attrParameters = attrDef.getParameters();
      const Record *odsEnum = findDomainODSEnum(records, domain);
      std::optional<StringRef> mnemonic = attrDef.getMnemonic();
      if (attrParameters.size() != 1 || !odsEnum || !mnemonic ||
          mnemonic->empty()) {
        PrintError(
            domain->getLoc(),
            "EnumAttrCapabilityDomain requires an attribute with exactly "
            "one ODS enum-typed parameter and a mnemonic");
        hadError = true;
        continue;
      }
      auto enumCases = odsEnum->getValueAsListOfDefs("enumerants");
      if (enumCases.empty()) {
        PrintError(domain->getLoc(),
                   "enum domain must have a non-empty vocabulary");
        hadError = true;
      }
      SmallSet<int64_t, 8> values;
      StringSet<> spellings;
      for (const Record *enumCase : enumCases) {
        int64_t value = enumCase->getValueAsInt("value");
        StringRef spelling = enumCase->getValueAsString("str");
        if (value < 0 || !values.insert(value).second) {
          PrintError(enumCase->getLoc(),
                     "enum domain requires distinct, non-negative raw values");
          hadError = true;
        }
        if (spelling.empty() || !spellings.insert(spelling).second) {
          PrintError(enumCase->getLoc(),
                     "enum domain requires distinct, non-empty spellings");
          hadError = true;
        }
      }
      continue;
    }

    if (isBoolDomain(domain)) {
      const Record *boolAttr = records.getDef(kBoolAttr);
      if (!boolAttr || domain->getValueAsDef(kBoolAttr) != boolAttr) {
        PrintError(domain->getLoc(),
                   "BoolAttrCapabilityDomain must reference the ODS BoolAttr");
        hadError = true;
      }
      continue;
    }
  }

  for (const Record *parameter : parameters) {
    if (!isTypeDomain(parameter->getValueAsDef(kDomain)))
      continue;
    PrintError(parameter->getLoc(),
               "capability parameters require an enum-attribute or BoolAttr "
               "domain; MLIRType domains are only supported on slot "
               "properties");
    hadError = true;
  }

  StringSet<> capabilityNames;
  for (const Record *capability : capabilities) {
    if (!capabilityNames.insert(getEffectiveName(capability)).second) {
      PrintError(capability->getLoc(), "duplicate CapabilityKey Name");
      hadError = true;
    }
  }
  for (const Record *schema : getDefinitions(records, "RequestSchema")) {
    StringSet<> seenSlotNames;
    SmallPtrSet<const Record *, 8> seenSlots;
    for (const Record *slot : schema->getValueAsListOfDefs(kSlots)) {
      if (!seenSlots.insert(slot).second ||
          !seenSlotNames.insert(getEffectiveName(slot)).second) {
        PrintError(slot->getLoc(), Twine("duplicate semantic slot '") +
                                       getEffectiveName(slot) +
                                       "' in capability schema");
        hadError = true;
      }
    }

    SmallPtrSet<const Record *, 8> seenParameters;
    StringSet<> seenParameterNames;
    for (const Record *parameter : schema->getValueAsListOfDefs(kParameters)) {
      if (seenParameters.insert(parameter).second &&
          seenParameterNames.insert(getEffectiveName(parameter)).second)
        continue;
      PrintError(parameter->getLoc(), Twine("duplicate parameter '") +
                                          getEffectiveName(parameter) +
                                          "' in capability schema");
      hadError = true;
    }
  }

  for (const Record *supportCase : supportCases) {
    const Record *capability = supportCase->getValueAsDef(kCapability);
    auto constraints = supportCase->getValueAsListOfDefs(kConstraints);
    StringSet<> schemaFields = getCapabilitySchemaFieldKeys(capability);
    StringSet<> typeFields = getCapabilityTypeFieldKeys(capability);
    StringSet<> constrainedFields;
    for (const Record *constraint : constraints) {
      if (constraint->isSubClassOf("TypeSignature"))
        continue;
      bool isAnyValue = constraint->isSubClassOf("AnyValue");
      bool isBoolValue = constraint->isSubClassOf("BoolValueConstraint");
      if (!constraint->isSubClassOf("EnumValueConstraint") && !isAnyValue &&
          !isBoolValue) {
        PrintError(constraint->getLoc(),
                   "unsupported CapabilityConstraint kind");
        hadError = true;
        continue;
      }

      const Record *field = constraint->getValueAsDef(kField);
      const Record *expectedDomain = getRequestFieldDomain(field);
      std::string fieldKey = getRequestFieldKey(field);
      if (!schemaFields.contains(fieldKey)) {
        PrintError(constraint->getLoc(),
                   Twine("constraint field '") + fieldKey +
                       "' is not declared by capability '" +
                       getEffectiveName(capability) + "'");
        hadError = true;
      }

      if (!fieldKey.empty() && !constrainedFields.insert(fieldKey).second) {
        PrintError(constraint->getLoc(),
                   Twine("duplicate constraint for request field '") +
                       fieldKey + "'");
        hadError = true;
      }

      if (isAnyValue)
        continue;

      if (isBoolValue) {
        if (!expectedDomain || !isBoolDomain(expectedDomain)) {
          PrintError(constraint->getLoc(),
                     "BoolValueConstraint requires a BoolAttr-backed field");
          hadError = true;
        }
        continue;
      }

      auto allowedValues = constraint->getValueAsListOfDefs(kValues);
      if (allowedValues.empty()) {
        PrintError(constraint->getLoc(),
                   "EnumValueConstraint must contain at least one enum case");
        hadError = true;
      }
      if (!expectedDomain || !isEnumDomain(expectedDomain)) {
        PrintError(constraint->getLoc(),
                   "EnumValueConstraint requires an enum-backed field");
        hadError = true;
        continue;
      }
      const Record *odsEnum = findDomainODSEnum(records, expectedDomain);
      if (!odsEnum)
        continue;
      auto enumCases = odsEnum->getValueAsListOfDefs("enumerants");
      for (const Record *enumCase : allowedValues) {
        if (llvm::is_contained(enumCases, enumCase))
          continue;
        PrintError(enumCase->getLoc(),
                   Twine("ODS enum case '") + enumCase->getName() +
                       "' does not belong to field '" + fieldKey + "'");
        hadError = true;
      }
    }

    StringSet<> signatureKeys;
    unsigned signatureCount = 0;
    for (const Record *signature : constraints) {
      if (!signature->isSubClassOf("TypeSignature"))
        continue;
      ++signatureCount;
      if (!signatureKeys.insert(getTypeSignatureKey(signature)).second) {
        PrintError(signature->getLoc(),
                   "duplicate type signature in SupportCase");
        hadError = true;
      }
      StringSet<> signatureFields;
      for (const Record *field : signature->getValueAsListOfDefs(kFields)) {
        std::string key = getRequestFieldKey(field);
        signatureFields.insert(key);
        if (!schemaFields.contains(key)) {
          PrintError(signature->getLoc(),
                     Twine("signature field '") + key +
                         "' is not declared by capability '" +
                         getEffectiveName(capability) + "'");
          hadError = true;
        }
        if (constrainedFields.contains(key)) {
          PrintError(signature->getLoc(),
                     Twine("signature field '") + key +
                         "' also has a case-level constraint");
          hadError = true;
        }
      }
      for (const auto &typeField : typeFields) {
        StringRef key = typeField.getKey();
        if (constrainedFields.contains(key) || signatureFields.contains(key))
          continue;
        PrintError(signature->getLoc(),
                   Twine("type request field '") + key +
                       "' must occur in every TypeSignature alternative or "
                       "have a case-level AnyValue");
        hadError = true;
      }
    }

    for (const auto &schemaField : schemaFields) {
      StringRef fieldKey = schemaField.getKey();
      if (constrainedFields.contains(fieldKey))
        continue;
      if (typeFields.contains(fieldKey)) {
        if (signatureCount == 0) {
          PrintError(supportCase->getLoc(),
                     Twine("type request field '") + fieldKey +
                         "' requires a TypeSignature or case-level AnyValue");
          hadError = true;
        }
        continue;
      }
      PrintError(supportCase->getLoc(),
                 Twine("non-type request field '") + fieldKey +
                     "' must have an EnumValueConstraint, BoolValueConstraint, "
                     "or AnyValue");
      hadError = true;
    }
  }

  SmallPtrSet<const Record *, 8> profiledArchitectures;
  for (const Record *profile : profiles) {
    const Record *architecture = profile->getValueAsDef(kArchitecture);
    if (!profiledArchitectures.insert(architecture).second) {
      PrintError(profile->getLoc(),
                 Twine("duplicate capability profile for architecture: ") +
                     architecture->getValueAsString(kName));
      hadError = true;
    }

    SmallPtrSet<const Record *, 8> seenCapabilities;
    for (const Record *coverage :
         profile->getValueAsListOfDefs(kCapabilities)) {
      const Record *capability = coverage->getValueAsDef(kCapability);
      if (!seenCapabilities.insert(capability).second) {
        PrintError(coverage->getLoc(),
                   Twine("duplicate capability coverage for '") +
                       getEffectiveName(capability) +
                       "' in architecture profile");
        hadError = true;
      }
    }
  }

  for (const Record *coverage : coverages) {
    const Record *capability = coverage->getValueAsDef(kCapability);
    StringMap<const Record *> normalizedCases;
    for (const Record *supportCase : coverage->getValueAsListOfDefs(kCases)) {
      if (supportCase->getValueAsDef(kCapability) != capability) {
        PrintError(
            supportCase->getLoc(),
            Twine("SupportCase capability does not match coverage for '") +
                getEffectiveName(capability) + "'");
        hadError = true;
        continue;
      }

      std::string normalizedKey = getNormalizedSupportCaseKey(supportCase);
      auto [it, inserted] =
          normalizedCases.try_emplace(normalizedKey, supportCase);
      if (inserted)
        continue;
      PrintError(supportCase->getLoc(),
                 Twine("duplicate normalized SupportCase '") +
                     supportCase->getName() + "' in coverage; equivalent to '" +
                     it->getValue()->getName() + "'");
      hadError = true;
    }
  }

  return hadError;
}

template <typename RecordVector>
static void emitEnum(raw_ostream &os, StringRef enumName,
                     const RecordVector &records) {
  os << "enum class " << enumName << " {\n";
  for (const Record *record : records)
    os << "  " << getEffectiveCppName(record) << ",\n";
  os << "};\n\n";
}

template <typename RecordVector>
static void emitStringifyDecl(raw_ostream &os, StringRef enumName,
                              const RecordVector &) {
  os << "::llvm::StringRef stringify" << enumName << "(" << enumName
     << " value);\n";
}

template <typename RecordVector>
static void emitStringifyDef(raw_ostream &os, StringRef enumName,
                             const RecordVector &records) {
  os << "::llvm::StringRef stringify" << enumName << "(" << enumName
     << " value) {\n  switch (value) {\n";
  for (const Record *record : records) {
    os << "  case " << enumName << "::" << getEffectiveCppName(record)
       << ": return \"";
    os.write_escaped(getEffectiveName(record));
    os << "\";\n";
  }
  os << "  default: return {};\n  }\n}\n\n";
}

static void emitSlotRequestField(raw_ostream &os, int64_t position,
                                 const Record *property) {
  os << "{CapabilityRequestFieldKind::Slot, " << position
     << ", static_cast<unsigned>(CapabilitySlotPropertyKind::"
     << getEffectiveCppName(property) << ")}";
}

static void emitRequestField(raw_ostream &os, const Record *field,
                             const Record *capability) {
  StringRef fieldKind = field->getValueAsString(kFieldKind);
  if (fieldKind == "slot" && field->isSubClassOf("SlotField")) {
    emitSlotRequestField(os, getSlotPosition(capability, field),
                         field->getValueAsDef(kProperty));
  } else if (fieldKind == "parameter" &&
             field->isSubClassOf("CapabilityParameter")) {
    os << "{CapabilityRequestFieldKind::Parameter, 0, "
          "static_cast<unsigned>(CapabilityParameterKind::"
       << getEffectiveCppName(field) << ")}";
  } else {
    PrintFatalError(field->getLoc(),
                    "cannot emit an invalid validated request field");
  }
}

enum class BindingSourceKind { Operand, Result, Attribute };
enum class BindingProjection {
  ElementType,
  Identity,
  AddressSpace,
  Presence,
  OptionalElementType,
  OptionalAddressSpace,
  AttributePresence,
  OptionalEnum,
  IndexDivisible
};

struct ExpectedSchemaField {
  std::string key;
  std::string display;
};

struct BoundField {
  const Record *field = nullptr;
  std::string methodName;
  std::string display;
  BindingSourceKind sourceKind = BindingSourceKind::Operand;
  BindingProjection projection = BindingProjection::Identity;
  std::string odsName;
  int odsIndex = -1;
  std::string valueType;
  std::string storageType;
  std::string convertFromStorage;
  bool hasDefault = false;
  // Address space may be absent on a tensor. from() still succeeds and leaves
  // this field empty so the query can set it.
  bool optionalStorage = false;
  std::string constBuilderTemplate;
  std::string qualifiedDefault;
  // Expression used when an optional operand or attribute is absent.
  std::string fallbackExpr;
  int divisor = 0;
};

struct ResolvedBinding {
  const Record *capability = nullptr;
  std::string className;
  std::string opClass;
  bool verify = true;
  SmallVector<BoundField, 0> fields;
};

static SmallVector<ExpectedSchemaField>
getExpectedSchemaFields(const Record *capability) {
  SmallVector<ExpectedSchemaField> fields;
  for (const Record *slot : getCapabilitySlots(capability)) {
    for (const Record *property : slot->getValueAsListOfDefs(kProperties)) {
      ExpectedSchemaField field;
      field.key = getSlotFieldKey(slot, property);
      field.display = getEffectiveName(slot) + "." + getEffectiveName(property);
      fields.push_back(std::move(field));
    }
  }
  for (const Record *parameter : getCapabilityParameters(capability)) {
    ExpectedSchemaField field;
    field.key = (Twine("parameter:") + parameter->getName()).str();
    field.display = getEffectiveName(parameter);
    fields.push_back(std::move(field));
  }
  return fields;
}

static std::string fieldMethodName(const Record *field) {
  std::string pascal;
  if (field->isSubClassOf("SlotField")) {
    pascal = (Twine(getEffectiveCppName(field->getValueAsDef(kSlot))) +
              getEffectiveCppName(field->getValueAsDef(kProperty)))
                 .str();
  } else {
    pascal = getEffectiveCppName(field).str();
  }
  if (!pascal.empty())
    pascal.front() = llvm::toLower(pascal.front());
  return pascal;
}

static std::string qualifyDefaultExpr(StringRef value, StringRef cppNamespace) {
  value = value.trim();
  if (value.empty() || value == "true" || value == "false" ||
      value.starts_with("::"))
    return value.str();
  const bool numeric = llvm::all_of(
      value, [](char c) { return llvm::isDigit(c) || c == '-' || c == '+'; });
  if (numeric || cppNamespace.empty())
    return value.str();
  return (cppNamespace + "::" + value).str();
}

static bool attributeDomainMatches(const Record *field,
                                   const mlir::tblgen::Attribute &attr) {
  const Record *domain = getRequestFieldDomain(field);
  const llvm::Record &base = attr.getBaseAttr().getDef();
  if (isEnumDomain(domain))
    return &base == domain->getValueAsDef(kEnumAttr);
  if (isBoolDomain(domain)) {
    const Record *boolAttr = domain->getValueAsDef(kBoolAttr);
    return &base == boolAttr || base.isSubClassOf(boolAttr->getName());
  }
  return false;
}

static bool bindExtractor(const Record *extractor,
                          const mlir::tblgen::Operator &op, StringRef display,
                          BoundField &out) {
  const Record *field = extractor->getValueAsDef(kField);
  const Record *source = extractor->getValueAsDef("Source");
  const Record *projection = extractor->getValueAsDef("Projection");
  const StringRef sourceKind = source->getValueAsString("SourceKind");
  const StringRef projectionKind =
      projection->getValueAsString("ProjectionKind");
  const StringRef odsName = source->getValueAsString("OdsName");
  const std::string opName = op.getCppClassName().str();

  const mlir::tblgen::NamedTypeConstraint *value = nullptr;
  const mlir::tblgen::NamedAttribute *attribute = nullptr;
  enum class Found { Missing, Operand, Result, Attribute };
  Found found = Found::Missing;
  int odsIndex = -1;
  int operandIndex = 0;
  for (int i = 0, end = op.getNumArgs(); i < end; ++i) {
    mlir::tblgen::Argument argument = op.getArg(i);
    if (auto *operand =
            dyn_cast<mlir::tblgen::NamedTypeConstraint *>(argument)) {
      if (operand->name == odsName) {
        value = operand;
        found = Found::Operand;
        odsIndex = operandIndex;
        break;
      }
      ++operandIndex;
    } else if (auto *namedAttr =
                   dyn_cast<mlir::tblgen::NamedAttribute *>(argument)) {
      if (namedAttr->name == odsName) {
        attribute = namedAttr;
        found = Found::Attribute;
        break;
      }
    }
  }
  if (found == Found::Missing) {
    int resultIndex = 0;
    for (const mlir::tblgen::NamedTypeConstraint &result : op.getResults()) {
      if (result.name == odsName) {
        value = &result;
        found = Found::Result;
        odsIndex = resultIndex;
        break;
      }
      ++resultIndex;
    }
  }

  auto reportSourceMismatch = [&](StringRef expected) {
    if (found == Found::Missing) {
      PrintError(extractor->getLoc(), Twine("capability binding: no ") +
                                          expected + " named '" + odsName +
                                          "' on '" + opName + "'");
      return;
    }
    StringRef actual = found == Found::Attribute ? "an attribute"
                       : found == Found::Result  ? "a result"
                                                 : "an operand";
    PrintError(extractor->getLoc(),
               Twine("capability binding: '") + odsName + "' on '" + opName +
                   "' is " + actual + ", not " +
                   (expected == "attribute" ? "an attribute"
                    : expected == "result"  ? "a result"
                                            : "an operand"));
  };

  if (sourceKind == "operand") {
    if (found != Found::Operand) {
      reportSourceMismatch("operand");
      return true;
    }
  } else if (sourceKind == "result") {
    if (found != Found::Result) {
      reportSourceMismatch("result");
      return true;
    }
  } else if (sourceKind == "attribute") {
    if (found != Found::Attribute) {
      reportSourceMismatch("attribute");
      return true;
    }
  } else {
    PrintError(source->getLoc(),
               "capability binding: unknown source kind '" + sourceKind + "'");
    return true;
  }

  out.field = field;
  out.display = display.str();
  out.odsName = odsName.str();
  out.methodName = fieldMethodName(field);
  out.odsIndex = odsIndex;

  const bool isValue = found != Found::Attribute;
  auto finishOptionalValue = [&](BindingProjection kind, StringRef valueType) {
    out.fallbackExpr = projection->getValueAsString("FallbackExpr").str();
    if (out.fallbackExpr.empty()) {
      PrintError(extractor->getLoc(),
                 "capability binding: optional projection requires a "
                 "non-empty FallbackExpr");
      return true;
    }
    out.sourceKind = BindingSourceKind::Operand;
    out.projection = kind;
    out.valueType = valueType.str();
    return false;
  };
  if (projectionKind == "optional_element_type") {
    if (found != Found::Operand ||
        !isTypeDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: optional_element_type requires an "
                 "operand and a type-domain field");
      return true;
    }
    return finishOptionalValue(BindingProjection::OptionalElementType,
                               "::mlir::Type");
  }
  if (projectionKind == "optional_address_space") {
    if (found != Found::Operand ||
        !isEnumDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: optional_address_space requires an "
                 "operand and an enum-domain field");
      return true;
    }
    out.optionalStorage = true;
    return finishOptionalValue(BindingProjection::OptionalAddressSpace,
                               "::mlir::hivm::AddressSpace");
  }
  if (projectionKind == "attribute_presence") {
    if (found != Found::Attribute ||
        !isBoolDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: attribute_presence requires an "
                 "attribute and a bool-domain field");
      return true;
    }
    out.sourceKind = BindingSourceKind::Attribute;
    out.projection = BindingProjection::AttributePresence;
    out.valueType = "bool";
    return false;
  }
  const bool optionalEnum = projectionKind == "optional_enum";
  if (optionalEnum && found != Found::Attribute) {
    PrintError(extractor->getLoc(),
               "capability binding: optional_enum requires an attribute");
    return true;
  }
  if (projectionKind == "index_divisible") {
    if (found != Found::Operand ||
        !isBoolDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: index_divisible requires an operand "
                 "and a bool-domain field");
      return true;
    }
    out.sourceKind = BindingSourceKind::Operand;
    out.projection = BindingProjection::IndexDivisible;
    out.valueType = "bool";
    out.divisor = projection->getValueAsInt("Divisor");
    if (out.divisor <= 0) {
      PrintError(extractor->getLoc(),
                 "capability binding: index_divisible requires a positive "
                 "Divisor");
      return true;
    }
    return false;
  }

  if (projectionKind == "presence") {
    if (!isValue || found != Found::Operand) {
      PrintError(extractor->getLoc(),
                 "capability binding: presence projection requires an operand");
      return true;
    }
    if (!isBoolDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: presence projection requires a "
                 "bool-domain field");
      return true;
    }
    out.sourceKind = BindingSourceKind::Operand;
    out.projection = BindingProjection::Presence;
    out.valueType = "bool";
    return false;
  }

  if (isValue && (value->isOptional() || value->isVariadic() ||
                  value->isVariadicOfVariadic())) {
    const StringRef kind = value->isOptional() ? "optional" : "variadic";
    const StringRef role = found == Found::Result ? "result" : "operand";
    PrintError(extractor->getLoc(), Twine("capability binding: ") + kind + " " +
                                        role + " '" + odsName +
                                        "' cannot feed a required field");
    return true;
  }

  if (projectionKind == "element_type") {
    if (!isValue) {
      PrintError(extractor->getLoc(),
                 "capability binding: element_type projection does not apply "
                 "to an attribute");
      return true;
    }
    if (!isTypeDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: element_type projection requires a "
                 "type-domain field");
      return true;
    }
    out.sourceKind = found == Found::Result ? BindingSourceKind::Result
                                            : BindingSourceKind::Operand;
    out.projection = BindingProjection::ElementType;
    out.valueType = "::mlir::Type";
    return false;
  }

  if (projectionKind == "address_space") {
    if (!isValue) {
      PrintError(extractor->getLoc(),
                 "capability binding: address_space projection does not apply "
                 "to an attribute");
      return true;
    }
    const Record *domain = getRequestFieldDomain(field);
    if (!domain || !isEnumDomain(domain)) {
      PrintError(extractor->getLoc(),
                 "capability binding: address_space projection requires an "
                 "enum-domain field");
      return true;
    }
    const Record *attr = domain->getValueAsDef(kEnumAttr);
    if (!attr->isSubClassOf("AttrDef")) {
      PrintError(extractor->getLoc(),
                 "capability binding: address_space domain has no AttrDef");
      return true;
    }
    mlir::tblgen::AttrDef attrDef(attr);
    auto parameters = attrDef.getParameters();
    if (parameters.size() != 1) {
      PrintError(extractor->getLoc(),
                 "capability binding: address_space attribute must have one "
                 "parameter");
      return true;
    }
    out.sourceKind = found == Found::Result ? BindingSourceKind::Result
                                            : BindingSourceKind::Operand;
    out.projection = BindingProjection::AddressSpace;
    out.optionalStorage = true;
    out.valueType = parameters.front().getCppType().str();
    return false;
  }

  if (projectionKind != "identity" && !optionalEnum) {
    PrintError(projection->getLoc(),
               "capability binding: unknown projection '" + projectionKind +
                   "'");
    return true;
  }

  out.projection = optionalEnum ? BindingProjection::OptionalEnum
                                : BindingProjection::Identity;
  if (isValue) {
    if (!isTypeDomain(getRequestFieldDomain(field))) {
      PrintError(extractor->getLoc(),
                 "capability binding: identity projection from an SSA value "
                 "requires a type-domain field");
      return true;
    }
    out.sourceKind = found == Found::Result ? BindingSourceKind::Result
                                            : BindingSourceKind::Operand;
    out.valueType = "::mlir::Type";
    return false;
  }

  const mlir::tblgen::Attribute &attr = attribute->attr;
  if (attr.isDerivedAttr()) {
    PrintError(extractor->getLoc(),
               Twine("capability binding: derived attribute '") + odsName +
                   "' cannot be bound");
    return true;
  }
  if (attr.isOptional() && !attr.hasDefaultValue() && !optionalEnum) {
    PrintError(extractor->getLoc(),
               Twine("capability binding: optional attribute '") + odsName +
                   "' has no ODS default");
    return true;
  }
  if (!optionalEnum && !attributeDomainMatches(field, attr)) {
    PrintError(extractor->getLoc(),
               Twine("capability binding: attribute '") + odsName +
                   "' does not match the domain of '" + display + "'");
    return true;
  }
  if (attr.hasDefaultValue() && !attr.isConstBuildable()) {
    PrintError(extractor->getLoc(), Twine("capability binding: attribute '") +
                                        odsName +
                                        "' default has no constBuilderCall");
    return true;
  }

  const mlir::tblgen::Attribute base = attr.getBaseAttr();
  out.sourceKind = BindingSourceKind::Attribute;
  out.valueType = base.getReturnType().trim().str();
  out.storageType = base.getStorageType().trim().str();
  out.convertFromStorage = base.getConvertFromStorageCall().trim().str();
  out.hasDefault = attr.hasDefaultValue();
  if (out.hasDefault) {
    out.constBuilderTemplate = attr.getConstBuilderTemplate().str();
    out.qualifiedDefault = qualifyDefaultExpr(
        attr.getDefaultValue(), op.getDialect().getCppNamespace());
  }
  if (out.valueType.empty() || out.storageType.empty() ||
      out.convertFromStorage.empty()) {
    PrintError(extractor->getLoc(), Twine("capability binding: attribute '") +
                                        odsName +
                                        "' has no C++ storage conversion");
    return true;
  }
  return false;
}

static bool resolveBinding(const Record *binding, ResolvedBinding &out) {
  const Record *capability = binding->getValueAsDef(kCapability);
  mlir::tblgen::Operator op(binding->getValueAsDef("Operation"));
  const std::string className =
      (getEffectiveCppName(capability) + Twine("CapabilityRequestBuilder"))
          .str();
  if (!isValidCppIdentifier(className)) {
    PrintError(binding->getLoc(),
               "capability binding: invalid builder name '" + className + "'");
    return true;
  }

  StringMap<const Record *> extractors;
  for (const Record *extractor : binding->getValueAsListOfDefs("Extractors")) {
    const std::string key =
        getRequestFieldKey(extractor->getValueAsDef(kField));
    if (!extractors.insert({key, extractor}).second) {
      const Record *covered = extractor->getValueAsDef(kField);
      const std::string label = covered->isSubClassOf("CapabilityParameter")
                                    ? getEffectiveName(covered)
                                    : covered->getName().str();
      PrintError(extractor->getLoc(), Twine("capability binding: '") +
                                          capability->getName() + "' covers '" +
                                          label + "' more than once");
      return true;
    }
  }

  StringSet<> expectedKeys;
  SmallVector<ExpectedSchemaField> expected =
      getExpectedSchemaFields(capability);
  for (const ExpectedSchemaField &field : expected)
    expectedKeys.insert(field.key);
  for (const auto &entry : extractors) {
    if (expectedKeys.contains(entry.getKey()))
      continue;
    PrintError(entry.getValue()->getLoc(),
               Twine("capability binding: field is not declared by '") +
                   capability->getName() + "'");
    return true;
  }

  out.capability = capability;
  out.className = className;
  out.opClass = op.getQualCppClassName();
  out.verify = binding->getValueAsBit("Verify");
  StringSet<> methodNames;
  for (const ExpectedSchemaField &field : expected) {
    const auto found = extractors.find(field.key);
    if (found == extractors.end()) {
      PrintError(binding->getLoc(),
                 Twine("capability binding: '") + capability->getName() +
                     "' does not cover '" + field.display + "'");
      return true;
    }
    BoundField bound;
    if (bindExtractor(found->getValue(), op, field.display, bound))
      return true;
    if (!isValidCppIdentifier(bound.methodName) || bound.methodName == "from" ||
        bound.methodName == "create" || bound.methodName == "build" ||
        bound.methodName == "hasUnsetAddressSpace") {
      PrintError(found->getValue()->getLoc(),
                 "capability binding: invalid builder method '" +
                     bound.methodName + "'");
      return true;
    }
    if (!methodNames.insert(bound.methodName).second) {
      PrintError(found->getValue()->getLoc(),
                 "capability binding: duplicate builder method '" +
                     bound.methodName + "'");
      return true;
    }
    out.fields.push_back(std::move(bound));
  }
  return false;
}

static bool validateCapabilityBindings(const RecordKeeper &records) {
  SmallVector<const Record *> bindings =
      getDefinitions(records, "CapabilityBinding");
  if (bindings.empty())
    return false;

  SmallVector<const Record *> seen;
  for (const Record *binding : bindings) {
    const Record *capability = binding->getValueAsDef(kCapability);
    if (llvm::is_contained(seen, capability)) {
      PrintError(binding->getLoc(),
                 Twine("capability binding: duplicate binding for '") +
                     capability->getName() + "'");
      return true;
    }
    seen.push_back(capability);
  }

  for (const Record *binding : bindings) {
    ResolvedBinding resolved;
    if (resolveBinding(binding, resolved))
      return true;
  }
  return false;
}

static bool storesCapabilityValue(const BoundField &field) {
  return field.sourceKind == BindingSourceKind::Attribute ||
         field.projection == BindingProjection::AddressSpace ||
         field.projection == BindingProjection::OptionalAddressSpace ||
         field.projection == BindingProjection::Presence ||
         field.projection == BindingProjection::IndexDivisible;
}

static void emitValueRead(raw_ostream &os, const BoundField &field) {
  if (field.projection == BindingProjection::AddressSpace) {
    const char *accessor = field.sourceKind == BindingSourceKind::Result
                               ? "getODSResults"
                               : "getODSOperands";
    // A missing memref scope leaves the field empty. from() still returns
    // the builder so the query can set srcAddressSpace and dstAddressSpace.
    os << "  {\n"
       << "    auto values = op." << accessor << "(" << field.odsIndex << ");\n"
       << "    if (!values.empty()) {\n"
       << "      if (auto space = ::mlir::hivm::getOptionalHIVMAddressSpace("
          "values.front().getType()))\n"
       << "        builder." << field.methodName << "_ = *space;\n"
       << "    }\n"
       << "  }\n";
    return;
  }
  if (field.projection == BindingProjection::OptionalElementType ||
      field.projection == BindingProjection::OptionalAddressSpace ||
      field.projection == BindingProjection::AttributePresence ||
      field.projection == BindingProjection::IndexDivisible) {
    os << "  {\n";
    if (field.projection == BindingProjection::AttributePresence) {
      os << "    builder." << field.methodName << "_ = op->getAttr(\""
         << field.odsName << "\") != nullptr;\n";
    } else if (field.projection == BindingProjection::IndexDivisible) {
      os << "    builder." << field.methodName << "_ = false;\n"
         << "    auto values = op.getODSOperands(" << field.odsIndex << ");\n"
         << "    if (!values.empty()) {\n"
         << "      if (auto cst = "
            "::mlir::getConstantIntValue(values.front()))\n"
         << "        builder." << field.methodName << "_ = *cst > 0 && *cst % "
         << field.divisor << " == 0;\n"
         << "    }\n";
    } else {
      os << "    auto values = op.getODSOperands(" << field.odsIndex << ");\n"
         << "    builder." << field.methodName << "_ = ";
      if (field.projection == BindingProjection::OptionalElementType) {
        os << "values.empty() ? " << field.fallbackExpr
           << " : ::llvm::cast<::mlir::ShapedType>(values.front().getType())"
              ".getElementType();\n";
      } else {
        os << field.fallbackExpr << ";\n"
           << "    if (!values.empty()) {\n"
           << "      if (auto space = "
              "::mlir::hivm::getOptionalHIVMAddressSpace("
              "values.front().getType()))\n"
           << "        builder." << field.methodName << "_ = *space;\n"
           << "    }\n";
      }
    }
    os << "  }\n";
    return;
  }
  os << "  if (auto " << field.methodName << " = [&]() -> ::std::optional<"
     << field.valueType << "> {\n";
  if (field.projection == BindingProjection::Presence) {
    os << "    auto values = op.getODSOperands(" << field.odsIndex << ");\n"
       << "    return !values.empty();\n";
  } else if (field.projection == BindingProjection::ElementType) {
    const char *accessor = field.sourceKind == BindingSourceKind::Result
                               ? "getODSResults"
                               : "getODSOperands";
    os << "    auto values = op." << accessor << "(" << field.odsIndex << ");\n"
       << "    if (values.empty())\n"
       << "      return ::std::nullopt;\n"
       << "    ::mlir::Value value = values.front();\n"
       << "    auto shaped = ::llvm::dyn_cast<::mlir::ShapedType>("
          "value.getType());\n"
       << "    if (!shaped)\n"
       << "      return ::std::nullopt;\n"
       << "    ::mlir::Type elementType = shaped.getElementType();\n"
       << "    if (!elementType)\n"
       << "      return ::std::nullopt;\n"
       << "    return elementType;\n";
  } else if (field.sourceKind != BindingSourceKind::Attribute) {
    const char *accessor = field.sourceKind == BindingSourceKind::Result
                               ? "getODSResults"
                               : "getODSOperands";
    os << "    auto values = op." << accessor << "(" << field.odsIndex << ");\n"
       << "    if (values.empty())\n"
       << "      return ::std::nullopt;\n"
       << "    ::mlir::Type type = values.front().getType();\n"
       << "    if (!type)\n"
       << "      return ::std::nullopt;\n"
       << "    return type;\n";
  } else {
    os << "    ::mlir::Attribute raw = op->getAttr(\"" << field.odsName
       << "\");\n"
       << "    if (!raw) {\n";
    if (field.hasDefault) {
      os << "      ::mlir::Builder odsBuilder(op->getContext());\n"
         << "      auto built = ";
      mlir::tblgen::FmtContext builderCtx;
      builderCtx.withBuilder("odsBuilder");
      os << mlir::tblgen::tgfmt(field.constBuilderTemplate, &builderCtx,
                                field.qualifiedDefault)
         << ";\n"
         << "      return ";
      mlir::tblgen::FmtContext builtCtx;
      builtCtx.withSelf("built");
      os << mlir::tblgen::tgfmt(field.convertFromStorage, &builtCtx) << ";\n";
    } else {
      os << "      return ::std::nullopt;\n";
    }
    os << "    }\n"
       << "    auto typed = ::llvm::dyn_cast<" << field.storageType
       << ">(raw);\n"
       << "    if (!typed)\n"
       << "      return ::std::nullopt;\n"
       << "    return ";
    mlir::tblgen::FmtContext typedCtx;
    typedCtx.withSelf("typed");
    os << mlir::tblgen::tgfmt(field.convertFromStorage, &typedCtx) << ";\n";
  }
  os << "  }())\n"
     << "    builder." << field.methodName << "_ = *" << field.methodName
     << ";\n"
     << "  else\n"
     << "    return ::std::nullopt;\n";
}

static void emitBindingClass(raw_ostream &os, const ResolvedBinding &binding) {
  const StringRef className = binding.className;
  os << "/// Builds a " << getEffectiveName(binding.capability)
     << " request without creating a temporary operation.\n"
     << "/// create() takes every required field, so a new schema field fails\n"
     << "/// to compile at call sites that do not supply it. from() reads the\n"
     << "/// bound operation and leaves that operation unchanged.\n"
     << "class " << className << " {\n"
     << "public:\n"
     << "  /// nullopt is an illegal binding input. It is not Unknown and it\n"
     << "  /// is not a hardware miss.\n"
     << "  static ::std::optional<" << className << "> from(" << binding.opClass
     << " op);\n\n"
     << "  static " << className << " create(";
  for (auto [index, field] : llvm::enumerate(binding.fields)) {
    if (index != 0)
      os << ", ";
    os << field.valueType << " " << field.methodName;
  }
  os << ");\n\n";
  for (const BoundField &field : binding.fields) {
    os << "  " << className << " &" << field.methodName << "("
       << field.valueType << " value) {\n"
       << "    " << field.methodName << "_ = value;\n"
       << "    return *this;\n"
       << "  }\n";
  }
  os << "\n  /// Owns the field array. The result does not borrow this "
        "builder.\n"
     << "  /// mlir::Type handles remain valid with their MLIRContext.\n"
     << "  CapabilityRequestStorage build() const;\n\n"
     << "  /// True when an address-space field was not read from the "
        "operation.\n"
     << "  /// build() would store a sentinel instead of a real scope.\n"
     << "  bool hasUnsetAddressSpace() const {\n    return ";
  bool anyAddressSpace = false;
  for (const BoundField &field : binding.fields) {
    if (field.projection != BindingProjection::AddressSpace ||
        !field.optionalStorage)
      continue;
    if (anyAddressSpace)
      os << " || ";
    anyAddressSpace = true;
    os << "!" << field.methodName << "_";
  }
  if (!anyAddressSpace)
    os << "false";
  os << ";\n  }\n\n"
     << "private:\n"
     << "  " << className << "() = default;\n";
  for (const BoundField &field : binding.fields) {
    if (field.optionalStorage)
      os << "  ::std::optional<" << field.valueType << "> ";
    else
      os << "  " << field.valueType << " ";
    os << field.methodName << "_{};\n";
  }
  os << "};\n\n";

  os << "inline ::std::optional<" << className << "> " << className << "::from("
     << binding.opClass << " op) {\n"
     << "  " << className << " builder;\n";
  for (const BoundField &field : binding.fields)
    emitValueRead(os, field);
  os << "  return builder;\n}\n\n";

  os << "inline " << className << " " << className << "::create(";
  for (auto [index, field] : llvm::enumerate(binding.fields)) {
    if (index != 0)
      os << ", ";
    os << field.valueType << " " << field.methodName;
  }
  os << ") {\n  " << className << " builder;\n";
  for (const BoundField &field : binding.fields)
    os << "  builder." << field.methodName << "_ = " << field.methodName
       << ";\n";
  os << "  return builder;\n}\n\n";

  os << "inline CapabilityRequestStorage " << className << "::build() const {\n"
     << "  CapabilityRequestStorage storage;\n"
     << "  storage.capability = CapabilityKind::"
     << getEffectiveCppName(binding.capability) << ";\n"
     << "  storage.values.reserve(" << binding.fields.size() << ");\n";
  for (const BoundField &field : binding.fields) {
    os << "  storage.values.push_back({";
    emitRequestField(os, field.field, binding.capability);
    if (storesCapabilityValue(field)) {
      os << ", CapabilityValue{CapabilityValueDomainKind::"
         << getEffectiveCppName(getRequestFieldDomain(field.field))
         << ", static_cast<int64_t>(";
      if (field.optionalStorage)
        os << field.methodName << "_.value_or(static_cast<" << field.valueType
           << ">(-1))";
      else if (field.projection == BindingProjection::OptionalEnum)
        os << field.methodName << "_.getSExtValue()";
      else
        os << field.methodName << "_";
      os << ")}});\n";
    } else {
      os << ", " << field.methodName << "_});\n";
    }
  }
  os << "  return storage;\n}\n\n";
}

static bool emitTargetCapabilityBindings(const RecordKeeper &records,
                                         raw_ostream &os) {
  if (validateCapabilityRecords(records) || validateCapabilityBindings(records))
    return true;

  CapabilityEmissionScope scope;
  if (parseEmissionScope(records, scope))
    return true;

  emitSourceFileHeader("Target Capability Bindings", os, records);
  emitIncludeGuardOpen(os, scope.bindingGuard);
  emitNamespaceOpen(os, scope, false);
  SmallVector<ResolvedBinding, 4> bindings;
  for (const Record *binding : getDefinitions(records, "CapabilityBinding")) {
    ResolvedBinding resolved;
    if (resolveBinding(binding, resolved))
      return true;
    bindings.push_back(std::move(resolved));
  }
  for (const ResolvedBinding &binding : bindings)
    emitBindingClass(os, binding);

  os << "inline void visitBoundCapabilityOps(\n"
        "    ::mlir::Operation *root,\n"
        "    ::llvm::function_ref<void(\n"
        "        ::mlir::Operation *op, bool bindingFailed, bool "
        "missingAddressSpace,\n"
        "        ::std::optional<CapabilityRequestStorage> request)> visit) {\n"
        "  if (!root)\n"
        "    return;\n";
  for (const ResolvedBinding &binding : bindings) {
    if (!binding.verify)
      continue;
    os << "  root->walk([&](" << binding.opClass << " op) {\n"
       << "    auto builder = " << binding.className << "::from(op);\n"
       << "    if (!builder) {\n"
       << "      visit(op.getOperation(), true, false, ::std::nullopt);\n"
       << "      return;\n"
       << "    }\n"
       << "    if (builder->hasUnsetAddressSpace()) {\n"
       << "      visit(op.getOperation(), false, true, ::std::nullopt);\n"
       << "      return;\n"
       << "    }\n"
       << "    visit(op.getOperation(), false, false, builder->build());\n"
       << "  });\n";
  }
  os << "}\n\n"
        "struct CapabilityCatalogue {\n"
        "  using RequestStorage = CapabilityRequestStorage;\n"
        "  using Request = CapabilityRequest;\n"
        "  using Support = NativeSupport;\n"
        "  using Completeness = ProfileCompleteness;\n"
        "  using Architecture = ::bishengir::npu::NPUArchitectureKind;\n"
        "  using PropertyKind = CapabilitySlotPropertyKind;\n"
        "  using ParameterKind = CapabilityParameterKind;\n"
        "  using FieldValue = CapabilityFieldValue;\n"
        "  using StoredValue = CapabilityValue;\n"
        "  using FieldKind = CapabilityRequestFieldKind;\n"
        "\n"
        "  template <typename Fn>\n"
        "  static void visit(::mlir::Operation *root, Fn &&fn) {\n"
        "    visitBoundCapabilityOps(root, std::forward<Fn>(fn));\n"
        "  }\n"
        "  static Support query(Architecture architecture, const Request "
        "&request) {\n"
        "    return queryNativeSupport(architecture, request);\n"
        "  }\n"
        "  static const ArchitectureCapabilityProfileDesc *\n"
        "  profile(Architecture architecture) {\n"
        "    return lookupArchitectureCapabilityProfile(architecture);\n"
        "  }\n"
        "  static ::llvm::StringRef completeness(Completeness value) {\n"
        "    return stringifyProfileCompleteness(value);\n"
        "  }\n"
        "  static ::llvm::StringRef property(PropertyKind kind) {\n"
        "    return stringifyCapabilitySlotPropertyKind(kind);\n"
        "  }\n"
        "  static ::llvm::StringRef parameter(ParameterKind kind) {\n"
        "    return stringifyCapabilityParameterKind(kind);\n"
        "  }\n"
        "  static ::std::optional<::llvm::StringRef> value(StoredValue stored) "
        "{\n"
        "    return stringifyCapabilityValue(stored);\n"
        "  }\n"
        "};\n\n";
  emitNamespaceClose(os, scope);
  emitIncludeGuardClose(os, scope.bindingGuard);
  return false;
}

static mlir::GenRegistration genMmadCapabilityBinding(
    "gen-mmad-capability-binding",
    "Generate MMAD capability request builders bound to ODS operations",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitTargetCapabilityBindings(records, os);
    });

} // namespace
