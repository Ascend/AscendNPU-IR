//===- TargetCapabilityGen.cpp - Target capability generator ------------===//
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

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
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
constexpr StringLiteral kCompleteness = "Completeness";
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

static void emitCapabilityRequestSchema(raw_ostream &os,
                                        const Record *capability) {
  bool first = true;
  auto separator = [&] {
    if (!first)
      os << ", ";
    first = false;
  };
  for (auto [position, slot] :
       llvm::enumerate(getCapabilitySlots(capability))) {
    for (const Record *property : slot->getValueAsListOfDefs(kProperties)) {
      separator();
      os << "{";
      emitSlotRequestField(os, position, property);
      os << ", CapabilityValueDomainKind::"
         << getEffectiveCppName(property->getValueAsDef(kDomain));
      os << "}";
    }
  }
  for (const Record *parameter : getCapabilityParameters(capability)) {
    separator();
    os << "{{CapabilityRequestFieldKind::Parameter, 0, "
          "static_cast<unsigned>(CapabilityParameterKind::"
       << getEffectiveCppName(parameter) << ")}, CapabilityValueDomainKind::"
       << getEffectiveCppName(parameter->getValueAsDef(kDomain)) << "}";
  }
}

static bool emitTargetCapabilityDecls(const RecordKeeper &records,
                                      raw_ostream &os) {
  if (validateCapabilityRecords(records))
    return true;

  auto valueDomains = getCapabilityValueDomains(records);
  auto properties = getDefinitions(records, "CapabilitySlotProperty");
  auto parameters =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityParameter"));
  auto capabilities =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityKey"));
  auto completenesses =
      asConstRecords(records.getAllDerivedDefinitions("ProfileCompleteness"));

  emitSourceFileHeader("Target Capability Declarations", os, records);
  os << R"(
#ifndef BISHENGIR_TARGET_CAPABILITY_H
#define BISHENGIR_TARGET_CAPABILITY_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace bishengir {
namespace npu {

)";

  emitEnum(os, "CapabilityValueDomainKind", valueDomains);
  emitEnum(os, "CapabilitySlotPropertyKind", properties);
  emitEnum(os, "CapabilityKind", capabilities);
  emitEnum(os, "CapabilityParameterKind", parameters);
  emitEnum(os, "ProfileCompleteness", completenesses);

  emitStringifyDecl(os, "CapabilityValueDomainKind", valueDomains);
  emitStringifyDecl(os, "CapabilitySlotPropertyKind", properties);
  emitStringifyDecl(os, "CapabilityKind", capabilities);
  emitStringifyDecl(os, "CapabilityParameterKind", parameters);
  emitStringifyDecl(os, "ProfileCompleteness", completenesses);

  os << R"(

enum class CapabilityRequestFieldKind { Slot, Parameter };

struct CapabilityRequestFieldDesc {
  CapabilityRequestFieldKind kind;
  /// Semantic slot position; zero for a named parameter.
  unsigned position;
  /// Property identity for Slot, parameter identity for Parameter.
  unsigned identity;

  static CapabilityRequestFieldDesc
  slot(unsigned position, CapabilitySlotPropertyKind property) {
    return {CapabilityRequestFieldKind::Slot, position,
            static_cast<unsigned>(property)};
  }
  static CapabilityRequestFieldDesc parameter(CapabilityParameterKind parameter) {
    return {CapabilityRequestFieldKind::Parameter, 0,
            static_cast<unsigned>(parameter)};
  }
};

/// Every explicitly declared field is required and has exactly one carrier.
struct CapabilityRequestSchemaFieldDesc {
  CapabilityRequestFieldDesc field;
  CapabilityValueDomainKind domain;
};

struct CapabilityValue {
  CapabilityValueDomainKind domain;
  int64_t rawValue;
};

struct CapabilityFieldValue {
  CapabilityRequestFieldDesc field;
  std::variant<::mlir::Type, CapabilityValue> value;
};

/// A non-owning view. Referenced arrays and the types' MLIRContexts must
/// remain alive while the request is used.
struct CapabilityRequest {
  CapabilityKind capability;
  ::llvm::ArrayRef<CapabilityFieldValue> values;
};

/// A generated predicate for one combination of explicitly named type fields.
struct TypeSignatureConstraintDesc {
  bool (*matches)(const CapabilityRequest &) = nullptr;
};

struct ValueConstraintDesc {
  CapabilityRequestFieldDesc field;
  ::llvm::ArrayRef<int64_t> allowedValues;
};

struct SupportCaseDesc {
  CapabilityKind capability;
  ::llvm::ArrayRef<TypeSignatureConstraintDesc> signatures;
  ::llvm::ArrayRef<ValueConstraintDesc> constraints;
};

struct ArchitectureCapabilityCoverageDesc {
  CapabilityKind capability;
  ProfileCompleteness completeness;
  ::llvm::ArrayRef<SupportCaseDesc> cases;
};

enum class NPUArchitectureKind;

struct ArchitectureCapabilityProfileDesc {
  ::bishengir::npu::NPUArchitectureKind architecture;
  ::llvm::ArrayRef<ArchitectureCapabilityCoverageDesc> capabilities;
};

enum class NativeSupport {
  Native,
  Unsupported,
  Unknown,
  InvalidRequest,
};

enum class SupportCaseMatchResult {
  Match,
  Mismatch,
};

struct CapabilityRequestStorage {
  CapabilityKind capability = static_cast<CapabilityKind>(-1);
  /// Owns field values, not the MLIRContexts that store any type handles.
  ::llvm::SmallVector<CapabilityFieldValue, 4> values;

  /// The returned view borrows storage and may be invalidated by mutation.
  CapabilityRequest getRequest() const {
    return {capability, values};
  }
};

/// Returns the complete canonical field schema declared by CapabilityKey.
::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc>
getCapabilityRequestSchema(CapabilityKind capability);

/// Returns the canonical ODS enum spelling for a domain/raw-value pair.
/// std::nullopt means the raw value is not an enumerant of that domain.
std::optional<::llvm::StringRef>
stringifyCapabilityValue(CapabilityValue value);

/// Returns false for duplicate, missing, extra, out-of-range, or incorrectly
/// typed request-field values, including null types. A non-null type does not
/// need to occur in a support rule to form a valid request.
bool isCapabilityRequestWellFormed(const CapabilityRequest &request);

/// Returns the generated profile, or nullptr if the architecture is unknown.
const ArchitectureCapabilityProfileDesc *
lookupArchitectureCapabilityProfile(
    ::bishengir::npu::NPUArchitectureKind architecture);

/// Performs an order-independent query without collapsing Unknown or malformed
/// requests into Unsupported.
NativeSupport queryNativeSupport(::bishengir::npu::NPUArchitectureKind architecture,
                                 const CapabilityRequest &request);

namespace detail {
/// Schema-parametric form used by the generated resolver and focused tests.
bool isCapabilityRequestWellFormed(
    const CapabilityRequest &request,
    ::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc> schema);

/// The caller must validate the request and select a case for its capability
/// before matching. Detected violations are fatal even in release builds; this
/// is not a substitute for request validation. A normal miss returns Mismatch.
SupportCaseMatchResult matchSupportCase(const SupportCaseDesc &supportCase,
                                        const CapabilityRequest &request);

/// Resolves one already-selected profile against an explicit request schema.
/// This keeps the matching engine independently testable from generated lookup.
NativeSupport resolveNativeSupport(
    const ArchitectureCapabilityProfileDesc *profile,
    const CapabilityRequest &request,
    ::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc> schema);
} // namespace detail

} // namespace npu
} // namespace bishengir

#endif // BISHENGIR_TARGET_CAPABILITY_H
)";
  return false;
}

// Record names may contain punctuation or collide with generated suffixes.
// Numeric identities keep all private symbols valid and globally distinct.
static std::string getRecordSymbol(const Record *record) {
  return "CapabilityRecord" + utostr(record->getID());
}

static bool emitTargetCapabilityDefs(const RecordKeeper &records,
                                     raw_ostream &os) {
  if (validateCapabilityRecords(records))
    return true;

  auto valueDomains = getCapabilityValueDomains(records);
  auto properties = getDefinitions(records, "CapabilitySlotProperty");
  auto parameters =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityParameter"));
  auto capabilities =
      asConstRecords(records.getAllDerivedDefinitions("CapabilityKey"));
  auto supportCases =
      asConstRecords(records.getAllDerivedDefinitions("SupportCase"));
  auto completenesses =
      asConstRecords(records.getAllDerivedDefinitions("ProfileCompleteness"));
  auto coverages = getDefinitions(records, "ArchitectureCapabilityCoverage");
  auto profiles = getDefinitions(records, "ArchitectureCapabilityProfile");
  emitSourceFileHeader("Target Capability Definitions", os, records);
  if (!profiles.empty())
    os << "#include \"bishengir/Target/NPU/NPUArchitecture.h\"\n";
  os << R"(
// Include the corresponding generated declarations before this fragment.

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinTypes.h"


namespace bishengir {
namespace npu {

)";

  emitStringifyDef(os, "CapabilityValueDomainKind", valueDomains);
  emitStringifyDef(os, "CapabilitySlotPropertyKind", properties);
  emitStringifyDef(os, "CapabilityKind", capabilities);
  emitStringifyDef(os, "CapabilityParameterKind", parameters);
  emitStringifyDef(os, "ProfileCompleteness", completenesses);

  os << "static bool isTypeDomain(CapabilityValueDomainKind domain) {\n"
        "  switch (domain) {\n";
  for (const Record *domain : valueDomains)
    if (isTypeDomain(domain))
      os << "  case CapabilityValueDomainKind::" << getEffectiveCppName(domain)
         << ": return true;\n";
  os << "  default: return false;\n  }\n}\n\n";

  os << R"(std::optional<::llvm::StringRef>
stringifyCapabilityValue(CapabilityValue value) {
  switch (value.domain) {
)";
  for (const Record *domain : valueDomains) {
    os << "  case CapabilityValueDomainKind::" << getEffectiveCppName(domain)
       << ":\n";
    if (isBoolDomain(domain)) {
      // Boolean domains spell their values stably as true/false instead of
      // leaking raw 0/1 into diagnostics and generated reports.
      os << "    switch (value.rawValue) {\n"
            "    case 0: return \"false\";\n"
            "    case 1: return \"true\";\n"
            "    default: return std::nullopt;\n"
            "    }\n";
      continue;
    }
    if (!isEnumDomain(domain)) {
      os << "    return std::nullopt;\n";
      continue;
    }
    os << "    switch (value.rawValue) {\n";
    for (const Record *enumCase : findDomainODSEnum(records, domain)
                                      ->getValueAsListOfDefs("enumerants")) {
      os << "    case " << enumCase->getValueAsInt("value") << ": return \"";
      os.write_escaped(enumCase->getValueAsString("str"));
      os << "\";\n";
    }
    os << "    default: return std::nullopt;\n    }\n";
  }
  os << R"(  default: return std::nullopt;
  }
}

static bool equalRequestField(const CapabilityRequestFieldDesc &lhs,
                              const CapabilityRequestFieldDesc &rhs) {
  return lhs.kind == rhs.kind && lhs.position == rhs.position &&
         lhs.identity == rhs.identity;
  }

static const CapabilityFieldValue *
findRequestField(const CapabilityRequest &request,
                 const CapabilityRequestFieldDesc &field) {
  auto value = llvm::find_if(request.values, [&](const CapabilityFieldValue &value) {
    return equalRequestField(value.field, field);
  });
  return value == request.values.end() ? nullptr : &*value;
}

)";

  for (const Record *supportCase : supportCases) {
    SmallVector<const Record *> caseSignatures;
    SmallVector<const Record *> caseValueConstraints;
    for (const Record *constraint :
         supportCase->getValueAsListOfDefs(kConstraints)) {
      if (constraint->isSubClassOf("TypeSignature"))
        caseSignatures.push_back(constraint);
      else if (constraint->isSubClassOf("EnumValueConstraint") ||
               constraint->isSubClassOf("BoolValueConstraint"))
        caseValueConstraints.push_back(constraint);
    }
    std::string recordName = getRecordSymbol(supportCase);
    const Record *capability = supportCase->getValueAsDef(kCapability);
    for (auto [index, signature] : llvm::enumerate(caseSignatures)) {
      auto fields = signature->getValueAsListOfDefs(kFields);
      auto types = signature->getValueAsListOfDefs(kTypes);
      os << "static bool match" << recordName << "Signature" << index
         << "(const CapabilityRequest &";
      if (!fields.empty())
        os << "request";
      os << ") {\n";
      // Resolve each explicit field in this capability's schema. Shared
      // signatures may appear in schemas with different slot positions.
      // The profile stores predicates, never context-bound MLIR types.
      for (auto [fieldIndex, field] : llvm::enumerate(fields)) {
        os << "  const auto *field" << fieldIndex
           << " = findRequestField(request, ";
        emitRequestField(os, field, capability);
        os << ");\n  if (!field" << fieldIndex
           << ")\n    ::llvm::report_fatal_error("
              "\"type signature references a missing request field\");\n"
              "  const auto *type"
           << fieldIndex << " = std::get_if<::mlir::Type>(&field" << fieldIndex
           << "->value);\n  if (!type" << fieldIndex << " || !*type"
           << fieldIndex
           << ")\n    ::llvm::report_fatal_error("
              "\"type signature requires a non-null MLIR type\");\n";
      }
      os << "  return ";
      if (types.empty())
        os << "true";
      for (auto [typeIndex, type] : llvm::enumerate(types)) {
        if (typeIndex)
          os << " &&\n         ";
        mlir::tblgen::FmtContext context;
        context.withSelf("(*type" + utostr(typeIndex) + ")");
        os << "(" << mlir::tblgen::tgfmt(getTypePredicate(type), &context)
           << ")";
      }
      os << ";\n}\n";
    }
    if (!caseSignatures.empty()) {
      os << "static const TypeSignatureConstraintDesc k" << recordName
         << "Signatures[] = {";
      for (unsigned index = 0; index < caseSignatures.size(); ++index) {
        if (index)
          os << ", ";
        os << "{match" << recordName << "Signature" << index << "}";
      }
      os << "};\n";
    }
    if (!caseValueConstraints.empty()) {
      for (auto [index, constraint] : llvm::enumerate(caseValueConstraints)) {
        os << "static const int64_t k" << recordName << "Constraint" << index
           << "Values[] = {";
        if (constraint->isSubClassOf("BoolValueConstraint")) {
          os << (constraint->getValueAsBit(kValue) ? "1" : "0");
        } else {
          auto allowedValues =
              asConstRecords(constraint->getValueAsListOfDefs(kValues));
          for (auto [valueIndex, enumCase] : llvm::enumerate(allowedValues)) {
            if (valueIndex)
              os << ", ";
            os << enumCase->getValueAsInt("value");
          }
        }
        os << "};\n";
      }
      os << "static const ValueConstraintDesc k" << recordName
         << "Constraints[] = {";
      for (auto [index, constraint] : llvm::enumerate(caseValueConstraints)) {
        if (index)
          os << ", ";
        os << "{";
        emitRequestField(os, constraint->getValueAsDef(kField), capability);
        os << ", k" << recordName << "Constraint" << index << "Values}";
      }
      os << "};\n";
    }
    os << "static const SupportCaseDesc k" << recordName << " = {"
       << "CapabilityKind::" << getEffectiveCppName(capability) << ", ";
    if (caseSignatures.empty())
      os << "{}";
    else
      os << "k" << recordName << "Signatures";
    os << ", ";
    if (caseValueConstraints.empty())
      os << "{}";
    else
      os << "k" << recordName << "Constraints";
    os << "};\n\n";
  }

  for (const Record *coverage : coverages) {
    auto cases = asConstRecords(coverage->getValueAsListOfDefs(kCases));
    std::string recordName = getRecordSymbol(coverage);
    if (!cases.empty()) {
      os << "static const SupportCaseDesc k" << recordName << "Cases[] = {";
      for (auto [index, supportCase] : llvm::enumerate(cases)) {
        if (index)
          os << ", ";
        os << "k" << getRecordSymbol(supportCase);
      }
      os << "};\n";
    }
    os << "static const ArchitectureCapabilityCoverageDesc k" << recordName
       << " = {CapabilityKind::"
       << getEffectiveCppName(coverage->getValueAsDef(kCapability))
       << ", ProfileCompleteness::"
       << getEffectiveCppName(coverage->getValueAsDef(kCompleteness)) << ", ";
    if (cases.empty())
      os << "{}";
    else
      os << "k" << recordName << "Cases";
    os << "};\n\n";
  }

  for (const Record *profile : profiles) {
    const Record *architecture = profile->getValueAsDef(kArchitecture);
    auto profileCapabilities =
        asConstRecords(profile->getValueAsListOfDefs(kCapabilities));
    std::string recordName = getRecordSymbol(profile);
    if (!profileCapabilities.empty()) {
      os << "static const ArchitectureCapabilityCoverageDesc k" << recordName
         << "Capabilities[] = {";
      for (auto [index, coverage] : llvm::enumerate(profileCapabilities)) {
        if (index)
          os << ", ";
        os << "k" << getRecordSymbol(coverage);
      }
      os << "};\n";
    }
    os << "static const ArchitectureCapabilityProfileDesc k" << recordName
       << " = {::bishengir::npu::NPUArchitectureKind::"
       << getNPUArchitectureEnumerator(architecture) << ", ";
    if (profileCapabilities.empty())
      os << "{}";
    else
      os << "k" << recordName << "Capabilities";
    os << "};\n\n";
  }

  for (const Record *capability : capabilities) {
    if (getCapabilitySchemaFieldKeys(capability).empty())
      continue;
    os << "static const CapabilityRequestSchemaFieldDesc k"
       << getRecordSymbol(capability) << "RequestSchema[] = {";
    emitCapabilityRequestSchema(os, capability);
    os << "};\n";
  }

  os << R"(

::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc>
getCapabilityRequestSchema(CapabilityKind capability) {
  switch (capability) {
)";
  for (const Record *capability : capabilities) {
    os << "  case CapabilityKind::" << getEffectiveCppName(capability)
       << ": return ";
    if (getCapabilitySchemaFieldKeys(capability).empty())
      os << "{}";
    else
      os << "k" << getRecordSymbol(capability) << "RequestSchema";
    os << ";\n";
  }
  os << R"(  default: return {};
  }
}

bool detail::isCapabilityRequestWellFormed(
    const CapabilityRequest &request,
    ::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc> schema) {
  // Empty schemas are valid, so missing keys cannot be detected by schema size.
  if (stringifyCapabilityKind(request.capability).empty())
    return false;

  for (const CapabilityRequestSchemaFieldDesc &schemaField : schema) {
    const CapabilityRequestFieldDesc &expected = schemaField.field;
    unsigned matches = 0;
    for (const CapabilityFieldValue &actual : request.values)
      if (equalRequestField(actual.field, expected))
        ++matches;
    if (matches != 1)
      return false;
  }

  for (const CapabilityFieldValue &actual : request.values) {
    auto expected = llvm::find_if(
        schema, [&](const CapabilityRequestSchemaFieldDesc &schemaField) {
          return equalRequestField(schemaField.field, actual.field);
        });
    if (expected == schema.end())
      return false;

    if (isTypeDomain(expected->domain)) {
      const auto *type = std::get_if<::mlir::Type>(&actual.value);
      if (!type || !*type)
        return false;
    } else {
      const auto *value = std::get_if<CapabilityValue>(&actual.value);
      if (!value || value->domain != expected->domain ||
          !stringifyCapabilityValue(*value))
        return false;
    }
  }
  return true;
}

bool isCapabilityRequestWellFormed(const CapabilityRequest &request) {
  return detail::isCapabilityRequestWellFormed(
      request, getCapabilityRequestSchema(request.capability));
}

SupportCaseMatchResult
detail::matchSupportCase(const SupportCaseDesc &supportCase,
                         const CapabilityRequest &request) {
  if (supportCase.capability != request.capability)
    ::llvm::report_fatal_error("support case and request capability mismatch");

  if (!supportCase.signatures.empty() &&
      !llvm::any_of(supportCase.signatures,
                    [&](const TypeSignatureConstraintDesc &sig) {
                      if (!sig.matches)
                        ::llvm::report_fatal_error(
                            "support case has a null type-signature matcher");
                      return sig.matches(request);
                    }))
    return SupportCaseMatchResult::Mismatch;

  for (const ValueConstraintDesc &constraint : supportCase.constraints) {
    const auto *fieldValue = findRequestField(request, constraint.field);
    if (!fieldValue)
      ::llvm::report_fatal_error(
          "value constraint references a missing request field");
    const auto *value = std::get_if<CapabilityValue>(&fieldValue->value);
    if (!value)
      ::llvm::report_fatal_error(
          "value constraint requires an enum or boolean value");
    if (constraint.allowedValues.empty())
      ::llvm::report_fatal_error("value constraint has no allowed values");
    if (!llvm::is_contained(constraint.allowedValues, value->rawValue))
      return SupportCaseMatchResult::Mismatch;
  }
  return SupportCaseMatchResult::Match;
}

const ArchitectureCapabilityProfileDesc *
lookupArchitectureCapabilityProfile(
    ::bishengir::npu::NPUArchitectureKind architecture) {
)";
  for (const Record *profile : profiles) {
    os << "  if (architecture == ::bishengir::npu::NPUArchitectureKind::"
       << getNPUArchitectureEnumerator(profile->getValueAsDef(kArchitecture))
       << ") return &k" << getRecordSymbol(profile) << ";\n";
  }
  os << R"(  return nullptr;
}

NativeSupport detail::resolveNativeSupport(
    const ArchitectureCapabilityProfileDesc *profile,
    const CapabilityRequest &request,
    ::llvm::ArrayRef<CapabilityRequestSchemaFieldDesc> schema) {
  if (!detail::isCapabilityRequestWellFormed(request, schema))
    return NativeSupport::InvalidRequest;

  if (!profile)
    return NativeSupport::Unknown;

  auto coverage = llvm::find_if(
      profile->capabilities,
      [&](const ArchitectureCapabilityCoverageDesc &candidate) {
        return candidate.capability == request.capability;
      });
  if (coverage == profile->capabilities.end())
    return NativeSupport::Unknown;

  for (const SupportCaseDesc &supportCase : coverage->cases)
    if (detail::matchSupportCase(supportCase, request) == SupportCaseMatchResult::Match)
      return NativeSupport::Native;

  return coverage->completeness == ProfileCompleteness::Complete
             ? NativeSupport::Unsupported
             : NativeSupport::Unknown;
}

NativeSupport queryNativeSupport(::bishengir::npu::NPUArchitectureKind architecture,
                                 const CapabilityRequest &request) {
  return detail::resolveNativeSupport(
      lookupArchitectureCapabilityProfile(architecture), request,
      getCapabilityRequestSchema(request.capability));
}

} // namespace npu
} // namespace bishengir
)";
  return false;
}

static mlir::GenRegistration
    genTargetCapabilityDecls("gen-target-capability-decls",
                             "Generate target capability declarations",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               return emitTargetCapabilityDecls(records, os);
                             });

static mlir::GenRegistration
    genTargetCapabilityDefs("gen-target-capability-defs",
                            "Generate target capability definitions",
                            [](const RecordKeeper &records, raw_ostream &os) {
                              return emitTargetCapabilityDefs(records, os);
                            });

} // namespace
