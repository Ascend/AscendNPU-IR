//===- FixpipePreQuantContractGen.cpp - pre-quant contract gen --*- C++ -*-===//
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
// Generates the target-independent Fixpipe pre-quant type contract from the
// typed enum-case metadata (HIVM_FixpipePreQuantCase). The enum case record
// is the single source of truth: this backend consumes its TypeSignatures
// list and emits the forward verifier, the signature descriptors and the
// reverse candidate lookup. Mode names are never parsed as schema.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <string>

using namespace llvm;

namespace {

constexpr StringLiteral kSymbol = "symbol";
constexpr StringLiteral kTypeSignatures = "TypeSignatures";
constexpr StringLiteral kSrcType = "SrcType";
constexpr StringLiteral kDstType = "DstType";

// Reads one ODS Type record and derives the runtime predicate and the
// diagnostic spelling from the constraint itself. No hand-written
// element-type vocabulary exists: adding a new dtype only requires a TD
// change.
struct ElementTypeContract {
  std::string predicate; // C++ expression taking one ::mlir::Type
  std::string summary;   // stable diagnostic text from the TypeConstraint
};

static bool getElementTypeContract(const Record *signature, bool isSrc,
                                   ElementTypeContract &contract) {
  const Record *typeDef = signature->getValueAsDef(isSrc ? kSrcType : kDstType);
  mlir::tblgen::TypeConstraint constraint(typeDef);
  mlir::tblgen::Pred predicate = constraint.getPredicate();
  if (predicate.isNull() || constraint.getConditionTemplate().empty()) {
    PrintError(signature->getLoc(),
               Twine("'") + typeDef->getName() +
                   "' is not an ODS Type constraint with a predicate");
    return true;
  }

  mlir::tblgen::FmtContext context;
  context.withSelf(isSrc ? "srcElementType" : "dstElementType");
  std::string expanded =
      mlir::tblgen::tgfmt(constraint.getConditionTemplate(), &context);
  bool degenerate = llvm::all_of(expanded, [](char c) {
    return c == '(' || c == ')' || llvm::isSpace(c);
  });
  if (expanded.find('$') != std::string::npos || degenerate) {
    PrintError(signature->getLoc(),
               Twine("predicate of '") + typeDef->getName() +
                   "' cannot be expanded in a self-only context (empty "
                   "predicate or unresolved placeholder)");
    return true;
  }

  contract.predicate = std::move(expanded);
  contract.summary = constraint.getSummary().empty()
                         ? typeDef->getName().str()
                         : constraint.getSummary().str();
  return false;
}

static SmallVector<const Record *>
getPreQuantCases(const RecordKeeper &records) {
  SmallVector<const Record *> cases;
  for (const Record *record :
       records.getAllDerivedDefinitions("HIVM_FixpipePreQuantCase"))
    cases.push_back(record);
  return cases;
}

// Return the canonical Fixpipe pre-quant enumerants in declaration order.
// This generator emits APIs typed as FixpipePreQuantMode, so accepting cases
// from another HIVM_I32Enum would generate invalid C++ enum references.
static SmallVector<const Record *>
getPreQuantEnumCases(const RecordKeeper &records) {
  SmallVector<const Record *> cases;
  const Record *enumDef = records.getDef("HIVM_FixpipePreQuantModeEnum");
  if (!enumDef)
    return cases;
  for (const Record *record : enumDef->getValueAsListOfDefs("enumerants"))
    cases.push_back(record);
  return cases;
}

static bool validatePreQuantRecords(const RecordKeeper &records) {
  bool hadError = false;
  auto cases = getPreQuantCases(records);

  StringSet<> seenSymbols;
  for (const Record *preQuantCase : cases) {
    StringRef symbol = preQuantCase->getValueAsString(kSymbol);
    if (!seenSymbols.insert(symbol).second) {
      PrintError(preQuantCase->getLoc(),
                 Twine("duplicate Fixpipe pre-quant case symbol: ") + symbol);
      hadError = true;
    }

    auto signatures = preQuantCase->getValueAsListOfDefs(kTypeSignatures);
    if (signatures.empty()) {
      PrintError(preQuantCase->getLoc(),
                 Twine("Fixpipe pre-quant case '") + symbol +
                     "' must declare at least one src/dst type signature");
      hadError = true;
    }

    StringSet<> seenSignatures;
    for (const Record *signature : signatures) {
      StringRef srcType = signature->getValueAsDef(kSrcType)->getName();
      StringRef dstType = signature->getValueAsDef(kDstType)->getName();
      ElementTypeContract srcContract;
      ElementTypeContract dstContract;
      hadError |=
          getElementTypeContract(signature, /*isSrc=*/true, srcContract);
      hadError |=
          getElementTypeContract(signature, /*isSrc=*/false, dstContract);

      std::string signatureKey = (Twine(srcType) + "->" + dstType).str();
      if (!seenSignatures.insert(signatureKey).second) {
        PrintError(signature->getLoc(),
                   Twine("duplicate signature '") + signatureKey +
                       "' in Fixpipe pre-quant case '" + symbol + "'");
        hadError = true;
      }
    }
  }

  // Every enumerant of HIVM_FixpipePreQuantModeEnum must carry typed
  // signatures; otherwise a newly registered mode silently bypasses the
  // contract.
  auto enumCases = getPreQuantEnumCases(records);
  for (const Record *enumCase : enumCases) {
    if (enumCase->isSubClassOf("HIVM_FixpipePreQuantCase"))
      continue;
    PrintError(enumCase->getLoc(),
               Twine("Fixpipe pre-quant enum case '") +
                   enumCase->getValueAsString(kSymbol) +
                   "' must derive from HIVM_FixpipePreQuantCase and declare "
                   "its src/dst type signatures");
    hadError = true;
  }

  // And a typed case that is not part of the enum would generate dead
  // contract code.
  StringSet<> enumSymbols;
  for (const Record *enumCase : enumCases)
    enumSymbols.insert(enumCase->getValueAsString(kSymbol));
  for (const Record *preQuantCase : cases) {
    StringRef symbol = preQuantCase->getValueAsString(kSymbol);
    if (enumSymbols.contains(symbol))
      continue;
    PrintError(preQuantCase->getLoc(), Twine("Fixpipe pre-quant case '") +
                                           symbol +
                                           "' is not referenced by "
                                           "HIVM_FixpipePreQuantModeEnum");
    hadError = true;
  }

  return hadError;
}

static bool emitFixpipePreQuantContractDecls(const RecordKeeper &records,
                                             raw_ostream &os) {
  if (validatePreQuantRecords(records))
    return true;

  emitSourceFileHeader("Fixpipe Pre-Quant Type Contract Declarations", os,
                       records);
  os << R"(
/// Runtime descriptor of one src/dst element-type signature. Spellings are
/// generated from the ODS Type records and are stable diagnostic text.
struct FixpipePreQuantSignatureDesc {
  ::llvm::StringRef srcSpelling;
  ::llvm::StringRef dstSpelling;
  bool (*matches)(::mlir::Type srcElementType, ::mlir::Type dstElementType);
};

/// All target-independent src/dst signatures allowed for \p mode.
::llvm::ArrayRef<FixpipePreQuantSignatureDesc>
getFixpipePreQuantSignatures(::mlir::hivm::FixpipePreQuantMode mode);

/// Verifies that \p srcElementType -> \p dstElementType satisfies the
/// target-independent contract of \p mode. Diagnostics name the mode, the
/// allowed signatures and the actual signature.
::mlir::LogicalResult verifyFixpipePreQuantSignature(
    ::mlir::hivm::FixpipePreQuantMode mode, ::mlir::Type srcElementType,
    ::mlir::Type dstElementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);

/// Returns every mode whose contract accepts \p srcElementType ->
/// \p dstElementType. One signature may satisfy several modes; the final
/// choice belongs to the caller's policy.
::llvm::SmallVector<::mlir::hivm::FixpipePreQuantMode>
getFixpipePreQuantCandidates(::mlir::Type srcElementType,
                             ::mlir::Type dstElementType);
)";
  return false;
}

static bool emitFixpipePreQuantContractDefs(const RecordKeeper &records,
                                            raw_ostream &os) {
  if (validatePreQuantRecords(records))
    return true;

  auto cases = getPreQuantCases(records);
  auto enumCases = getPreQuantEnumCases(records);

  emitSourceFileHeader("Fixpipe Pre-Quant Type Contract Definitions", os,
                       records);
  os << R"(
#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace {
)";

  // One matcher per signature record. The predicate expressions come from the
  // ODS Type records; mode names are never parsed.
  // The matcher name is case-qualified: the same signature record may be
  // shared by several modes (e.g. Sig_F32_F32 backs both NO_QUANT and
  // QF322F32_PRE).
  for (const Record *preQuantCase : cases) {
    for (const Record *signature :
         preQuantCase->getValueAsListOfDefs(kTypeSignatures)) {
      ElementTypeContract srcContract;
      ElementTypeContract dstContract;
      getElementTypeContract(signature, /*isSrc=*/true, srcContract);
      getElementTypeContract(signature, /*isSrc=*/false, dstContract);
      os << "static bool match" << preQuantCase->getName() << "_"
         << signature->getName()
         << "(::mlir::Type srcElementType, ::mlir::Type dstElementType) {\n"
         << "  return " << srcContract.predicate << " && "
         << dstContract.predicate << ";\n"
         << "}\n";
    }
  }

  for (const Record *preQuantCase : cases) {
    auto signatures = preQuantCase->getValueAsListOfDefs(kTypeSignatures);
    os << "static const FixpipePreQuantSignatureDesc k"
       << preQuantCase->getName() << "Signatures[] = {";
    for (auto [index, signature] : llvm::enumerate(signatures)) {
      ElementTypeContract srcContract;
      ElementTypeContract dstContract;
      getElementTypeContract(signature, /*isSrc=*/true, srcContract);
      getElementTypeContract(signature, /*isSrc=*/false, dstContract);
      if (index)
        os << ", ";
      os << "{\"" << srcContract.summary << "\", \"" << dstContract.summary
         << "\", match" << preQuantCase->getName() << "_"
         << signature->getName() << "}";
    }
    os << "};\n";
  }

  os << R"(} // namespace

::llvm::ArrayRef<FixpipePreQuantSignatureDesc>
getFixpipePreQuantSignatures(::mlir::hivm::FixpipePreQuantMode mode) {
  switch (mode) {
)";
  for (const Record *enumCase : enumCases) {
    assert(enumCase->isSubClassOf("HIVM_FixpipePreQuantCase") &&
           "validated enum case must carry typed signatures");
    os << "  case ::mlir::hivm::FixpipePreQuantMode::"
       << enumCase->getValueAsString(kSymbol) << ": return k"
       << enumCase->getName() << "Signatures;\n";
  }
  os << R"(  }
  llvm_unreachable("unknown FixpipePreQuantMode");
}

::mlir::LogicalResult verifyFixpipePreQuantSignature(
    ::mlir::hivm::FixpipePreQuantMode mode, ::mlir::Type srcElementType,
    ::mlir::Type dstElementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::llvm::ArrayRef<FixpipePreQuantSignatureDesc> signatures =
      getFixpipePreQuantSignatures(mode);
  for (const FixpipePreQuantSignatureDesc &signature : signatures)
    if (signature.matches(srcElementType, dstElementType))
      return ::mlir::success();

  auto diag = emitError() << "pre_quant mode '"
                          << ::mlir::hivm::stringifyFixpipePreQuantMode(mode)
                          << "' requires ";
  if (signatures.size() == 1) {
    diag << "src/dst element type signature "
         << signatures.front().srcSpelling << " -> "
         << signatures.front().dstSpelling;
  } else {
    diag << "one of [";
    ::llvm::interleaveComma(
        signatures, diag, [&](const FixpipePreQuantSignatureDesc &signature) {
          diag << signature.srcSpelling << " -> " << signature.dstSpelling;
        });
    diag << "]";
  }
  diag << ", but got " << srcElementType << " -> " << dstElementType;
  return ::mlir::failure();
}

::llvm::SmallVector<::mlir::hivm::FixpipePreQuantMode>
getFixpipePreQuantCandidates(::mlir::Type srcElementType,
                             ::mlir::Type dstElementType) {
  ::llvm::SmallVector<::mlir::hivm::FixpipePreQuantMode> candidates;
)";
  for (const Record *enumCase : enumCases) {
    os << "  for (const FixpipePreQuantSignatureDesc &signature : k"
       << enumCase->getName() << "Signatures)\n"
       << "    if (signature.matches(srcElementType, dstElementType)) {\n"
       << "      candidates.push_back(::mlir::hivm::FixpipePreQuantMode::"
       << enumCase->getValueAsString(kSymbol) << ");\n"
       << "      break;\n"
       << "    }\n";
  }
  os << "  return candidates;\n}\n";
  return false;
}

} // namespace

static mlir::GenRegistration genFixpipePreQuantContractDecls(
    "gen-fixpipe-pre-quant-contract-decls",
    "Generate Fixpipe pre-quant type contract declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitFixpipePreQuantContractDecls(records, os);
    });

static mlir::GenRegistration genFixpipePreQuantContractDefs(
    "gen-fixpipe-pre-quant-contract-defs",
    "Generate Fixpipe pre-quant type contract definitions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitFixpipePreQuantContractDefs(records, os);
    });
