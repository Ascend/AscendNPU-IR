//===- NPUArchitectureGen.cpp - NPU architecture generation ---------------===//
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

std::string getNPUArchitectureEnumerator(const Record *architecture) {
  std::string result = "k";
  bool capitalizeNext = false;
  for (char character : architecture->getName()) {
    if (character == '_') {
      capitalizeNext = true;
      continue;
    }
    result.push_back(capitalizeNext ? llvm::toUpper(character) : character);
    capitalizeNext = false;
  }
  return result;
}

namespace {

static bool isValidCppIdentifier(StringRef name) {
  if (name.empty() || (!llvm::isAlpha(name.front()) && name.front() != '_'))
    return false;
  for (char character : name.drop_front())
    if (!llvm::isAlnum(character) && character != '_')
      return false;
  return true;
}

static bool emitNPUArchitectureDecls(const RecordKeeper &records,
                                     raw_ostream &os) {
  auto architectures = records.getAllDerivedDefinitions("NPUArchitecture");
  StringSet<> spellings;
  StringSet<> enumerators;
  bool hadError = false;
  for (const Record *architecture : architectures) {
    StringRef spelling = architecture->getValueAsString("Name");
    std::string enumerator = getNPUArchitectureEnumerator(architecture);
    if (spelling.empty()) {
      PrintError(architecture->getLoc(),
                 "NPU architecture spelling must not be empty");
      hadError = true;
    } else if (spelling == "unknown") {
      PrintError(architecture->getLoc(),
                 "NPU architecture spelling 'unknown' is reserved");
      hadError = true;
    } else if (!spellings.insert(spelling).second) {
      PrintError(architecture->getLoc(),
                 Twine("duplicate NPU architecture spelling: ") + spelling);
      hadError = true;
    }
    if (enumerator == "k") {
      PrintError(architecture->getLoc(),
                 "NPU architecture C++ enumerator has an empty name stem");
      hadError = true;
    } else if (!isValidCppIdentifier(enumerator)) {
      PrintError(architecture->getLoc(),
                 Twine("invalid NPU architecture C++ enumerator: ") +
                     enumerator);
      hadError = true;
    } else if (enumerator == "kUnknown") {
      PrintError(architecture->getLoc(),
                 "NPU architecture C++ enumerator 'kUnknown' is reserved");
      hadError = true;
    } else if (!enumerators.insert(enumerator).second) {
      PrintError(architecture->getLoc(),
                 Twine("duplicate NPU architecture C++ enumerator: ") +
                     enumerator);
      hadError = true;
    }
  }
  if (hadError)
    return true;

  emitSourceFileHeader("NPU Architecture Declarations", os, records);
  os << R"(
#ifndef BISHENGIR_NPU_ARCHITECTURE_H
#define BISHENGIR_NPU_ARCHITECTURE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace bishengir {
namespace npu {

enum class NPUArchitectureKind {
)";
  for (const Record *architecture : architectures)
    os << "  " << getNPUArchitectureEnumerator(architecture) << ",\n";
  os << R"(  kUnknown,
};

inline ::llvm::StringRef
stringifyNPUArchitecture(NPUArchitectureKind architecture) {
  switch (architecture) {
)";
  for (const Record *architecture : architectures) {
    os << "  case NPUArchitectureKind::"
       << getNPUArchitectureEnumerator(architecture) << ": return \"";
    os.write_escaped(architecture->getValueAsString("Name"));
    os << "\";\n";
  }
  os << R"(  case NPUArchitectureKind::kUnknown: return "unknown";
  }
  llvm_unreachable("unknown NPUArchitectureKind value");
}

inline NPUArchitectureKind
symbolizeNPUArchitecture(::llvm::StringRef name) {
  return ::llvm::StringSwitch<NPUArchitectureKind>(name)
)";
  for (const Record *architecture : architectures) {
    os << "      .Case(\"";
    os.write_escaped(architecture->getValueAsString("Name"));
    os << "\", NPUArchitectureKind::"
       << getNPUArchitectureEnumerator(architecture) << ")\n";
  }
  os << R"(      .Default(NPUArchitectureKind::kUnknown);
}

} // namespace npu
} // namespace bishengir

#endif // BISHENGIR_NPU_ARCHITECTURE_H
)";
  return false;
}

static mlir::GenRegistration
    genNPUArchitectureDecls("gen-npu-architecture-decls",
                            "Generate NPU architecture identity declarations",
                            [](const RecordKeeper &records, raw_ostream &os) {
                              return emitNPUArchitectureDecls(records, os);
                            });

} // namespace
