//===- CompileConfigGen.cpp - Compile tool's config generation --*- C++ -*-===//
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
// CompileConfigGen generates fields and methods related to compile tool's
// config class.
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
static const char *const kGetSetMethodDefs = R"(
public:
  {0} &set{1}(const {2}& val) {
    {3} = val;
    return *this;
  }

  {2} get{1}() const {
    return {3};
  }

)";

/// Main entry to emit compile configs.
bool emitCompileConfigs(const RecordKeeper &records, raw_ostream &OS) {
  std::vector<Record *> specs =
      records.getAllDerivedDefinitions(kOptionClassName);
  for (const Record *R : specs) {
    ConfigOption cfgOpt(R);

    if (!cfgOpt.getExternalStorageLocation())
      continue;

    if (cfgOpt.shouldEmitConfigField()) {
      // Generate variable declaration (protected)
      // Example:
      // ```cpp
      //   int64_t externalVar = 42;
      // ```
      OS << "protected:\n";
      OS << "  " << getContainerType(cfgOpt);
      OS << " " << cfgOpt.getExternalStorageLocation();

      if (std::optional<StringRef> defaultVal = cfgOpt.getDefaultValue()) {
        OS << " = " << defaultVal << ";\n";
      } else {
        OS << ";\n";
      }
    }

    if (!cfgOpt.shouldEmitGetterSetter())
      continue;

    // Generate getter and setter methods (public)
    if (!cfgOpt.getCompileConfigName().has_value()) {
      PrintFatalError(R, "Unknown config name");
      return true;
    }
    OS << formatv(kGetSetMethodDefs,
                  /*config type*/ cfgOpt.getCompileConfigName(),
                  /*cpp variable name*/ cfgOpt.getCapitalizedCppVariableName(),
                  /*cpp type name*/ getContainerType(cfgOpt),
                  /*location*/ cfgOpt.getExternalStorageLocation());
  }
  return false;
}

} // namespace

// Registers the generator to bishengir-options-tblgen.
static mlir::GenRegistration
    genCompileConfigs("gen-compile-configs",
                      "Generate config for bishengir compile tool",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitCompileConfigs(records, os);
                      });
