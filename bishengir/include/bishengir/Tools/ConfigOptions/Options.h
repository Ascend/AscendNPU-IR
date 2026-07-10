//===- Options.h - TableGen Option definitions ------------------*- C++ -*-===//
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
#ifndef BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H
#define BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class Record;
} // namespace llvm

namespace bishengir {
namespace tblgen {

constexpr llvm::StringLiteral kOptionClassName = "Option";
namespace OptionFields {
constexpr llvm::StringLiteral kType = "type";
constexpr llvm::StringLiteral kDefaultValue = "defaultValue";
constexpr llvm::StringLiteral kDescription = "description";
constexpr llvm::StringLiteral kAdditionalOptFlags = "additionalOptFlags";
constexpr llvm::StringLiteral kExternalStorage = "externalStorage";
constexpr llvm::StringLiteral kExternalStateObject = "externalStateObject";
constexpr llvm::StringLiteral kExternalStateField = "externalStateField";
constexpr llvm::StringLiteral kExternalStorageLocation =
    "externalStorageLocation";
constexpr llvm::StringLiteral kPassGroup = "passGroup";
constexpr llvm::StringLiteral kOptionScope = "optionScope";
constexpr llvm::StringLiteral kCompileOptionCategories =
    "compileOptionCategories";
constexpr llvm::StringLiteral kCompileConfigName = "compileConfigName";
constexpr llvm::StringLiteral kEmitGetterSetter = "emitGetterSetter";
constexpr llvm::StringLiteral kEmitOptionRegistration =
    "emitOptionRegistration";
constexpr llvm::StringLiteral kListOption = "ListOption";
constexpr llvm::StringLiteral kExternalStateOption = "ExternalStateOption";
constexpr llvm::StringLiteral kInternalStateOption = "InternalStateOption";
} // namespace OptionFields

class ConfigOption {
public:
  explicit ConfigOption(const llvm::Record *def);

  /// Get the original record.
  const llvm::Record *getDef() const { return def; }

  /// Return the name for the C++ option variable.
  mlir::StringRef getCppVariableName() const;

  /// Return the name for the capitalized C++ option variable.
  mlir::StringRef getCapitalizedCppVariableName() const;

  /// Return the command line argument to use for this option.
  mlir::StringRef getArgument() const;

  /// Return the C++ type of the option.
  mlir::StringRef getType() const;

  /// Return the default value of the option.
  std::optional<mlir::StringRef> getDefaultValue() const;

  /// Return the description for this option.
  mlir::StringRef getDescription() const;

  /// Return the additional flags passed to the option constructor.
  std::optional<mlir::StringRef> getAdditionalFlags() const;

  /// Return the external storage location for the config option.
  std::optional<mlir::StringRef> getExternalStorageLocation() const;

  /// Return whether CLI registration should use external storage.
  bool shouldUseCLIExternalStorage() const;

  /// Return the external storage location used by CLI registration.
  std::optional<mlir::StringRef> getCLIExternalStorageLocation() const;

  /// Return the list of pass pipeline groups that this option belongs to.
  std::vector<mlir::StringRef> getPassGroups() const;

  /// Return the declared option surface scopes.
  std::vector<mlir::StringRef> getOptionScopes() const;

  /// Return the list of categories that this option belongs to.
  std::vector<mlir::StringRef> getOptionCategories() const;

  /// Returns the compile config's class name.
  std::optional<mlir::StringRef> getCompileConfigName() const;

  /// Returns whether to emit command line option registration.
  bool shouldRegisterCLIOption() const;

  /// Returns whether to emit the config field declaration.
  bool shouldEmitConfigField() const;

  /// Returns whether to emit getter or setter methods.
  bool shouldEmitGetterSetter() const;

  /// Returns whether this option should be visible as a pass / pass-pipeline
  /// option in generated code.
  bool shouldEmitPassOption() const;

  /// Flag indicating if this is a list option.
  bool isListOption() const;

  /// Returns whether this option only exists as internal compile config state.
  bool isInternalStateOption() const;

  /// Returns whether this option is tracked as A3-only in the shared schema.
  bool isA3OnlyOption() const;

  /// Returns whether this option exists only on the A5 surface.
  bool isA5OnlyOption() const;

  /// Returns whether this option is shared by both surfaces.
  bool isSharedOption() const;

  /// Returns whether this option stores into an external object instead of the
  /// generated compile config.
  bool isExternalStateOption() const;

private:
  const llvm::Record *def;

  std::string cppName{};
  std::string capitalizedCppName{};
  std::string argument{};
  std::string externalStorageLocation{};
};

/// Get the cpp container type for the config variable.
/// If the config is a list option, the container type is
///   `std::vector<Type>`
/// Otherwise, it's just `Type`.
std::string getContainerType(const ConfigOption &opt);

} // namespace tblgen
} // namespace bishengir

#endif // BISHENGIR_TOOLS_CONFIGOPTIONS_OPTIONS_H
