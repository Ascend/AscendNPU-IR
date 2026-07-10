//===- Options.td - BiShengIR Options -------------------------*-tablegen-*-==//
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

#ifndef BISHENGIR_TOOLS_CONFIG_OPTIONS_OPTIONS_TD
#define BISHENGIR_TOOLS_CONFIG_OPTIONS_OPTIONS_TD

/// This class holds information regarding an abstract Option.
///
/// An "Option" can have many forms, for example:
///   a) a binary tool's command line option
///   b) a pipeline pass's pass option
///   c) a field in a cpp class where the value can be set or accessed
///
/// When we have to register a new option that is exposed to the users,
/// there are many places that needs to be modified.
/// This tablegen class is designed to handle this, as it will automatically
/// generate code that register and handle the option.
class Option {
  // The name of the Record will be used to construct the cpp variable name
  // as well as the CLI argument.
  // For example:
  // ```tablegen
  //   def FooBar : Option {}
  // ```
  // The corresponding cpp variable name is "fooBar".
  // The argument name is "foo-bar".
  // Note that we expect the name to be camel case (with capitalized letter).

  // A description for this option.
  string description = "";

  // The C++ type of the option.
  string type = ?;

  // The default value of the option.
  string defaultValue = ?;

  // A set of additional flags to pass along to the option constructor.
  string additionalOptFlags = "";

  // Whether the option has external storage.
  //
  // This is only used for generating compile tool options.
  // Note that the external storage location is the cppName appended
  // with "Flag".
  // For example:
  //   cppName = "fooBar";
  //   location = "fooBarFlag";
  bit externalStorage = 1;

  // Explicit external object / field override.
  //
  // This is the structured form of external storage, used by options that
  // write into an object outside the generated compile config.
  string externalStateObject = "";
  string externalStateField = "";

  // Explicit external storage location override.
  //
  // This remains as an escape hatch for historical field compatibility.
  // When empty, the location is derived from
  // `externalStateObject.externalStateField`, or `<derivedCppName>Flag`
  // (`<derivedCppName>Flags` for list options).
  string externalStorageLocation = "";

  // The groups that this option belong to.
  //
  // If specified, macro guards will be generated to wrap the
  // option. For example:
  //   passGroup = ["HFUSION"]
  //
  // Generated code:
  // ```cpp
  //   #ifdef GEN_HFUSION_OPTION_REGISTRATION
  //     XXXX
  //   #undef
  // ```
  list<string> passGroup = [];

  // Scope of the option surface.
  //
  // SHARED: present on both A3 and A5 surfaces.
  // A3_ONLY: kept only for A3 surface / migration tracking.
  // A5_ONLY: introduced only on A5 surface.
  list<string> optionScope = ["SHARED"];

  // The list of categories for the CLI option.
  list<string> compileOptionCategories = [];

  // The compile config's cpp class name.
  //
  // This is used to generate getter/setters. For example:
  //   compileConfigName = Config
  //
  // Generated code:
  // ```cpp
  //   Config setFoo(int64_t val) {
  //     foo = val;
  //     return *this;
  //   }
  // ```
  string compileConfigName = ?;

  // Whether to emit the getter/setter methods for this config.
  bit emitGetterSetter = 1;

  // Whether to emit option's command line registration.
  bit emitOptionRegistration = 1;
}

/// Comma separated list option.
class ListOption : Option {
  let defaultValue = "";
}

/// Marker for a CLI-visible option that writes into an external storage object
/// instead of the generated compile config class.
class ExternalStateOption;

/// Marker for config state that exists in the generated compile config but is
/// intentionally not surfaced as a generated command line option.
class InternalStateOption;

#endif // BISHENGIR_TOOLS_CONFIG_OPTIONS_OPTIONS_TD
