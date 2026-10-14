//===- NPUArchitectureFeatureBase.td - NPU feature schema -*- tablegen -*-===//
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

#ifndef BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATUREBASE_TD
#define BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATUREBASE_TD

include "bishengir/Target/NPU/NPUArchitecture.td"

//===----------------------------------------------------------------------===//
// Common feature declarations
//===----------------------------------------------------------------------===//

// Architecture features answer target-only questions. They do not depend on
// an Op, type signature, or attribute request.
//
// A feature declares its type and values; each architecture profile selects
// one setting per feature. Adding a feature does not add another template
// argument to ArchitectureFeatureProfile.
class ArchitectureFeature<string cppName, string name = "",
                          string accessor = ""> {
  // Generated C++ type name, for example "ExecutionModel".
  string CppName = cppName;
  // Stable spelling override; empty derives kebab-case from CppName.
  string Name = name;
  // Query-name override; empty derives "get" + CppName.
  string Accessor = accessor;
}

// Base for a feature setting selected by an architecture profile.
class ArchitectureFeatureSetting<ArchitectureFeature feature> {
  ArchitectureFeature Feature = feature;
}

//===----------------------------------------------------------------------===//
// Enum architecture features
//===----------------------------------------------------------------------===//

// Declares a feature whose settings are enum values. The generator emits a
// typed C++ enum, a profile field, and a free query function for the feature.
class EnumArchitectureFeature<string cppName, string name = "",
                              string accessor = "">
    : ArchitectureFeature<cppName, name, accessor>;

// Declares an enum value that can be selected by an architecture profile.
class EnumArchitectureFeatureValue<EnumArchitectureFeature feature,
                                   string cppName, string name = "">
    : ArchitectureFeatureSetting<feature> {
  // Enum case suffix; the generator adds the Google-style "k" prefix.
  string CppName = cppName;
  // Stable spelling override; empty derives kebab-case from CppName.
  string Name = name;
}

//===----------------------------------------------------------------------===//
// Architecture feature profiles
//===----------------------------------------------------------------------===//

// Features belong to the instruction architecture, not individual SKUs.
// Every declared architecture must have exactly one profile, and every
// profile must select exactly one setting for every declared feature.
class ArchitectureFeatureProfile<NPUArchitecture architecture,
                                 list<ArchitectureFeatureSetting> features> {
  NPUArchitecture Architecture = architecture;
  list<ArchitectureFeatureSetting> Features = features;
}

#endif // BISHENGIR_TARGET_NPU_NPUARCHITECTUREFEATUREBASE_TD
