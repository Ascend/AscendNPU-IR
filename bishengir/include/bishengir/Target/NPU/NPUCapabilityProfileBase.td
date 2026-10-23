//===- NPUCapabilityProfileBase.td - NPU profiles -*- tablegen -*-===//
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

#ifndef BISHENGIR_TARGET_NPU_NPUCAPABILITYPROFILEBASE_TD
#define BISHENGIR_TARGET_NPU_NPUCAPABILITYPROFILEBASE_TD

include "bishengir/Target/Capability/CapabilityBase.td"
include "bishengir/Target/NPU/NPUArchitecture.td"

//===----------------------------------------------------------------------===//
// Architecture Capability Profiles
//===----------------------------------------------------------------------===//

// The common schema defines the query; this layer groups support facts by
// architecture. Each coverage entry references one capability and its cases:
//
//   ArchitectureCapabilityProfile
//     |-- Architecture: NPUArchitecture
//     `-- Capabilities[]: ArchitectureCapabilityCoverage
//           |-- Capability: CapabilityKey --> RequestSchema
//           |-- Cases[]: SupportCase (alternative supported combinations)
//           `-- Completeness: ProfileCompleteness
//
// A matching case establishes Native support. If no case matches, the
// coverage's completeness determines whether the answer is Unsupported or
// Unknown. Completeness is not a property of an individual SupportCase.

// Completeness belongs to the entire capability coverage, not an individual
// support row. The owner must choose explicitly whether a miss establishes
// non-Native support or merely reflects an incomplete database.
class ArchitectureCapabilityCoverage<CapabilityKey capability,
                                     list<SupportCase> cases,
                                     ProfileCompleteness completeness> {
  CapabilityKey Capability = capability;
  list<SupportCase> Cases = cases;
  ProfileCompleteness Completeness = completeness;
}

// Profiles share the architecture identity used by device specs and features.
// A device SKU does not duplicate the support rules of its architecture.
class ArchitectureCapabilityProfile<
    NPUArchitecture architecture,
    list<ArchitectureCapabilityCoverage> capabilities> {
  NPUArchitecture Architecture = architecture;
  list<ArchitectureCapabilityCoverage> Capabilities = capabilities;
}

#endif // BISHENGIR_TARGET_NPU_NPUCAPABILITYPROFILEBASE_TD
