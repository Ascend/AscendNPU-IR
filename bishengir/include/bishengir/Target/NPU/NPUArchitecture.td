//===- NPUArchitecture.td - NPU architecture vocabulary -*- tablegen -*-===//
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

#ifndef BISHENGIR_TARGET_NPU_NPUARCHITECTURE_TD
#define BISHENGIR_TARGET_NPU_NPUARCHITECTURE_TD

// Stable instruction-architecture identity shared by TargetSpec and native
// capability profiles. Record names use the project TD convention (Dav_C310);
// the generator emits Google-style C++ enumerators (kDavC310). Name is the
// external spelling retained for IR serialization, diagnostics and tool flags.
class NPUArchitecture<string name> { string Name = name; }

def Dav_C220 : NPUArchitecture<"dav-c220">;
def Dav_M300 : NPUArchitecture<"dav-m300">;
def Dav_C310 : NPUArchitecture<"dav-c310">;

#endif // BISHENGIR_TARGET_NPU_NPUARCHITECTURE_TD
