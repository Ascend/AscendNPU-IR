//===- NPUArchitectureGen.h - NPU architecture generation -----*- C++ -*-===//
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

#ifndef BISHENGIR_TOOLS_TARGET_SPEC_TBLGEN_NPUARCHITECTUREGEN_H
#define BISHENGIR_TOOLS_TARGET_SPEC_TBLGEN_NPUARCHITECTUREGEN_H

#include <string>

namespace llvm {
class Record;
} // namespace llvm

/// Derive a C++ enumerator from a TD record name, e.g. Dav_C220 -> kDavC220.
std::string getNPUArchitectureEnumerator(const llvm::Record *architecture);

#endif // BISHENGIR_TOOLS_TARGET_SPEC_TBLGEN_NPUARCHITECTUREGEN_H
