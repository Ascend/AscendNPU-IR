//===- SIMTTrigCommon.h - Shared constants for template-lib trig ops -*- C++
//-*-===//
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
// Shared compile-time constants between the device-side template library
// (bishengir/lib/Template/lib/RegBase/Vector/Sin.cpp) and the host-side
// conversion (bishengir/lib/Conversion/MathToHFusion/MathToHFusion.cpp).
//
// Both sides must agree on the highest memref rank that has a
// template-library SIMT entry point. Changing SIN_SIMT_MAX_RANK here
// automatically updates both the DEFINE_SIN_SIMT_ENTRY instantiations in
// Sin.cpp and kMaxTemplateLibTrigRank in MathToHFusion.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENG_TEMPLATE_SIMT_TRIG_COMMON_H
#define BISHENG_TEMPLATE_SIMT_TRIG_COMMON_H

#ifndef SIN_SIMT_MAX_RANK
#define SIN_SIMT_MAX_RANK 5
#endif

#endif // BISHENG_TEMPLATE_SIMT_TRIG_COMMON_H
