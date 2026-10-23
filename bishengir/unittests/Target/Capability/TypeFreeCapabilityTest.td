//===- TypeFreeCapabilityTest.td --------------------------*- tablegen -*-===//
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

include "bishengir/Target/Capability/CapabilityBase.td"

// Empty and parameter-only schemas need no type slots or type declarations.
def Empty : CapabilityKey<RequestSchema<[]>>;
def EmptyCase : SupportCase<Empty, []>;
def Enabled : BoolAttrCapabilityParameter<>;
def ParameterOnly : CapabilityKey<RequestSchema<[], [Enabled]>>;
def ParameterCase : SupportCase<ParameterOnly, [AnyValue<Enabled>]>;
