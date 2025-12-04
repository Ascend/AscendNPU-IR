//===- NPUTargetSpec.td ------------------------------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPEC_TD
#define BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPEC_TD

/// Base class for the Target Specification info.
class TargetSpec <string name> {
  // The target device's name.
  string Name = name;

  // The number of physical AI Cores.
  int AiCoreCount = ?;

  // The number of physical AI Cube Cores.
  int CubeCoreCount = ?;

  // The number of physical AI Vector Cores.
  int VectorCoreCount = ?;

  // Size of unified buffer in bits.
  int UbSize = ?;

  // Size of L1 buffer in bits.
  int L1Size = ?;

  // Size of L0a buffer in bits.
  int L0aSize = ?;

  // Size of L0b buffer in bits.
  int L0bSize = ?;

  // Size of L0c buffer in bits.
  int L0cSize = ?;

  // Align size of UB buffer in bits.
  int UbAlignSize = ?;

  // Align size of L1 buffer in bits.
  int L1AlignSize = ?;

  // Align size of L0c buffer in bits.
  int L0cAlignSize = ?;
}

//===----------------------------------------------------------------------===//
// Ascend910B
//===----------------------------------------------------------------------===//

class Ascend910B_BaseSpec <string name> : TargetSpec<name> {
  let UbSize = 1572864; // bits = 192KB
  let L1Size = 4194304; // bits = 512KB
  let L0aSize = 524288; // bits = 64KB
  let L0bSize = 524288; // bits = 64KB
  let L0cSize = 1048576; // bits = 128KB
  let UbAlignSize = 256; // bits = 32B
  let L1AlignSize = 256; // bits = 32B
  let L0cAlignSize = 4096; // bits = 512B
}

def Ascend910B1Spec : Ascend910B_BaseSpec<"Ascend910B1"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

def Ascend910B2Spec : Ascend910B_BaseSpec<"Ascend910B2"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

def Ascend910B3Spec : Ascend910B_BaseSpec<"Ascend910B3"> {
  let AiCoreCount = 20;
  let CubeCoreCount = 20;
  let VectorCoreCount = 40;
}

def Ascend910B4Spec : Ascend910B_BaseSpec<"Ascend910B4"> {
  let AiCoreCount = 20;
  let CubeCoreCount = 20;
  let VectorCoreCount = 40;
}

def Ascend910B2CSpec : Ascend910B_BaseSpec<"Ascend910B2C"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

//===----------------------------------------------------------------------===//
// Ascend910_93
//===----------------------------------------------------------------------===//

class Ascend910_93_BaseSpec <string name> : TargetSpec<name> {
  let UbSize = 1572864; // bits = 192KB
  let L1Size = 4194304; // bits = 512KB
  let L0aSize = 524288; // bits = 64KB
  let L0bSize = 524288; // bits = 64KB
  let L0cSize = 1048576; // bits = 128KB
  let UbAlignSize = 256; // bits = 32B
  let L1AlignSize = 256; // bits = 32B
  let L0cAlignSize = 4096; // bits = 512B
}

def Ascend910_9362_Spec : Ascend910_93_BaseSpec<"Ascend910_9362"> {
  let AiCoreCount = 20;
  let CubeCoreCount = 20;
  let VectorCoreCount = 40;
}

def Ascend910_9372_Spec : Ascend910_93_BaseSpec<"Ascend910_9372"> {
  let AiCoreCount = 20;
  let CubeCoreCount = 20;
  let VectorCoreCount = 40;
}

def Ascend910_9381_Spec : Ascend910_93_BaseSpec<"Ascend910_9381"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

def Ascend910_9382_Spec : Ascend910_93_BaseSpec<"Ascend910_9382"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

def Ascend910_9391_Spec : Ascend910_93_BaseSpec<"Ascend910_9391"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

def Ascend910_9392_Spec : Ascend910_93_BaseSpec<"Ascend910_9392"> {
  let AiCoreCount = 24;
  let CubeCoreCount = 24;
  let VectorCoreCount = 48;
}

//===----------------------------------------------------------------------===//
// Ascend910_95
//===----------------------------------------------------------------------===//

class Ascend910_95_BaseSpec <string name> : TargetSpec<name> {
  let UbSize = -1;
  let L1Size = -1;
  let L0aSize = -1;
  let L0bSize = -1;
  let L0cSize = -1;
  let UbAlignSize = -1;
  let L1AlignSize = -1;
  let L0cAlignSize = -1;
  let AiCoreCount = -1;
  let CubeCoreCount = -1;
  let VectorCoreCount = -1;
}

def Ascend910_950z_Spec : Ascend910_95_BaseSpec<"Ascend910_950z">{}

def Ascend910_9579_Spec : Ascend910_95_BaseSpec<"Ascend910_9579">{}

def Ascend910_957b_Spec : Ascend910_95_BaseSpec<"Ascend910_957b">{}

def Ascend910_957d_Spec : Ascend910_95_BaseSpec<"Ascend910_957d">{}

def Ascend910_9581_Spec : Ascend910_95_BaseSpec<"Ascend910_9581">{}

def Ascend910_9589_Spec : Ascend910_95_BaseSpec<"Ascend910_9589">{}

def Ascend910_958a_Spec : Ascend910_95_BaseSpec<"Ascend910_958a">{}

def Ascend910_958b_Spec : Ascend910_95_BaseSpec<"Ascend910_958b">{}

def Ascend910_9599_Spec : Ascend910_95_BaseSpec<"Ascend910_9599">{}

#endif // BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPEC_TD
