//===- NPUTargetSpec.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  // Minimal size of data cache for simt in bits
  int MinimalDCacheSize = 0;

  // Maximum size of data cache for simt in bits
  int MaximumDCacheSize = 0;

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

  // Arch
  string Arch = ?;
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
  let Arch = "dav-c220";
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
  let Arch = "dav-c220";
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
// Ascend310B
//===----------------------------------------------------------------------===//

class Ascend310B_BaseSpec <string name> : TargetSpec<name> {
  let UbSize = 2097152; // bits = 256KB
  let L1Size = 8388608; // bits = 1024KB
  let L0aSize = 524288; // bits = 64KB
  let L0bSize = 524288; // bits = 64KB
  let L0cSize = 1048576; // bits = 128KB
  let UbAlignSize = 256; // bits = 32B
  let L1AlignSize = 256; // bits = 32B
  let L0cAlignSize = 4096; // bits = 512B
  let Arch = "dav-m300";
}

def Ascend310B1Spec : Ascend310B_BaseSpec<"Ascend310B1"> {
  let AiCoreCount = 1;
  let CubeCoreCount = 1;
  let VectorCoreCount = 1;
}

def Ascend310B2Spec : Ascend310B_BaseSpec<"Ascend310B2"> {
  let AiCoreCount = 1;
  let CubeCoreCount = 1;
  let VectorCoreCount = 1;
}

def Ascend310B3Spec : Ascend310B_BaseSpec<"Ascend310B3"> {
  let AiCoreCount = 1;
  let CubeCoreCount = 1;
  let VectorCoreCount = 1;
}

def Ascend310B4Spec : Ascend310B_BaseSpec<"Ascend310B4"> {
  let AiCoreCount = 1;
  let CubeCoreCount = 1;
  let VectorCoreCount = 1;
}

//===----------------------------------------------------------------------===//
// Ascend950
//===----------------------------------------------------------------------===//

class Ascend950_BaseSpec <string name> : TargetSpec<name> {
  let UbSize = 2031616; // bits = 248KB, reserve 8KB for compiler
  let MinimalDCacheSize = 262144; // bits = 32KB
  let MaximumDCacheSize = 983040; // bits = 120KB
  let L1Size = 4194304; // bits = 512KB
  let L0aSize = 524288; // bits = 64KB
  let L0bSize = 524288; // bits = 64KB
  let L0cSize = 2097152; // bits = 256KB
  let UbAlignSize = 256; // bits = 32B
  let L1AlignSize = 256; // bits = 32B
  let L0cAlignSize = 4096; // bits = 512B
  let Arch = "dav-c310";
}

def Ascend910_950z_Spec : Ascend950_BaseSpec<"Ascend910_950z"> {
  let AiCoreCount = 4;
  let CubeCoreCount = 4;
  let VectorCoreCount = 8;
}

def Ascend910_9579_Spec : Ascend950_BaseSpec<"Ascend910_9579"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend910_957b_Spec : Ascend950_BaseSpec<"Ascend910_957b"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend910_957d_Spec : Ascend950_BaseSpec<"Ascend910_957d"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend910_9581_Spec : Ascend950_BaseSpec<"Ascend910_9581"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend910_9589_Spec : Ascend950_BaseSpec<"Ascend910_9589"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend910_958a_Spec : Ascend950_BaseSpec<"Ascend910_958a"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend910_958b_Spec : Ascend950_BaseSpec<"Ascend910_958b"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend910_9599_Spec : Ascend950_BaseSpec<"Ascend910_9599"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950PR_950z_Spec : Ascend950_BaseSpec<"Ascend950PR_950z"> {
  let AiCoreCount = 4;
  let CubeCoreCount = 4;
  let VectorCoreCount = 8;
}

def Ascend950PR_9579_Spec : Ascend950_BaseSpec<"Ascend950PR_9579"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950PR_957a_Spec : Ascend950_BaseSpec<"Ascend950PR_957a"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950PR_957b_Spec : Ascend950_BaseSpec<"Ascend950PR_957b"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950PR_957c_Spec : Ascend950_BaseSpec<"Ascend950PR_957c"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950PR_957d_Spec : Ascend950_BaseSpec<"Ascend950PR_957d"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950PR_9589_Spec : Ascend950_BaseSpec<"Ascend950PR_9589"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950PR_958a_Spec : Ascend950_BaseSpec<"Ascend950PR_958a"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950PR_958b_Spec : Ascend950_BaseSpec<"Ascend950PR_958b"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950PR_958c_Spec : Ascend950_BaseSpec<"Ascend950PR_958c"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950PR_958d_Spec : Ascend950_BaseSpec<"Ascend950PR_958d"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950PR_9599_Spec : Ascend950_BaseSpec<"Ascend950PR_9599"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950PR_959a_Spec : Ascend950_BaseSpec<"Ascend950PR_959a"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950PR_959b_Spec : Ascend950_BaseSpec<"Ascend950PR_959b"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_950x_Spec : Ascend950_BaseSpec<"Ascend950DT_950x"> {
  let AiCoreCount = 8;
  let CubeCoreCount = 8;
  let VectorCoreCount = 16;
}

def Ascend950DT_950y_Spec : Ascend950_BaseSpec<"Ascend950DT_950y"> {
  let AiCoreCount = 8;
  let CubeCoreCount = 8;
  let VectorCoreCount = 16;
}

def Ascend950DT_9571_Spec : Ascend950_BaseSpec<"Ascend950DT_9571"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9572_Spec : Ascend950_BaseSpec<"Ascend950DT_9572"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9573_Spec : Ascend950_BaseSpec<"Ascend950DT_9573"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9574_Spec : Ascend950_BaseSpec<"Ascend950DT_9574"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9575_Spec : Ascend950_BaseSpec<"Ascend950DT_9575"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9576_Spec : Ascend950_BaseSpec<"Ascend950DT_9576"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9577_Spec : Ascend950_BaseSpec<"Ascend950DT_9577"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9578_Spec : Ascend950_BaseSpec<"Ascend950DT_9578"> {
  let AiCoreCount = 28;
  let CubeCoreCount = 28;
  let VectorCoreCount = 56;
}

def Ascend950DT_9581_Spec : Ascend950_BaseSpec<"Ascend950DT_9581"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9582_Spec : Ascend950_BaseSpec<"Ascend950DT_9582"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9583_Spec : Ascend950_BaseSpec<"Ascend950DT_9583"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9584_Spec : Ascend950_BaseSpec<"Ascend950DT_9584"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9585_Spec : Ascend950_BaseSpec<"Ascend950DT_9585"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9586_Spec : Ascend950_BaseSpec<"Ascend950DT_9586"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9587_Spec : Ascend950_BaseSpec<"Ascend950DT_9587"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9588_Spec : Ascend950_BaseSpec<"Ascend950DT_9588"> {
  let AiCoreCount = 32;
  let CubeCoreCount = 32;
  let VectorCoreCount = 64;
}

def Ascend950DT_9591_Spec : Ascend950_BaseSpec<"Ascend950DT_9591"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_9592_Spec : Ascend950_BaseSpec<"Ascend950DT_9592"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_9595_Spec : Ascend950_BaseSpec<"Ascend950DT_9595"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_9596_Spec : Ascend950_BaseSpec<"Ascend950DT_9596"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_95A1_Spec : Ascend950_BaseSpec<"Ascend950DT_95A1"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

def Ascend950DT_95A2_Spec : Ascend950_BaseSpec<"Ascend950DT_95A2"> {
  let AiCoreCount = 36;
  let CubeCoreCount = 36;
  let VectorCoreCount = 72;
}

#endif // BISHENGIR_DIALECT_HACC_TARGETS_NPUTARGETSPEC_TD
