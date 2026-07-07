//===- HIVMIntrinOps.td - HIVM dialect intrinsic operation -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect intrinsic operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTRINOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTRINOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// HIVM Intrinsic Operations.
//===----------------------------------------------------------------------===//

class HIVM_IntrinOp<string mnemonic, list<int> overloadOperands = [],
                    list<Trait> traits = [], int numResults = 1,
                    list<int> immArgPositions = [],
                    list<string> immArgAttrNames = []>
    : LLVM_IntrOpBase<
          /*Dialect dialect=*/HIVM_Dialect,
          /*string opName=*/"intr."#mnemonic,
          /*string enumName=*/"HIVMIntrinsics::"#!subst(".", "_", mnemonic),
          /*list<int> overloadedResults=*/[],
          /*list<int> overloadedOperands=*/overloadOperands,
          /*list<Trait> traits=*/traits,
          /*int numResults=*/numResults,
          /*requiresAccessGroup=*/0,
          /*requiresAliasAnalysis=*/0,
          /*requiresFastmath=*/0, immArgPositions, immArgAttrNames>;

def GetBlockIdxInstrOp : HIVM_IntrinOp<"hivm.GET.BLOCK.IDX">;

def GetBlockNumInstrOp : HIVM_IntrinOp<"hivm.GET.BLOCK.NUM">;

def GetSubBlockIdxInstrOp : HIVM_IntrinOp<"hivm.GET.SUBBLOCKID">;

def GetSubBlockNumInstrOp : HIVM_IntrinOp<"hivm.GET.SUBBLOCKDIM">;

class SetWaitFlag_InstrOp<string mnemonic, int numResults = 0>
    : HIVM_IntrinOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults,
          /*immArgPositions=*/[0, 1],
          /*immArgAttrNames=*/["set_pipe", "wait_pipe"]> {
  let arguments = (ins I64Attr:$set_pipe, I64Attr:$wait_pipe, I<64>:$event_id);
}

class SetWaitFlagImm_InstrOp<string mnemonic, int numResults = 0>
    : HIVM_IntrinOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults,
          /*immArgPositions=*/[0, 1, 2],
          /*immArgAttrNames=*/["set_pipe", "wait_pipe", "event_id"]> {
  let arguments = (ins I64Attr:$set_pipe, I64Attr:$wait_pipe, I64Attr:$event_id);
}

def PipeBarrierInstrOp : HIVM_IntrinOp<
          /*string opName=*/"hivm.BARRIER",
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/0,
          /*immArgPositions=*/[0],
          /*immArgAttrNames=*/["pipe"]> {
    let arguments = (ins I64Attr:$pipe);
}

def SetFlagImmInstrOp : SetWaitFlagImm_InstrOp<"hivm.SET.FLAG.IMM">;
def WaitFlagImmInstrOp : SetWaitFlagImm_InstrOp<"hivm.WAIT.FLAG.IMM">;
def SetFlagRegInstrOp : SetWaitFlag_InstrOp<"hivm.SET.FLAG.REG">;
def WaitFlagRegInstrOp : SetWaitFlag_InstrOp<"hivm.WAIT.FLAG.REG">;

def SetFftsBaseAddrInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.SET.FFTS.BASE.ADDR",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0> {
  let summary = "Set SPR.FFTS_BASE_ADDR.";
  let arguments = (ins I<64>:$addr);
}

def SetCrossCoreInstrOp : HIVM_IntrinOp<
          /*string opName=*/"hivm.SET.CROSS.CORE",
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/0,
          /*immArgPositions=*/[0],
          /*immArgAttrNames=*/["pipe"]> {
  let summary = "FFTS cross-core synchronization instruction";
  let description = [{
    This instruction send data (including mode and flag ID) to the destination
    address of FFTS indicated by SPR.FFTS_BASE_ADDR.
    - This op takes 2 inputs:
       - '$pipe' : pipe type.
       - '$config' : including mode and flagID.
          '$config' = (0x0001 | ((mode & 0x0f) << 4) | ((flagID & 0x0f) << 8))
       - flagID : Each flag ID has a counter. If the flag ID has been set by FFTS,
        the counter will be incremented. 8 bits.
       - mode : There are three modes. 2 bits.
          - mode 0: inter block synchronization.
          - mode 1: inter subblock synchronization.
          - mode 2: intra block synchronization.
  }];
  let arguments = (ins I64Attr:$pipe,
                       I<64>:$config);
}


def WaitFlagDevInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.WAIT.FLAG.DEV.REG",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0> {
  let summary = "FFTS block/subblock synchronization instructions.";
  let description = [{
    - This op takes 1 inputs:
      '$flag_id' : Each flag ID has a counter. If the flag ID has been set by
        FFTS, the counter will be incremented. If the WAIT_FLAG_DEV of the same
        flag ID has been executed, the counter will be decremented. If the counter
        of flag ID is zero, all the instructions after this instruction will be
        blocked. It will wait until the counter is not zero.
  }];
  let arguments = (ins I<64>:$flag_id);
}

def WaitFlagDevPipeImmInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.WAIT.FLAG.DEV.PIPE.IMM",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0, 1],
                              /*immArgAttrNames=*/["pipe", "flag_id"]> {
  let summary = "FFTS block/subblock synchronization instructions.";
  let description = [{
    - This op takes 1 inputs:
      '$flag_id' : Each flag ID has a counter. If the flag ID has been set by
        FFTS, the counter will be incremented. If the WAIT_FLAG_DEV of the same
        flag ID has been executed, the counter will be decremented. If the counter
        of flag ID is zero, all the instructions after this instruction will be
        blocked. It will wait until the counter is not zero.
  }];
  let arguments = (ins I64Attr:$pipe, I64Attr:$flag_id);
}

def WaitFlagDevPipeRegInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.WAIT.FLAG.DEV.PIPE.REG",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0],
                              /*immArgAttrNames=*/["pipe"]> {
  let summary = "FFTS block/subblock synchronization instructions.";
  let description = [{
    - This op takes 1 inputs:
      '$flag_id' : Each flag ID has a counter. If the flag ID has been set by
        FFTS, the counter will be incremented. If the WAIT_FLAG_DEV of the same
        flag ID has been executed, the counter will be decremented. If the counter
        of flag ID is zero, all the instructions after this instruction will be
        blocked. It will wait until the counter is not zero.
  }];
  let arguments = (ins I64Attr:$pipe, I<64>:$flag_id);
}

def SetIntraBlockRegInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.SET.INTRA.BLOCK.mode",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0],
                              /*immArgAttrNames=*/["pipe"]> {
  let summary = "";
  let description = [{
    - This op takes 1 input: sync id, the sync id will be stored into one register.
      For a mix-mode block, it will be launched to 1 CUBECORE and 2 VECCORE. CUBECORE has 32 ID and
      each VECCORE has 16 ID, the ID mapping rule among 3 cores is shown below:
      1. ID 0-15 of SET operation launched by VECCORE0 will be mapped to ID 0-15 in CUBECORE
      2. ID 0-15 of SET operation launched by VECCORE1 will be mapped to ID 16-31 in CUBECORE
      3. ID 0-15 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE0
      4. ID 16-31 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE1
      and each ID there is a 4-bit counter. Counter of each sync ID of certain core can either be incremented by
      SET operation from the other core, or be decremented by WAIT operation of its own.
      For SET operation, it will increment the corresponding counter of the other Core by 1, and won’t block
      further instructions in the same pipeline.
      For WAIT operation, it will stall pipeline if counter of corresponding ID is zero, until this ID is set by the
      other core, otherwise if the counter is non-zero, it will decrement the ID’s counter by 1.
  }];
  let arguments = (ins I64Attr:$pipe,
                       I<64>:$sync_id);
}

def WaitIntraBlockRegInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.WAIT.INTRA.BLOCK.mode",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0],
                              /*immArgAttrNames=*/["pipe"]>{
  let summary = "";
  let description = [{
    - This op takes 1 input: sync id, the sync id will be stored in one register.
      For a mix-mode block, it will be launched to 1 CUBECORE and 2 VECCORE. CUBECORE has 32 ID and
      each VECCORE has 16 ID, the ID mapping rule among 3 cores is shown below:
      1. ID 0-15 of SET operation launched by VECCORE0 will be mapped to ID 0-15 in CUBECORE
      2. ID 0-15 of SET operation launched by VECCORE1 will be mapped to ID 16-31 in CUBECORE
      3. ID 0-15 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE0
      4. ID 16-31 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE1
      and each ID there is a 4-bit counter. Counter of each sync ID of certain core can either be incremented by
      SET operation from the other core, or be decremented by WAIT operation of its own.
      For SET operation, it will increment the corresponding counter of the other Core by 1, and won’t block
      further instructions in the same pipeline.
      For WAIT operation, it will stall pipeline if counter of corresponding ID is zero, until this ID is set by the
      other core, otherwise if the counter is non-zero, it will decrement the ID’s counter by 1.
  }];
  let arguments = (ins I64Attr:$pipe,
                       I<64>:$sync_id);
}

def SetIntraBlockImmInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.SET.INTRA.BLOCKI.mode",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0, 1],
                              /*immArgAttrNames=*/["pipe", "sync_id"]>{
  let summary = "";
  let description = [{
    - This op takes 1 input: sync id, the sync id is encoded in the instruction.
      For a mix-mode block, it will be launched to 1 CUBECORE and 2 VECCORE. CUBECORE has 32 ID and
      each VECCORE has 16 ID, the ID mapping rule among 3 cores is shown below:
      1. ID 0-15 of SET operation launched by VECCORE0 will be mapped to ID 0-15 in CUBECORE
      2. ID 0-15 of SET operation launched by VECCORE1 will be mapped to ID 16-31 in CUBECORE
      3. ID 0-15 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE0
      4. ID 16-31 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE1
      and each ID there is a 4-bit counter. Counter of each sync ID of certain core can either be incremented by
      SET operation from the other core, or be decremented by WAIT operation of its own.
      For SET operation, it will increment the corresponding counter of the other Core by 1, and won’t block
      further instructions in the same pipeline.
      For WAIT operation, it will stall pipeline if counter of corresponding ID is zero, until this ID is set by the
      other core, otherwise if the counter is non-zero, it will decrement the ID’s counter by 1.
  }];
  let arguments = (ins I64Attr:$pipe,
                       I64Attr:$sync_id);
}

def WaitIntraBlockImmInstrOp : HIVM_IntrinOp<
                              /*string opName=*/"hivm.WAIT.INTRA.BLOCKI.mode",
                              /*list<int> overloadedOperands=*/[],
                              /*list<Trait> traits=*/[],
                              /*int numResults=*/0,
                              /*immArgPositions=*/[0, 1],
                              /*immArgAttrNames=*/["pipe", "sync_id"]> {
  let summary = "";
  let description = [{
    - This op takes 1 input: sync id, the sync id is encoded in the instruction.
      For a mix-mode block, it will be launched to 1 CUBECORE and 2 VECCORE. CUBECORE has 32 ID and
      each VECCORE has 16 ID, the ID mapping rule among 3 cores is shown below:
      1. ID 0-15 of SET operation launched by VECCORE0 will be mapped to ID 0-15 in CUBECORE
      2. ID 0-15 of SET operation launched by VECCORE1 will be mapped to ID 16-31 in CUBECORE
      3. ID 0-15 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE0
      4. ID 16-31 of SET operation launched by CUBECORE will be mapped to ID 0-15 in VECCORE1
      and each ID there is a 4-bit counter. Counter of each sync ID of certain core can either be incremented by
      SET operation from the other core, or be decremented by WAIT operation of its own.
      For SET operation, it will increment the corresponding counter of the other Core by 1, and won’t block
      further instructions in the same pipeline.
      For WAIT operation, it will stall pipeline if counter of corresponding ID is zero, until this ID is set by the
      other core, otherwise if the counter is non-zero, it will decrement the ID’s counter by 1.
  }];
  let arguments = (ins I64Attr:$pipe,
                       I64Attr:$sync_id);
}

def SetMaskNormInstrOp : HIVM_IntrinOp<
                             /*string opName=*/"hivm.SET.MASK.NORM",
                             /*list<int> overloadedOperands=*/[],
                             /*list<Trait> traits=*/[],
                             /*int numResults=*/0>;

def GetCtrlInstrOp : HIVM_IntrinOp<"hivm.GET.CTRL">;

def SetCtrlInstrOp : HIVM_IntrinOp</*string opName=*/"hivm.SET.CTRL",
                                   /*list<int> overloadedOperands=*/[],
                                   /*list<Trait> traits=*/[],
                                   /*int numResults=*/0> {
  let summary = "set ctrl";
  let arguments = (ins I<64>:$config);
}

class SBitSet_InstrOp<string mnemonic, int numResults = 1>
    : HIVM_IntrinOp<
          /*string opName=*/mnemonic,
          /*list<int> overloadedOperands=*/[],
          /*list<Trait> traits=*/[],
          /*int numResults=*/numResults> {
  let arguments = (ins I<64>:$x, I<64>:$idx);
}

def SBitSet0InstrOp : SBitSet_InstrOp<"hivm.SBITSET0">;
def SBitSet1InstrOp : SBitSet_InstrOp<"hivm.SBITSET1">;

def DCCIDstInstrOp : HIVM_IntrinOp</*string opName=*/"hivm.DCCI.DST",
                                   /*list<int> overloadedOperands=*/[],
                                   /*list<Trait> traits=*/[],
                                   /*int numResults=*/0> {
  let summary = "dcci.dst.gm instr op";
  let arguments = (ins Arg<LLVM_AnyPointer,"address">:$ptr, 
                       I<64>:$entire, 
                       I<64>:$dst);
}

def DCCIDstUBInstrOp : HIVM_IntrinOp</*string opName=*/"hivm.DCCI.DST.UB",
                                   /*list<int> overloadedOperands=*/[],
                                   /*list<Trait> traits=*/[],
                                   /*int numResults=*/0> {
  let summary = "dcci.dst.ub instr op";
  let arguments = (ins Arg<LLVM_AnyPointer,"address">:$ptr, 
                       I<64>:$entire, 
                       I<64>:$dst);
}
#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTRINOPS_TD
