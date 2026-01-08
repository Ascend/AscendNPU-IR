//===- HIVMSynchronizationOps.td - HIVM dialect Sync. Ops. -*- tablegen -*-===//
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
//
// This is the definition file for HIVM dialect synchronization operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// HIVM pipe synchronization operations.
//===----------------------------------------------------------------------===//

class HIVM_SynchronizationOp<string mnemonic, list<Trait> traits = []> :
  HIVM_Op<mnemonic, !listconcat(
    [DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface,
      ["inferCoreType"]>
    ], traits)> {
}

def SetFlagOp : HIVM_SynchronizationOp<"set_flag"> {
  let summary = "hivm set flag.";
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id
  );
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
  let hasVerifier = 1;
}

def WaitFlagOp : HIVM_SynchronizationOp<"wait_flag"> {
  let summary = "hivm wait flag.";
  let arguments = (ins HIVM_PipeAttr:$set_pipe,
                       HIVM_PipeAttr:$wait_pipe,
                       OptionalAttr<HIVM_EventAttr>:$static_event_id,
                       Optional<I64>:$dynamic_event_id
  );
  let assemblyFormat = [{
    `[`
    $set_pipe
    `,` $wait_pipe
    `,` custom<EventID>($static_event_id, $dynamic_event_id)
    `]` attr-dict
  }];
  let hasVerifier = 1;
}

def PipeBarrierOp : HIVM_SynchronizationOp<"pipe_barrier"> {
  let summary = "hivm pipe barrier.";
  let arguments = (ins HIVM_PipeAttr:$pipe);
  let assemblyFormat = "`[` $pipe `]` attr-dict";
}

//===----------------------------------------------------------------------===//
// HIVM block synchronization operations.
//===----------------------------------------------------------------------===//

def SyncBlockOp : HIVM_SynchronizationOp<"sync_block"> {
  let summary = "hivm sync block between different kernels.";
  let description = [{
    There are three sync block modes:
      - ALL_CUBE : All cube are synchronized to a same point.
                   `tcube_pipe` needs to be set to the pipe that
                   the cube core is waiting for.
      - ALL_VECTOR : All vector are synchronized to a same point.
                   `tvector_pipe` needs to be set to the pipe that the
                   vector core is waiting for.
      - BARRIER_CUBE : Used for cube-cube synchronization, it's going to
                   be lowered to a barrie.pipe_all and would only be
                   copied the aic kernel.
      - BARRIER_VECTOR : Used for cube-cube synchronization, it's going to
                   be lowered to a barrie.pipe_all and  would only be
                   copied the aiv kernel.
      - ALL : All aic/aiv are synchronized to same point.
              `tvector_pipe` needs to be set to the pipe that the
              vector core is waiting for.

    Note:
      - SyncBlockOp can only use after data is moved to gm.
      - `$ffts_base_addr` must be set in Ascend910B. Every time FFTS collect
        one specific `$flag_id` from all subblocks, FFTS would set the flag ID
        back to the block in the group to do synchronization.
  }];
  let arguments = (ins HIVM_SyncBlockModeAttr:$sync_block_mode,
                       OptionalAttr<Builtin_IntegerAttr>:$flag_id,
                       Optional<I64>:$ffts_base_addr,
                       OptionalAttr<HIVM_PipeAttr>:$tcube_pipe,
                       OptionalAttr<HIVM_PipeAttr>:$tvector_pipe
  );
  let assemblyFormat = [{
    attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
    (`ffts_base_addr` `=` $ffts_base_addr^)?
    (`tcube_pipe` `=` $tcube_pipe^)?
    (`tvector_pipe` `=` $tvector_pipe^)?
  }];
  let hasVerifier = 1;
}

def SyncBlockSetOp : HIVM_SynchronizationOp<"sync_block_set", [AttrSizedOperandSegments]> {
  let summary = "hivm set block sync.";
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id,
                       Optional<I64>:$ffts_base_addr,
                       DefaultValuedOptionalAttr<HIVM_SyncBlockInstrModeAttr,
                         "INTRA_BLOCK_SYNCHRONIZATION">:$tsync_instr_mode
  );
  let assemblyFormat = [{
    attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
    `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
    (`ffts_base_addr` `=` $ffts_base_addr^)?
    (`syn_instr_mode` `=` $tsync_instr_mode^)?
  }];
  let builders = [
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id)>,
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id, "Value":$ffts_base_addr,
                   "SyncBlockInstrModeAttr":$tsync_instr_mode)>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult getFlagId();
  }];
  let hasVerifier = 1;
}

def SyncBlockWaitOp : HIVM_SynchronizationOp<"sync_block_wait"> {
  let summary = "hivm wait block sync.";
  let arguments = (ins HIVM_TCoreTypeAttr:$tcore_type,
                       HIVM_PipeAttr:$tpipe,
                       HIVM_PipeAttr:$pipe,
                       OptionalAttr<Builtin_IntegerAttr>:$static_flag_id,
                       Optional<I64>:$dynamic_flag_id
  );
  let assemblyFormat = [{
    attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
    `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
  }];
  let builders = [
    OpBuilder<(ins "TCoreTypeAttr":$tcore_type, "PipeAttr":$tpipe, "PipeAttr":$pipe,
                   "OpFoldResult":$flag_id)>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult getFlagId();
  }];
  let hasVerifier = 1;
}

def CreateSyncBlockLockOp
    : HIVM_Op<"create_sync_block_lock"> {
  let summary = "create sync block lock operation.";
  let description = [{
    The `create_sync_block_lock` operation allocates a region of lock memory,
    which is used to make the code between lock and unlock execute in order among blocks.
    Example:

    ```mlir
      hivm.hir.create_sync_block_lock() : memref<1xi64>
      hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
    ```
  }];
  let arguments = (ins Optional<AnyMemRef>:$lockArg);
  let results = (outs Res<AnyMemRef, "",
                        [MemAlloc<DefaultResource, 0, FullEffect>]>:$memref);

  let assemblyFormat = [{
    (`from` $lockArg^)?
        attr-dict `:` (`from` type($lockArg)^ `to`)? type($memref)
  }];

  let hasVerifier = 1;
}

def SyncBlockLockOp
  : HIVM_Op<"sync_block_lock"> {
  let summary = "sync block lock operation.";
  let description = [{
    The sync_block_lock operation will not release until the lock_var equals the block idx.
    Example:

    ```mlir
      hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
    ```
  }];
  let arguments = (ins MemRefRankOf<[I64], [1]>:$lock_var);

  let assemblyFormat = [{
    attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "sync_block_lock"; }
  }];
}

def SyncBlockUnlockOp
  : HIVM_Op<"sync_block_unlock"> {
  let summary = "sync block unlock operation.";
  let description = [{
    The `sync_block_lock` operation will increase and release the lock_var.
    Example:

    ```mlir
      hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
    ```
  }];
  let arguments = (ins MemRefRankOf<[I64], [1]>:$lock_var);

  let assemblyFormat = [{
    attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "sync_block_unlock"; }
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMSYNCHRONIZATIONOPS_TD
