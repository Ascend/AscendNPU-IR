// REQUIRES: asserts
// RUN: bishengir-opt -mlir-disable-threading -hivm-delayed-cross-core-gss -debug-only=hivm-delayed-cross-core-gss -split-input-file -verify-diagnostics %s -o /dev/null 2>&1 | FileCheck %s
// RUN: bishengir-opt -mlir-disable-threading -pass-pipeline="builtin.module(hivm-delayed-cross-core-gss{solver-version=v1})" -split-input-file -verify-diagnostics %s -o /dev/null
// RUN: bishengir-opt -mlir-disable-threading -pass-pipeline="builtin.module(hivm-delayed-cross-core-gss{solver-version=v2})" -split-input-file -verify-diagnostics %s -o /dev/null

// Unit tests for the delayed cross-core graph sync solver (DCCGSS).
//
// Each case is a (backup, cube, vector) kernel triplet: the backup mix
// function carries the anchor skeleton, and the split cube/vector kernels
// carry the actual RW ops between matching anchors. The pass merges each
// anchor interval into synthetic RW ops on the mix side, solves once, and
// fans the sync decisions back out to all three kernels.
//
// Every case reproduces the full IR translator dump printed under
// -debug-only=hivm-delayed-cross-core-gss as a comment block ("before:" is
// the synthetic mix IR followed by both split sides, "after-mix/cube/vector:"
// are the IRs after solving and codegen). CHECK lines are interleaved where
// the output is verified; purely structural lines are plain comments. If the
// translated IR is correct, the solving and codegen built on top of it
// operate on the right hazards.

// -----
// Single-side intervals: a cube-only interval and a vector-only interval are
// merged into synthetic RW ops whose MemInfos carry the per-side core type
// and pipe of the source op.
module {
  func.func @kern_backup() attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern_mix_aic(%arg0: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern_mix_aiv(%arg0: memref<?xf32, #hivm.address_space<gm>>, %arg1: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>)
    hivm.hir.anchor {id = 2 : i64}
    return
  }
}
// Synthetic mix IR: one merged RW op per single-side interval.
// Split-side dumps keep the original ops; their MemInfos have no per-MemInfo
// core type or pipe (those are only attached during interval merging).
// No cross-core hazard (RAR on %arg0 only), so no sync ops are inserted.

// Expected IR translator dump:
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aic, 0))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aiv, 0))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: hivm.hir.load ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aiv, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aic, 0))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aiv, 0))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: hivm.hir.load ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern_mix_aiv, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }

// -----
// Both sides have RW ops in the same interval: the merge produces a single
// CUBE_AND_VECTOR RW op keeping each side's MemInfo core types, and logs the
// "unexpected" diagnostic for the ambiguous interval.
module {
  func.func @kern2_backup(%arg0: memref<?xi32>) attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    return
  }
  func.func @kern2_mix_aic(%arg0: memref<?xi32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<?xi32, #hivm.address_space<gm>>
    hivm.hir.anchor {id = 1 : i64}
    return
  }
  func.func @kern2_mix_aiv(%arg0: memref<?xi32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<?xi32, #hivm.address_space<gm>>
    hivm.hir.anchor {id = 1 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK: createRWOperation: unexpected for both cube and vector kernels to have rw ops between given anchors, check anchor-id=0
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE_AND_VECTOR>] [<PIPE_S>]
//           CHECK: read: MemInfo<CUBE, PIPE_S>(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aic, 0))
//           CHECK: read: MemInfo<VECTOR, PIPE_S>(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aiv, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<?xi32, #hivm.address_space<gm>> [<CUBE>] [<PIPE_S>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aic, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<?xi32, #hivm.address_space<gm>> [<VECTOR>] [<PIPE_S>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aiv, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE_AND_VECTOR>] [<PIPE_S>]
//           CHECK: read: MemInfo<CUBE, PIPE_S>(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aic, 0))
//           CHECK: read: MemInfo<VECTOR, PIPE_S>(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aiv, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : memref<?xi32, #hivm.address_space<gm>> [<CUBE>] [<PIPE_S>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aic, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: %0 = memref.load %arg0[%c0] {hivm.tcore_type = #hivm.tcore_type<VECTOR>} : memref<?xi32, #hivm.address_space<gm>> [<VECTOR>] [<PIPE_S>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xi32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern2_mix_aiv, 0))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }

// -----
// Cross-core hazard: the cube side writes a GM buffer through
// nd2nz/mmadL1/fixpipe and the vector side reads the same buffer, so the
// solver inserts a CUBE set / VECTOR wait pair that is fanned out to the
// mix, cube, and vector kernels.
module {
  func.func @kern3_backup() attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern3_mix_aic(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<256xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%0, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>)
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern3_mix_aiv(%arg0: memref<16xf32, #hivm.address_space<gm>>, %arg1: memref<256xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>)
    hivm.hir.anchor {id = 2 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<16xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern3_mix_aic, 0))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE1>, <PIPE_M>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE1>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_M>(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo<CUBE, PIPE_FIX>(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_FIX>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aic, 1))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<16xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern3_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.mmadL1 ins(%0, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>) [<CUBE>] [<PIPE_MTE1>, <PIPE_M>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>) [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aic, 1))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: hivm.hir.load ins(%arg1 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.store ins(%0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<16xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern3_mix_aic, 0))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<CUBE>] [<PIPE_MTE1>, <PIPE_M>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE1>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_M>(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo<CUBE, PIPE_FIX>(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_FIX>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aic, 1))
//         CHECK: SetFlagOp [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: Anchor (anchor-id=1)
//         CHECK: WaitFlagOp [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<16xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern3_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.mmadL1 ins(%0, %0, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>, memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%1 : memref<256xf32, #hivm.address_space<cc>>) [<CUBE>] [<PIPE_MTE1>, <PIPE_M>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%1 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>) [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo(%1 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aic, 1))
//         CHECK: SetFlagOp [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: WaitFlagOp [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: hivm.hir.load ins(%arg1 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: hivm.hir.store ins(%0 : memref<256xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern3_mix_aiv, 1))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }

// -----
// Loops and multibuffering: the vector side runs inside an scf.for with a
// double-buffered UB pointer cast; the synthetic mix IR must reproduce the
// loop/scope nesting and the multibuffer address list.
module {
  func.func @kern4_backup(%arg0: index) attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    scf.for %i = %arg0 to %arg0 step %arg0 {
      hivm.hir.anchor {id = 1 : i64}
      hivm.hir.anchor {id = 2 : i64}
    }
    hivm.hir.anchor {id = 3 : i64}
    return
  }
  func.func @kern4_mix_aic(%arg0: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    scf.for %i = %arg0 to %arg0 step %arg0 {
      hivm.hir.anchor {id = 1 : i64}
      hivm.hir.anchor {id = 2 : i64}
    }
    hivm.hir.anchor {id = 3 : i64}
    return
  }
  func.func @kern4_mix_aiv(%arg0: index, %arg1: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    %c2048_i64 = arith.constant 2048 : i64
    scf.for %i = %arg0 to %arg0 step %arg0 {
      hivm.hir.anchor {id = 1 : i64}
      %0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>)
      hivm.hir.anchor {id = 2 : i64}
    }
    hivm.hir.anchor {id = 3 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//               CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//               CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//               CHECK: write: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE3>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE3>(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE3>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         PlaceHolder
//         CHECK: Loop static-loop-count=0 {
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             CHECK: hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//               CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: hivm.hir.store ins(%0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<?xf32, #hivm.address_space<gm>>) [<VECTOR>] [<PIPE_MTE3>]
//               CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0, 16384], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//               CHECK: write: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern4_mix_aiv, 1))
//             CHECK: Anchor (anchor-id=2)
//             PlaceHolder
//           }
//         }
//         PlaceHolder
//         CHECK: Anchor (anchor-id=3)
//         PlaceHolder
//       }
//     }
//   }

// -----
// Both sides in one interval with disagreeing pipes: the merged RW op falls
// back to PIPE_UNASSIGNED while each MemInfo keeps its own core type and
// pipe.
module {
  func.func @kern5_backup() attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    return
  }
  func.func @kern5_mix_aic(%arg0: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    hivm.hir.anchor {id = 1 : i64}
    return
  }
  func.func @kern5_mix_aiv(%arg0: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%0, %1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%1 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.anchor {id = 1 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK: createRWOperation: unexpected for both cube and vector kernels to have rw ops between given anchors, check anchor-id=0
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE_AND_VECTOR>] [<PIPE_UNASSIGNED>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern5_mix_aic, 0))
//           CHECK: read: MemInfo<VECTOR, PIPE_V>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: read: MemInfo<VECTOR, PIPE_V>(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_V>(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern5_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.vadd ins(%0, %1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%1 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_V>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: read: MemInfo(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE_AND_VECTOR>] [<PIPE_UNASSIGNED>]
//           CHECK: read: MemInfo<CUBE, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern5_mix_aic, 0))
//           CHECK: read: MemInfo<VECTOR, PIPE_V>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: read: MemInfo<VECTOR, PIPE_V>(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<VECTOR, PIPE_V>(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>) [<CUBE>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern5_mix_aic, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>, PointerLikeInfo(cbuf, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.vadd ins(%0, %1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%1 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_V>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: read: MemInfo(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(%1 = hivm.hir.pointer_cast(%c4096_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [32768], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=1)
//         PlaceHolder
//       }
//     }
//   }

// -----
// Control flow: an scf.if on the vector side becomes a Condition with a
// trueScope in the synthetic mix IR, holding the merged RW op.
module {
  func.func @kern6_backup(%arg0: i1) attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    scf.if %arg0 {
      hivm.hir.anchor {id = 1 : i64}
    }
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern6_mix_aic(%arg0: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    scf.if %arg0 {
      hivm.hir.anchor {id = 1 : i64}
    }
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern6_mix_aiv(%arg0: i1, %arg1: memref<?xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c0_i64 = arith.constant 0 : i64
    scf.if %arg0 {
      %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.anchor {id = 1 : i64}
    }
    hivm.hir.anchor {id = 2 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern6_mix_aiv, 1))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern6_mix_aiv, 1))
//               CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern6_mix_aiv, 1))
//               CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Condition {
//           CHECK: (trueScope)
//           Scope {
//             PlaceHolder
//             CHECK: hivm.hir.load ins(%arg1 : memref<?xf32, #hivm.address_space<gm>>) outs(%0 : memref<16xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//               CHECK: read: MemInfo(<block argument> of type 'memref<?xf32, #hivm.address_space<gm>>' at index: 1, FuncArgInfo(kern6_mix_aiv, 1))
//               CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 512, isWorkSpace=false, isTightlyCoupledBuffer=false))
//             CHECK: Anchor (anchor-id=1)
//             PlaceHolder
//           }
//         }
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }

// -----
// A backup function without its split cube/vector kernels is skipped with a
// warning instead of failing the pipeline.
module {
  // expected-warning@+1 {{delayed-cross-core-gss: split kernels not found.}}
  func.func @kern7_backup() attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    return
  }
}

// -----
// Pre-existing intra-block sync ops are erased before solving; the solver
// re-derives the sync from the actual hazards (flag 3 is dropped, a fresh
// EVENT_ID0 pair is inserted).
module {
  func.func @kern8_backup() attributes {hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 3
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 3
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern8_mix_aic(%arg0: memref<256xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    %c64_i64 = arith.constant 64 : i64
    %0 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.fixpipe {enable_nz2nd} ins(%0 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg0 : memref<256xf32, #hivm.address_space<gm>>)
    hivm.hir.anchor {id = 1 : i64}
    hivm.hir.anchor {id = 2 : i64}
    return
  }
  func.func @kern8_mix_aiv(%arg0: memref<256xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.anchor {id = 1 : i64}
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>)
    hivm.hir.anchor {id = 2 : i64}
    return
  }
}

// Expected IR translator dump:
// CHECK-LABEL: before:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo<CUBE, PIPE_FIX>(%0 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_FIX>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aic, 0))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aiv, 0))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg0 : memref<256xf32, #hivm.address_space<gm>>) [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aic, 0))
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: hivm.hir.load ins(%arg0 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aiv, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-mix:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: RWOperation [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo<CUBE, PIPE_FIX>(%0 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo<CUBE, PIPE_FIX>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aic, 0))
//         CHECK: SetFlagOp [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: Anchor (anchor-id=1)
//         CHECK: WaitFlagOp [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: RWOperation [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo<VECTOR, PIPE_MTE2>(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aiv, 0))
//           CHECK: write: MemInfo<VECTOR, PIPE_MTE2>(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-cube:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : memref<256xf32, #hivm.address_space<cc>>) outs(%arg0 : memref<256xf32, #hivm.address_space<gm>>) [<CUBE>] [<PIPE_FIX>]
//           CHECK: read: MemInfo(%0 = hivm.hir.pointer_cast(%c64_i64) : memref<256xf32, #hivm.address_space<cc>>, PointerLikeInfo(cc, null, [512], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//           CHECK: write: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aic, 0))
//         CHECK: SetFlagOp [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: Anchor (anchor-id=1)
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
// CHECK-LABEL: after-vector:
//   Function {
//     Scope {
//       FunctionBlock {
//         PlaceHolder
//         CHECK: Anchor (anchor-id=0)
//         CHECK: Anchor (anchor-id=1)
//         CHECK: WaitFlagOp [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID0)]
//         CHECK: hivm.hir.load ins(%arg0 : memref<256xf32, #hivm.address_space<gm>>) outs(%0 : memref<256xf32, #hivm.address_space<ub>>) [<VECTOR>] [<PIPE_MTE2>]
//           CHECK: read: MemInfo(<block argument> of type 'memref<256xf32, #hivm.address_space<gm>>' at index: 0, FuncArgInfo(kern8_mix_aiv, 0))
//           CHECK: write: MemInfo(%0 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>, PointerLikeInfo(ub, null, [0], 8192, isWorkSpace=false, isTightlyCoupledBuffer=false))
//         CHECK: Anchor (anchor-id=2)
//         PlaceHolder
//       }
//     }
//   }
