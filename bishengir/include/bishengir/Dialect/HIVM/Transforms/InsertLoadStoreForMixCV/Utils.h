//===------ Utils.h ---- utility of Insert Load Store for Mix CV ----------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_UTILS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_UTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace hivm {

constexpr llvm::StringLiteral kPropagateUpAttr = "propagate_up";
constexpr llvm::StringLiteral kPropagateDownAttr = "propagate_down";

namespace PropagatorUtil {

const llvm::SmallDenseMap<hivm::AddressSpace, TCoreType, 2> kAddressSpace2CoreType = {
  {hivm::AddressSpace::UB, TCoreType::VECTOR},
  {hivm::AddressSpace::L1, TCoreType::CUBE},
  {hivm::AddressSpace::GM, TCoreType::CUBE_OR_VECTOR},
  {hivm::AddressSpace::L0C, TCoreType::CUBE_OR_VECTOR},
};

//===----------------------------------------------------------------------===//
// Propagation marker model
//===----------------------------------------------------------------------===//
//
// Propagation uses `builtin.unrealized_conversion_cast` markers tagged with
// `propagate_up` or `propagate_down`, plus `hivm.tcore_type` and
// `hivm.address_space`.
//
//   propagate_up on operand O     requirement flows to O's producer
//   propagate_down on result R    requirement flows to R's users
//
// InsertPropagation seeds markers on ops. PropagateUp/PropagateDown patterns
// then walk the IR and mirror markers across region boundaries and structured
// op boundaries until ResolvePropagation can insert loads/stores.
//
// Structured ops (DMA, CustomOp, CustomMacroOp) share one boundary rule:
//
//   Direction reached op via          Action on the structured op
//   ------------------------          ---------------------------
//   propagate_down -> operand         input/temp: up on matched operand only
//                                     init:       up on matched init +
//                                                 down on all results
//   propagate_up   -> result          down on all results +
//                                     up on all init operands
//
// Custom-like ops additionally map pipe(s) to scope at insert time:
//   pipes[0]  -> inputs and temp_buffers (up)
//   pipes[n-1] or pipes[0] when n==1 -> inits (up) and results (down)
//
//===----------------------------------------------------------------------===//

/// Map a pipe to the core type and address space used for scope propagation.
std::pair<TCoreType, hivm::AddressSpace>
getPropagationInfoForPipe(hivm::PIPE pipe);

/// Resolved in/out scopes derived from `pipes` (1 pipe: same in/out; 2 pipes:
/// [inPipe, outPipe]).
struct CustomOpPipePropagationInfo {
  TCoreType inCoreType;
  hivm::AddressSpace inAddressSpace;
  TCoreType outCoreType;
  hivm::AddressSpace outAddressSpace;
};

CustomOpPipePropagationInfo
getCustomOpPropagationInfo(llvm::ArrayRef<hivm::PIPE> pipes);

/// Seed propagation markers on CustomOp / CustomMacroOp from their pipe(s).
void insertPropagatorsForCustomLikeOp(llvm::ArrayRef<hivm::PIPE> pipes,
                                      Operation *op, PatternRewriter &rewriter);

/// Mirror a down-propagator that reached `op` through operand `use`.
LogicalResult
propagateDownForCustomLikeOp(Operation *op, OpOperand *use,
                             UnrealizedConversionCastOp propagateOp,
                             PatternRewriter &rewriter);

/// Mirror an up-propagator that reached `op` through one of its results.
LogicalResult propagateUpForCustomLikeOp(Operation *op,
                                         UnrealizedConversionCastOp propagateOp,
                                         PatternRewriter &rewriter);

/// Read core-type/address-space metadata carried by a propagation cast.
std::pair<TCoreType, SmallVector<hivm::AddressSpace, 2>>
extractPropagatorInfo(UnrealizedConversionCastOp propagateOp);

/// Create a propagation cast tagged with direction/core/address metadata.
/// Returns the original value for non-shaped types.
Value createPropagator(Value v, StringRef directionAttrName, TCoreType coreType,
                       ArrayRef<hivm::AddressSpace> addressSpaces,
                       OpBuilder &builder);

Value createPropagator(Value v, StringRef directionAttrName, TCoreType coreType,
                       OpBuilder &builder);

Value createPropagator(Value v, StringRef directionAttrName,
                       ArrayRef<hivm::AddressSpace> addressSpaces,
                       OpBuilder &builder);

Value createPropagator(Value v, StringRef directionAttrName,
                       UnrealizedConversionCastOp propagateOp,
                       OpBuilder &builder);

/// Tag `operand` with propagate_up and core/address metadata.
void createPropagatorUp(OpOperand *operand, TCoreType coreType,
                        ArrayRef<hivm::AddressSpace> addressSpaces,
                        PatternRewriter &rewriter);

void createPropagatorUp(OpOperand *operand, TCoreType coreType,
                        PatternRewriter &rewriter);

void createPropagatorUp(OpOperand *operand,
                        ArrayRef<hivm::AddressSpace> addressSpaces,
                        PatternRewriter &rewriter);

void createPropagatorUp(OpOperand *operand,
                        UnrealizedConversionCastOp propagateOp,
                        PatternRewriter &rewriter);

/// Tag `res` (or its sole user) with propagate_down and core/address metadata.
void createPropagatorDown(Value res, TCoreType coreType,
                          ArrayRef<hivm::AddressSpace> addressSpaces,
                          PatternRewriter &rewriter);

void createPropagatorDown(Value res, TCoreType coreType,
                          PatternRewriter &rewriter);

void createPropagatorDown(Value res, ArrayRef<hivm::AddressSpace> addressSpaces,
                          PatternRewriter &rewriter);

void createPropagatorDown(Value res, UnrealizedConversionCastOp propagateOp,
                          PatternRewriter &rewriter);

void createPropagatorsUp(Operation *op, TCoreType coreType,
                         ArrayRef<hivm::AddressSpace> addressSpaces,
                         PatternRewriter &rewriter);

void createPropagatorsUp(Operation *op, TCoreType coreType,
                         PatternRewriter &rewriter);

void createPropagatorsUp(Operation *op,
                         ArrayRef<hivm::AddressSpace> addressSpaces,
                         PatternRewriter &rewriter);

void createPropagatorsUp(Operation *op, UnrealizedConversionCastOp propagateOp,
                         PatternRewriter &rewriter);

void createPropagatorsDown(Operation *op, TCoreType coreType,
                           ArrayRef<hivm::AddressSpace> addressSpaces,
                           PatternRewriter &rewriter);

void createPropagatorsDown(Operation *op, TCoreType coreType,
                           PatternRewriter &rewriter);

void createPropagatorsDown(Operation *op,
                           ArrayRef<hivm::AddressSpace> addressSpaces,
                           PatternRewriter &rewriter);

void createPropagatorsDown(Operation *op,
                           UnrealizedConversionCastOp propagateOp,
                           PatternRewriter &rewriter);

/// Get propagated core type from cast op, defaulting to CUBE_OR_VECTOR.
TCoreType getCoreType(UnrealizedConversionCastOp op);

/// Get propagated address spaces from cast op.
SmallVector<AddressSpace, 2> getAddressSpace(UnrealizedConversionCastOp op);

/// Return the up-propagator attached to an operand, if present.
UnrealizedConversionCastOp getUpPropagator(OpOperand *operand);

/// Return the down-propagator attached to a result, if present.
UnrealizedConversionCastOp getDownPropagator(OpResult res);

/// Insert a store that materializes `value` into a local buffer. UB targets use
/// memref.alloc; GM workspace targets use memref_ext.alloc_workspace.
hivm::StoreOp
insertStore(Value value, Location loc, PatternRewriter &rewriter,
            std::optional<hivm::AddressSpace> dstAddressSpace = std::nullopt);

/// Insert a load that rematerializes `value` with matching element type.
hivm::LoadOp insertLoad(Value value, Location loc, PatternRewriter &rewriter);

/// Convenience helper to insert store followed by load for `value`.
std::pair<hivm::StoreOp, hivm::LoadOp> insertStoreAndLoad(
    Value value, Location loc, PatternRewriter &rewriter,
    std::optional<hivm::AddressSpace> dstAddressSpace = std::nullopt);

} // namespace PropagatorUtil

LogicalResult verifyPropagation(func::FuncOp funcOp);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_UTILS_H
