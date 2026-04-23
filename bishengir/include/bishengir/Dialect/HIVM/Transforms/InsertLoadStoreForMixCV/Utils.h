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

/// Insert a store that materializes `value` into local workspace.
hivm::StoreOp insertStore(Value value, Location loc, PatternRewriter &rewriter);

/// Insert a load that rematerializes `value` with matching element type.
hivm::LoadOp insertLoad(Value value, Location loc, PatternRewriter &rewriter);

/// Convenience helper to insert store followed by load for `value`.
std::pair<hivm::StoreOp, hivm::LoadOp>
insertStoreAndLoad(Value value, Location loc, PatternRewriter &rewriter);

} // namespace PropagatorUtil

LogicalResult verifyPropagation(func::FuncOp funcOp);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INSERT_LOAD_STORE_FOR_MIX_CV_UTILS_H
