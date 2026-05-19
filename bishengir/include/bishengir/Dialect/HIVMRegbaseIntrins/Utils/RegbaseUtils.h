//===- RegbaseUtils.h - Utilities to support HIVMRegbase dialect -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_HIVM_REGBASEINTRINS_UTILS_REGBASEUTILS_H
#define MLIR_DIALECT_HIVM_REGBASEINTRINS_UTILS_REGBASEUTILS_H

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace hivm_regbaseintrins {
bool isaVstsInstrOp(Operation *op);

Operation *findPredicateMapOp(Value result, PatternRewriter &rewriter,
                              Location loc, VectorType argVectorType);

Operation *buildVBrOp(Value broadcastSource, Type elementType,
                      PatternRewriter &rewriter);

Operation *buildVldsOp(Value basePtr, Value offset, Type elementType,
                       PatternRewriter &rewriter, unsigned int dist = 0,
                       bool postUpdate = false);

Operation *buildVldusOp(Value basePtr, Value alignData, Type elementType,
                        PatternRewriter &rewriter);

Operation *buildVldusPostOp(Value basePtr, Value alignData, Value cceVL,
                            Type elementType, PatternRewriter &rewriter);

Operation *buildVldasOp(Value basePtr, PatternRewriter &rewriter);

Operation *buildVstsOp(Value srcRegister, Value basePtr, Value offset,
                       Value predicateVector, PatternRewriter &rewriter,
                       unsigned int dist = 0);

Operation *buildVstasOp(Value srcRegister, Value basePtr,
                        PatternRewriter &rewriter);

Operation *buildVstusPostOp(Value srcRegister, Value basePtr, Value eleSize,
                            Value tmp, PatternRewriter &rewriter);

Operation *buildPsetOp(Value pattern, unsigned elementBitWidth,
                       PatternRewriter &rewriter);

Operation *buildPpackOp(Location loc, Value part, Value src,
                        PatternRewriter &rewriter);

Operation *buildPunpackOp(Location loc, Value part, Value src,
                          PatternRewriter &rewriter);

Operation *buildVpackOp(Location loc, Value part, Value src, Type vectorType,
                        PatternRewriter &rewriter);

Operation *buildVunpackOp(Location loc, Value part, Value src, Type vectorType,
                           PatternRewriter &rewriter);
                                                  
Operation *buildPltOp(Value count, unsigned elementBitWidth,
                      PatternRewriter &rewriter);

Operation *buildPstuOp(Value data, Value dataPtr, PatternRewriter &rewriter,
                       int elementAlignment, Value alignData);

Operation *buildMovvpOp(Location loc, Type type, Value extractOp,
                        PatternRewriter &rewriter, int elementAlignment);

Operation *buildVdupOp(Value srcRegister, Value predicateVector,
                       Type elementType, PatternRewriter &rewriter);

Operation *buildAddOp(Value lhs, Value rhs, Value pred,
                      PatternRewriter &rewriter);

Operation *buildMaxOp(Value lhs, Value rhs, Value pred,
                      PatternRewriter &rewriter);

Operation *buildCumulateAddOp(Value lhs, Value pred, PatternRewriter &rewriter);

Operation *buildCumulateMaxOp(Value lhs, Value pred, PatternRewriter &rewriter);

Operation *buildVselOp(Value srcReg1, Value srcReg2, Value predicateVector,
                       PatternRewriter &rewriter);

Value remove1DVectorHighDims(Location loc, Value value,
                             PatternRewriter &rewriter);

int64_t getVectorSizeByElementType(Type t);

bool isHIVMSupportedType(Type t);

int getVldsBrcDist(const int &dataBitWidth);

VectorType adjustTypeToDavidSupportedType(VectorType vecTy);

Value createPredicateVector(Operation *op, VectorType vecType,
                            PatternRewriter &rewriter);

VectorType createVLVectorType(Type elementType);

bool deviceFuncHasVFCalls(func::CallOp caller, SymbolTable &symtab);

Value getVLRegValueOrSelf(Value orig, OpBuilder &b);

} // namespace hivm_regbaseintrins
} // namespace mlir

#endif // MLIR_DIALECT_HIVM_REGBASEINTRINS_UTILS_REGBASEUTILS_H
