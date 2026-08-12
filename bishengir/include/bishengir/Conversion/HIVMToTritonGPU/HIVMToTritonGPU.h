//===- HIVMToTritonGPU.h - HIVM to TritonGPU conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H
#define BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H

#include <memory>
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;
class Type;

#define GEN_PASS_DECL_CONVERTHIVMTOTRITONGPU
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {

/// Lowering options for the HIVM → TritonGPU conversion pipeline.
struct LowerToTritonGPUOptions {
  /// When true, MemRef function arguments are converted to bare pointers
  /// instead of full MemRef descriptors.
  bool useBarePtrCallConv = false;
};

/// Type converter for lowering HIVM types to TritonGPU-compatible types.
///
/// Extends mlir::LLVMTypeConverter to provide type conversion logic specific
/// to the HIVM → TritonGPU conversion pipeline. It handles:
/// - Function signature conversion (bare-ptr and descriptor-based calling
///   conventions)
/// - MemRef type conversion (to bare pointer or multi-field descriptor)
/// - Address space mapping for different memory scopes (global, shared, etc.)
///
/// The conversion behavior is configured by LowerToTritonGPUOptions.
class TritonTypeConverter : public mlir::LLVMTypeConverter {
 public:
  TritonTypeConverter(mlir::MLIRContext *ctx,
      const LowerToTritonGPUOptions &options,
      const mlir::DataLayoutAnalysis *analysis = nullptr)
      : mlir::LLVMTypeConverter(ctx, analysis), options(options) {
  }

  /// Converts a Triton function signature, expanding MemRef arguments into
  /// bare pointers or descriptor fields based on the calling convention.
  /// Also collects shared-memory argument indices and fractal layout attribute
  /// mappings.
  FunctionType convertTTFunctionSignature(func::FuncOp funcOp,
      bool useBarePtrCallConv, SignatureConversion &result,
      SmallVector<std::optional<int>> &sharedIds,
      SmallVector<std::pair<int, Attribute>> &fractalAttrArgIds) const;

  /// Converts a type according to the Triton calling convention. When
  /// useBarePointerCallConv is true, MemRef types become bare pointers;
  /// otherwise they become descriptor-based types.
  Type convertTTCallingConventionType(
      Type type, bool useBarePointerCallConv = false) const;

  /// Converts a MemRef type to a bare pointer (tt.ptr) in the appropriate
  /// Triton address space based on the MemRef's memory space.
  Type convertMemRefToTTBarePtr(BaseMemRefType type) const;

  /// Returns the Triton address space corresponding to a MemRef's memory
  /// space attribute. Maps HIVM memory spaces (e.g. UB, shared) to Triton
  /// address space values.
  uint32_t getMemRefAddressSpace(BaseMemRefType type) const;

  /// Returns the descriptor field types for a ranked MemRef. A descriptor
  /// typically contains: { basePtr, alignedPtr, offset, sizes[rank],
  /// strides[rank] }.
  SmallVector<Type, 5> getMemRefDescriptorFields(MemRefType type) const;

  /// Returns the descriptor field types for an unranked MemRef, which
  /// contains the rank and a pointer to a descriptor struct.
  SmallVector<Type, 2> getUnrankedMemRefDescriptorFields(Type elemTy) const;

  /// Returns the lowering options associated with this type converter.
  const LowerToTritonGPUOptions &getOptions() const {
    return options;
  }

 private:
  LowerToTritonGPUOptions options;
};

Type HIVMToTritonTypeConvert(Type ty);
void populateHIVMToTritonPatterns(RewritePatternSet &patterns);
void populateFuncToTritonPatterns(
    TritonTypeConverter &converter, RewritePatternSet &patterns);
void populateBufferizationToTritonPatterns(RewritePatternSet &patterns);
void populateTensorToTritonPatterns(RewritePatternSet &patterns);
void populateAffineToTritonPatterns(RewritePatternSet &patterns);
void populateMemRefLoadToTritonPatterns(RewritePatternSet &patterns);
void populateReinterpretCastToUnrealizedCastPatterns(RewritePatternSet &patterns);
void populateExtractAlignedPointerToTritonPatterns(RewritePatternSet &patterns);

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult structTTFuncArgTypeConverter(const TritonTypeConverter &converter,
    Type type, SmallVectorImpl<Type> &result);

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult bareTTPtrFuncArgTypeConverter(
    const TritonTypeConverter &converter, Type type,
    SmallVectorImpl<Type> &result);
} // namespace hivm

/// Creates a pass to convert the hivm dialect to the triton dialect.
std::unique_ptr<Pass> createHIVMToTritonGPUConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOTRITONGPU_H
