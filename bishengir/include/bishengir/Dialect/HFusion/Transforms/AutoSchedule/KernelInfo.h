//===- KernelInfo.h --- Definition for Kernel Info --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H

#include "bishengir/Dialect/HFusion/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/FusibleProducerAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hfusion {
namespace detail {
struct OpInfo {
  /// Topological ordering in the kernel function.
  size_t idx{0};
  /// Dimension information w.r.t. the anchor value for each input and
  /// init/result operands (if applicable).
  SmallVector<BitVector> inputsAnchorDimension;
  SmallVector<BitVector> resultsAnchorDimension;
  /// This describes how the op's axis should be interchanged for each
  /// input and init/result operands (if applicable).
  SmallVector<SmallVector<int64_t>> inputsInterchange;
  SmallVector<SmallVector<int64_t>> resultsInterchange;
};

struct MatmulInfo : public OpInfo {
  size_t tensorAId{0};
  size_t tensorBId{0};
  bool transposeA{false};
  bool transposeB{false};
  int64_t numParallel{0};
  int64_t numReduction{0};
};

struct ReduceInfo : public OpInfo {
  /// Total number of loops (reduction + parallel).
  int64_t numLoops{0};
  /// Original Reduction axes index.
  SetVector<int64_t> reductionDims;
  /// Number of results (for multi-reduce).
  int64_t numResults{0};
};

struct BroadcastInfo : public OpInfo {
  /// Total number of loops.
  int64_t numLoops{0};
  /// Broadcast axes index.
  SetVector<int64_t> broadcastDims;
};

struct ExtractSliceInfo : public OpInfo {
  /// Dims partial sliced
  SetVector<int64_t> partialSlicedDims;
  /// Dims full sliced
  SetVector<int64_t> fullSliceDims;
};

struct TransposeInfo : public OpInfo {
  /// Total number of loops
  int64_t numLoops{0};
  // Dims to be swapped
  std::pair<int64_t, int64_t> permuteDims;
  // Bitwidth of the element type, decides how much to align to
  int64_t elemBitwidth;
  // Check if transpose last dim
  bool transposeLastDim;
};

struct StoreOpInfo : public OpInfo {
  /// By default, when there is no reduce ops, all axes are
  /// `strictlyParallelDims`.
  explicit StoreOpInfo(size_t numLoops);

  // Loosely reduction dims means that the dimension
  // in the output dim corresponds on a reduction dim in the anchor.
  //
  // `looselyReductionDims` and `strictlyParallelDims` are disjoint set.
  SetVector<int64_t> looselyReductionDims;
  SetVector<int64_t> strictlyParallelDims;
};

struct ConcatInfo : public OpInfo {
  /// Rank number.
  int64_t rank{0};
  /// Concat dim.
  int64_t concatDim{0};
  // Bitwidth of the element type, decides how much to align to
  int64_t elemBitwidth;
};

struct CastInfo : public OpInfo {
  /// Rank number.
  int64_t rank{0};
  /// Type of src and dst.
  Type srcType;
  Type dstType;
  // Type of the src and dst element.
  Type srcElemType;
  Type dstElemType;
  // Static shape of cast op.
  SmallVector<int64_t> shape;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// KernelInfo
//===----------------------------------------------------------------------===//

/// Base structure for holding tiling-agnostic kernel information.
/// Kernel info for specific fusion scheduler should derive from this class.
class KernelInfo {
public:
  /// (alignment idx, alignment unit)
  using DimAndAlignment = std::pair<int, int>;

  KernelInfo() = default;
  KernelInfo(FusionKind kind, MLIRContext *ctx) : kind_(kind), ctx_(ctx) {}
  KernelInfo &operator=(KernelInfo const &) = delete;
  virtual ~KernelInfo() = default;

  FusionKind getFusionKind() const { return kind_; }
  static bool classof(const KernelInfo *) { return true; }

  /// Create and initialize dimension analyzer.
  LogicalResult initializeDimensionAnalyzer();

  /// Get the number of bits of the smallest tensor element type in kernel.
  int64_t getSmallestElementTypeBits();

  /// Get the stride alignment dimension and unit.
  /// \note Currently the lowest dimension of
  ///       broadcast/reduce/extract_slice/transpose op needs alignment.
  ///       Return SmallVector for different alignment requirements.
  SmallVector<DimAndAlignment> getStrideAlignments();

  /// Get the size alignment dimension and unit.
  /// The size align rules are simliar as stride align rules,
  /// with different usages:
  /// - size align rules work on tile sizes
  /// - stride align rules work on dim sizes.
  /// \note Currently the dimension of transpose op needs alignment.
  ///       Return SmallVector for different alignments requirements.
  SmallVector<DimAndAlignment> getSizeAlignments();

  /// Get the tile alignment dimension and unit.
  /// \note Tile alignment is the combination of stride and size alignments.
  SmallVector<DimAndAlignment> getTileAlignments();

  MLIRContext *getContext() const { return ctx_; };

  /// Whether the kernel is purely static.
  bool isPureStaticKernel();

  /// Get the `idx`-th function argument of kernel function
  BlockArgument getKernelFuncArg(size_t idx);

  /// Get pointer to dimension analyzer.
  detail::DimensionAnalyzer *getAnalyzer() { return analyzer_.get(); }
  const detail::DimensionAnalyzer *getAnalyzer() const {
    return analyzer_.get();
  }
  /// Get the index of parallel block dim in tiling data.
  uint32_t getParallelBlockDimTilingDataIdx();

  /// Get the index of reduce block dim in tiling data.
  uint32_t getReduceBlockDimTilingDataIdx();

public:
  /// Number of inputs.
  size_t numInputs{0};
  /// Number of outputs.
  size_t numOutputs{0};
  /// Topological ordering of the output values.
  SmallVector<int64_t> outputOrdering{};
  /// Indices to function arguments that are "tied to" function return values.
  SetVector<int64_t> funcArgIdxWithTiedReturnValue{};
  /// Indices to function arguments that are reshaped before use.
  SetVector<int64_t> funcArgWithReshapeIndices;
  /// Mapping from the index of the function return value to the index of the
  /// tied function arguments.
  DenseMap<int64_t, int64_t> returnValueIdx2TiedFuncArg;
  /// Indices to function returns values are reshaped values.
  SetVector<int64_t> returnValueWithReshapeIndices;
  /// Indices to function arguments that needs cache reading.
  SetVector<int64_t> cacheReadFuncArgIndices;
  /// Original kernel name.
  std::string baseKernelName{};
  /// Smallest element type of the tensors in the kernel.
  Type smallestElementType{Type()};
  /// Kernel function's input types.
  SmallVector<Type> inputTypes{};
  /// Kernel function's output types.
  SmallVector<Type> outputTypes{};
  /// Unscheduled, original kernel function.
  func::FuncOp originalKernel{nullptr};
  /// Maximum number of buffers that need to co-exist on local memory at the
  /// same time.
  int64_t maxBufferCnt{0};
  /// Block dimension.
  uint32_t blockDim{0};
  /// Cube tiling tuning parameters for SingleCube Schedule.
  ArrayRef<int64_t> cubeTilingTuning{};

  /// Operation and their infos.
  std::map<Operation *, detail::MatmulInfo> matmulOp2Info{};
  std::map<Operation *, detail::ReduceInfo> reduceOp2Info{};
  std::map<Operation *, detail::ExtractSliceInfo> extractSliceOp2Info{};
  std::map<Operation *, detail::TransposeInfo> transposeOp2Info{};
  std::map<Operation *, detail::BroadcastInfo> broadcastOp2Info{};
  std::map<Operation *, detail::StoreOpInfo> storeOp2Info{};
  std::map<Operation *, detail::ConcatInfo> concatOp2Info{};
  std::map<Operation *, detail::CastInfo> castOp2Info{};

  /// Kernel function's inputs, contains two types of values following the order
  /// of origin kernel arguments:
  /// 1. if the kernel arg is not reshaped, use the origin `BlockArgument` value
  /// 2. if the kernel arg is reshaped, use the result value from reshaped op
  SmallVector<Value> inputValues{};

  /// Kernel function's return values, contains two types of values following
  /// the order of origin kernel outputs (i.e. operands of `func.return` op):
  /// 1. if the kernel output is not reshaped, use the original value
  /// 2. if the kernel output is reshaped, use the value before reshaped
  SmallVector<Value> outputValues;

  /// Whether multi core reduce is enabled
  bool enableMultiCoreReduce{false};

protected:
  /// Dimension analyzer.
  std::shared_ptr<detail::DimensionAnalyzer> analyzer_;

private:
  /// Get the stride alignment dimension and factor for reduce ops.
  std::optional<DimAndAlignment> getStrideAlignmentsForReduceOp();

  /// Get the stride alignment dimension and factor for broadcast ops.
  std::optional<DimAndAlignment> getStrideAlignmentsForBroadcastOp();

  /// Get the stride alignment dimension and factor for extract slice ops.
  std::optional<KernelInfo::DimAndAlignment>
  getStrideAlignmentsForExtractSliceOp();

  /// Get the stride alignment dimension and factor for transpose ops.
  /// Although transpose requires both transpose axes should be aligned, we only
  /// have to return the alignment dim with larger alignment unit.
  std::optional<DimAndAlignment> getStrideAlignmentsForTransposeOp();

  /// Get the stride alignment dimension and factor for concat ops.
  std::optional<DimAndAlignment> getStrideAlignmentsForConcatOp();

  /// Get the size alignment dimension and factor for transpose ops.
  SmallVector<DimAndAlignment> getSizeAlignmentsForTransposeOp();

  /// Get the size alignment dimension and factor for cast ops.
  SmallVector<DimAndAlignment> getSizeAlignmentsForCastOp();

  /// Get the size alignment dimension and factor for concat ops.
  SmallVector<DimAndAlignment> getSizeAlignmentsForConcatOp();

  /// Underlying fusion kind.
  FusionKind kind_;
  MLIRContext *ctx_{nullptr};
};
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H
