//===- DimensionAnalyzer.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/UnionFind.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#ifndef BISHENGIR_DIALECT_DIMENSION_ANALYZER_H
#define BISHENGIR_DIALECT_DIMENSION_ANALYZER_H

namespace mlir {
namespace detail {

using DimensionPosition = std::pair<int64_t, int64_t>;
static constexpr int64_t kUndefinedShaped = -1;
static constexpr DimensionPosition kMaxDimPos = {
    std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};
/// Base class for Union-Find data structure.

/// Simple Union-Find implementation.
class SimpleUnionFind : public UnionFindBase {
public:
  SimpleUnionFind(size_t n = 0) : UnionFindBase(n) {}

  virtual ~SimpleUnionFind() = default;
};

/// Extended Union-Find with additional shape information.
class ExtendedUnionFind : public UnionFindBase {
public:
  ExtendedUnionFind(size_t n = 0)
      : UnionFindBase(n), shape_(n, ShapedType::kDynamic) {}
  bool join(int a, int b) override;
  void allocateMinimum(size_t n) override;
  std::pair<DimensionPosition, int64_t> getMinParentAndShapePair(int a);

  SmallVector<int64_t> shape_;
  SmallVector<std::pair<int64_t, int64_t>> minParentIndex_;
};

struct ConnectedLeftRight {
  bool leftConnected = true;
  bool rightConnected = true;
  tensor::reshape_utils::ElementKind elementKind =
      tensor::reshape_utils::ElementKind::NoMutation;
  ConnectedLeftRight() = default;
};

class DimensionAnalyzerOptions {
public:
  DimensionAnalyzerOptions() = default;
  bool registerBased = false;
};

/// The DimensionAnalyzer class implements the logic to analyze dimensions
/// relationship and can flatten operations in MLIR.
/// It processes operations and flattens their iteration spaces when possible.
class DimensionAnalyzerBase {
public:
  /// Dimension is a pair<A: Value, B: long> represents a dynamic symbol
  /// from the ShapedType A at index B.
  /// For example %0 = tensor.empty() : tensor<Mx20xNxf32>,
  /// M can be represented by {%0, 0}
  /// N can be represented by {%0, 2}
  using Dimension = std::pair<Value, int64_t>;

  /// Constructor that takes the operation to flatten.
  explicit DimensionAnalyzerBase(Operation *op, DimensionAnalyzerOptions options = {});
  virtual ~DimensionAnalyzerBase() = default;

  virtual LogicalResult initialize();

  //===--------------------------------------------------------------------===//
  // Dimension Analyzer API.
  //===--------------------------------------------------------------------===//

  SmallVector<int64_t> getDimShape(Value v);

  /// @description: Takes an Dimension object as input and returns the
  /// earliest occurrence of dimension.
  ///
  /// @param dim The input dimension to find the earliest dimension.
  /// @return The earliest dimension based on the given dimension.
  Dimension getEarliestDimension(Dimension dim);

  /// @description: Get the Dimension from the union find parent index
  ///
  /// @param parentIndex The index of the parent in the union find structure
  /// @return Dimension The Dimension corresponding to the parent
  /// index
  Dimension getDimension(int64_t parentIndex);

  /// @description: Check whether \c lhs is equivalent to \c rhs
  ///
  /// @param lhs A pair of value and dim idx.
  /// @param rhs A pair of value and dim idx.
  /// @param isStrict if set to false, two dims are considered equal if they're
  ///                 structurally equivalent (e.g., before and after concat).
  bool areDimensionsEqual(Dimension lhs, Dimension rhs, bool isStrict = false);

protected:
  /// Helper to represent connected dimensions.
  using DimensionIndex = SmallVector<int64_t>;
  using DimensionShape = SmallVector<int64_t>;
  using ArgumentIndex = SmallVector<int64_t>;

  //===--------------------------------------------------------------------===//
  // Functions for initialization
  //===--------------------------------------------------------------------===//

  /// Initializes internal data structures.
  virtual void initializeStructures();

  /// Processes the arguments using BFS traversal.
  virtual void processBFS();

  /// Unifies groups of segments.
  void unifyGroups();

  void propagateConnection(int parent, int child);
  void propagateConnection();
  void spreadConnection();

  // Logging and debugging purposes
  void markAttributes();
  /// Utility functions.
  int64_t allocateArguments(int rank, ArrayRef<int64_t> dimensionRef);

  /// Updates the previous type of a value.
  void updatePreviousType(const Value &val);
  void updatePreviousType(const Value &val, const ShapedType &curType);

  /// Propagates collapse information.
  void collapsePropagateOrVerify(Operation *op, const Value &refVal);
  void collapsePropagateOrVerify(const Value &newVal, const Value &arg);
  void initCollapseOrVerify(const Value &val, int64_t refPtr);

  // Receive an argument value and adjust it
  void processArgument(Value arg);

  /// Creates dummy references if they don't exist.
  void createDummyRefIfNotExist(ArrayRef<Value> values);

  /// Processes the arguments before running BFS.
  virtual void combineInferable();

  /// Shape element binding and materialization system.
  ///
  /// This system maintains a mapping between shape elements and unique
  /// identifiers for efficient analysis and materialization of dynamic
  /// dimensions.
  ///
  /// Example workflow:
  /// Initial tensor shape extraction:
  ///  ```
  ///  // For a value `Val` with shape <2xAxBx4x5xf32>
  ///  SmallVector<OpFoldResult> shape = tensor::getMixedValues(Val);
  ///  // shapeElements refer to shape[0], shape[1], shape[2], shape[3],
  ///  shape[4]
  ///  // Corresponding to dimensions: 2, A, B, 4, 5
  ///  ```
  ///
  /// After transformations (e.g., reduction):
  ///  ```
  ///  // Resulting shape: <AxBx5xf32>
  ///  // Preserved identifiers: 5 (A), 6 (B)
  ///  // Shape indices: [5, 6, 8]
  ///  ```
  ///
  /// To materialize dimension A or B, use the stored identifiers (5, 6)
  /// Reverse lookup: identifier -> original shapeElement -> materialized
  ///
  /// Throughout the entire function, only the identifiers are stored and
  /// propagated through transformations. When materialization is needed, the
  /// reverse mapping from identifier to shapeElement enables reconstruction of
  /// the actual dimension values.
  ///
  /// Note: Every dimension is represented using a unique index in the analyzer.
  /// This enables efficient tracking and materialization of dynamic shape
  /// elements across complex transformations, also if there are any shape
  /// element bindings, these identifiers can be marked and unified through
  /// the union-find disjoint set method

  /// @description: Computes and populates the reverse mapping from shape
  /// elements to tensor values.
  ///
  /// This function builds a reverse lookup map (reverseShapeElem) that maps
  /// shape element indices to their corresponding tensor shape `Value`.
  /// It processes both function arguments and operation results, establishing
  /// the relationship between analyzed shape elements and the tensors they
  /// belong to.
  void computeReverseElementMap();

  //===--------------------------------------------------------------------===//
  // Processors for operations
  //===--------------------------------------------------------------------===//

  /// Processes an individual operation during BFS.
  virtual bool processOperation(Operation *op, Value current);

  /// Processes element-wise operations.
  void processParallelOp(Operation *op, Value current);

  /// Processes values to unify shapes.
  void processValue(Value v, Value current);

  /// Processes operations that decrease dimensions (e.g., reductions).
  size_t processDecreasingDimensions(ArrayRef<int64_t> inputArgs,
                                     ArrayRef<int64_t> dimensions,
                                     const Value &output);

  /// Processes permutations (e.g., transpose operations).
  size_t processPermutation(ArrayRef<int64_t> inputArgs, ArrayRef<int64_t> perm,
                            const Value &output);

  /// Processes matmul operations.
  void processMatmulOp(Operation *op, bool isTransposeA = false,
                       bool isTransposeB = false);

  /// Processes batch_matmul op.
  void processBatchMatmulOp(linalg::BatchMatmulOp batchMatmulOp);

  /// Processes concat and pad op.
  void processConcatOp(tensor::ConcatOp concatOp);

  void processPadOp(tensor::PadOp padOp);

  template <class T, typename = std::enable_if_t<
                         std::is_same_v<T, tensor::ExtractSliceOp> ||
                         std::is_same_v<T, tensor::InsertSliceOp>>>
  void processSlicingOp(T slicingOp);
  void processExtractSliceOp(tensor::ExtractSliceOp extractSliceOp);
  void processIfOp(scf::IfOp op);
  void processForOp(scf::ForOp op);
  void processInsertOp(tensor::InsertOp insertOp);
  void processInsertSliceOp(tensor::InsertSliceOp insertSliceOp);

  //===--------------------------------------------------------------------===//
  // Union-find handler
  //===--------------------------------------------------------------------===//

  void joinShape(int a, int b);
  void joinCollapser(int a, int b);
  void disconnect(int a, int b);
  bool isConnected(int a, int b);

  //===--------------------------------------------------------------------===//
  // Helper function
  //===--------------------------------------------------------------------===//
  void dumpModuleOP() const;

  void dumpArgumentsRef();

  void dumpIsConnected();

  void dumpArgumentsRefPointer();

  std::optional<SmallVector<int64_t>> getParentShapeRef(Value v);

  /// @description: Retrieves the shape of a specific element by its index.
  ///
  /// This function looks up a shape element by its index and constructs an
  /// OpFoldResult representing the dimension size. It uses the reverse element
  /// map to find the tensor value containing the element, then creates a
  /// tensor.dim operation for the appropriate dimension index.
  ///
  /// @param elemIndex The index of the shape element to retrieve
  /// @return Optional OpFoldResult containing the dimension size, or nullopt if
  /// not found
  std::optional<OpFoldResult> getElementShape(int elemIndex);

  SmallVector<int64_t> getArgumentRef(Value v) const;
  SmallVector<int64_t> getArgumentRefOrCreateDummy(Value v);

protected:
  Operation *op_;
  int64_t dimensionAllocation_ = 0;

  SmallVector<Value> argumentList_;
  ArgumentIndex argumentIndex_;
  SmallVector<Operation *> outList_;
  int64_t argumentTotalLength_ = 0;
  SmallVector<ConnectedLeftRight, 4> isConnected_;
  SmallVector<DimensionIndex> argumentsRef_;
  DenseMap<Value, int> argumentsRefPointer_;
  DenseMap<Value, ShapedType> previousType_;

  // +----------------------------------+  +----------------------------------+
  // |%arg0 = <AxBxCxf32>               |  |%arg1 = <AxDxCxf32>               |
  // |solverShapeElem =     [S0, S1, S2]|  |solverShapeElem =     [S0, S3, S2]|
  // |solverCollapserElem = [S0, S1, S2]|  |solverCollapserElem = [S0, S1, S2]|
  // +------------+---------------------+  +--------------------+-------------+
  //              |                                             |
  //              |                                             |
  //              |                                             |
  //              |   +-------------------------------------+   |
  //              |   |%concat = %arg0 + %arg1 : <AxExCxf32>|   |
  //              +-->|solverShapeElem =     [S0, S4, S2]   |<--+
  //                  |solverCollapserElem = [S0, S1, S2]   |
  //                  +------------------+------------------+
  //                                     |
  //                                     |
  //                    +----------------v-----------------+
  //                    |%unary: %concat <AxExCxf32>       |
  //                    |solverShapeElem =     [S0, S4, S2]|
  //                    |solverCollapserElem = [S0, S1, S2]|
  //                    +----------------------------------+
  //
  // after analysis solverCollapserElem_ considers B, D and E as S1
  // but the solverShapeElem_ distinguish the S1, S3, and S4
  // thats how when we want to analyze the max dimension,
  // we do max across the S1's owner axis
  // thus, giving the max dynamic as max(S1, S3, S4)
  //
  // This has type inference, weak union find
  std::unique_ptr<ExtendedUnionFind> solverShapeElem_;

  // This has no type inference, strong collapse union find
  // Related: concatOp, padOp, extractSliceOp
  std::unique_ptr<SimpleUnionFind> solverCollapserElem_;
  std::unique_ptr<SimpleUnionFind> solverSegments_;

  DenseMap<int64_t, Value> reverseShapeElem_;
  bool bindUsingTensorDim = true;
  DimensionAnalyzerOptions options;

  bool isAllowedType(Type type);
  bool isHeadOperation(Operation *op);
  bool isTailOperation(Operation *op);
  void separateMemref(Value memrefVal);
  void separateGroup(Value val, BitVector contiguousMask, ArrayRef<int64_t> shape);

private:
  void combineEmptyOp(Value arg);
  void linkDimToEmpty(tensor::DimOp dimOp, int64_t emptyRefElement);
};

} // namespace detail
} // namespace mlir
#endif