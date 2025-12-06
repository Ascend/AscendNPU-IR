//===- TilingUtils.h -- Utilities for Auto Schedule Tiling ------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_TILINGUTILS_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_TILINGUTILS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/StringRef.h"

#include <variant>

namespace mlir {
namespace hfusion {

class KernelInfo;
class TilingInfo;

/// Get \c {hacc.arg_type = #hacc.arg_type<tiling_data>} attribute.
NamedAttribute getTilingDataAttr(OpBuilder &opBuilder);

/// Get \c {hacc.arg_type = #hacc.arg_type<tiling_key>} attribute.
NamedAttribute getTilingKeyAttr(OpBuilder &opBuilder);

//===----------------------------------------------------------------------===//
// Expr
//===----------------------------------------------------------------------===//

enum class ExprKind : uint8_t {
  // Regular expression, can represent a constant or the result of arithmetic
  // computation
  kRegular = 0,
  // A placeholder symbol for tensor dimension, either bound at runtime
  // (dynamic shape) or compile time (static shape).
  kDimSymbol
};

class StmtExprBuilder;

/// Base type for Expression.
///
/// Expression can be used to compute tiling values in host tiling functions.
/// Each expression is bound to a IR value that is constructed on-the-fly as
/// users expression computation logic using expressions.
///
/// The bound IR value is either
///   1) the result of `affine::ApplyOp` or
///   2) the result of `arith::CmpXXOp` or
///   3) the result of `tensor::DimOp`.
class Expr {
public:
  Expr() = default;
  virtual ~Expr() = default;
  explicit Expr(ExprKind kind) : kind_(kind){};
  explicit Expr(const Value &value, ExprKind kind, StmtExprBuilder *builder)
      : v_(value), kind_(kind), builder_(builder) {}

  // FIXME: Implement me
  Expr operator+(int64_t cst);
  Expr operator+(const Expr &other);
  Expr operator-(int64_t cst);
  Expr operator-(const Expr &other);
  Expr operator*(int64_t cst);
  Expr operator*(const Expr &other);
  Expr ceilDiv(uint64_t cst);
  Expr ceilDiv(const Expr &other);
  Expr floorDiv(uint64_t cst);
  Expr floorDiv(const Expr &other);
  Expr operator%(uint64_t cst);
  Expr operator%(const Expr &other);

  /// Returns the next integer that is greater than or equal to \p this and is a
  /// multiple of \p align. \p align must be non-zero.
  ///
  /// Examples:
  /// \code
  ///   alignTo(5, 8) = 8
  ///   alignTo(17, 8) = 24
  ///   alignTo(321, 255) = 510
  /// \endcode
  Expr alignTo(uint64_t align);
  Expr alignTo(const Expr &align);

  /// Returns the largest integer less than or equal to \p this and is a
  /// multiple of \p align. \p align must be non-zero.
  ///
  /// Examples:
  /// \code
  ///   alignTo(5, 8) = 0
  ///   alignTo(17, 8) = 16
  ///   alignTo(321, 255) = 255
  /// \endcode
  Expr alignDown(uint64_t align);
  Expr alignDown(const Expr &align);

  Expr operator>(int64_t cst);
  Expr operator>(const Expr &other);
  Expr operator>=(int64_t cst);
  Expr operator>=(const Expr &other);
  Expr operator<(int64_t cst);
  Expr operator<(const Expr &other);
  Expr operator<=(int64_t cst);
  Expr operator<=(const Expr &other);
  Expr operator==(int64_t cst);
  Expr operator==(const Expr &other);
  Expr operator!=(int64_t cst);
  Expr operator!=(const Expr &other);

  /// Get the underlying Value.
  Value getMaterializedValue() const { return v_; }

  MLIRContext *getContext() const { return v_.getContext(); }

  /// Return the classification for this type.
  ExprKind getExprKind() const { return kind_; }

  static bool classof(const Expr *) { return true; }

  /// Get the underlying builder with location.
  StmtExprBuilder &getBuilder() { return *builder_; }

private:
  /// Underlying IR Value.
  Value v_;
  /// Expression kind.
  ExprKind kind_{ExprKind::kRegular};
  /// OpBuilder
  StmtExprBuilder *builder_{nullptr};
};

/// Expression representing a tensor's dimension size.
class DimSymbol : public Expr {
public:
  DimSymbol() : Expr(ExprKind::kDimSymbol) {}
  ~DimSymbol() override = default;
  explicit DimSymbol(const Value &value, StmtExprBuilder *builder)
      : Expr(value, ExprKind::kDimSymbol, builder) {}
  static bool classof(const Expr *e) {
    return e->getExprKind() == ExprKind::kDimSymbol;
  }
};

/// Commonly used operations for Expr.
Expr max(Expr lhs, Expr rhs);
Expr max(Expr lhs, int64_t rhs);
Expr min(Expr lhs, Expr rhs);
Expr select(Expr condition, Expr trueValue, Expr falseValue);
Expr select(Expr condition, Expr trueValue, int64_t falseValue);

//===----------------------------------------------------------------------===//
// Stmt
//===----------------------------------------------------------------------===//

class Stmt {
public:
  Stmt() = default;
  virtual ~Stmt() = default;
  explicit Stmt(StmtExprBuilder *builder) : builder_(builder) {}

  StmtExprBuilder &getBuilder() { return *builder_; }

private:
  StmtExprBuilder *builder_;
};

class CallStmt : public Stmt {
public:
  CallStmt() = default;
  ~CallStmt() override = default;
  explicit CallStmt(StmtExprBuilder *builder, func::CallOp callOp)
      : Stmt(builder), callOp_(callOp) {}

private:
  func::CallOp callOp_;
};

//===----------------------------------------------------------------------===//
// StmtExprBuilder
//===----------------------------------------------------------------------===//
class KernelInfo;

/// Stmt and Expression Builder class.
///
/// Users can only create constant-value `Expr` or `DimSymbol` using this class.
/// The base auto scheduler is responsible for creating an instance of
/// `StmtExprBuilder` and setting the insertion point into the host tiling
/// function.
class StmtExprBuilder : public OpBuilder {
public:
  explicit StmtExprBuilder(MLIRContext *ctx) : OpBuilder(ctx) {}
  explicit StmtExprBuilder(TilingInfo *info, KernelInfo *kernelInfo,
                           MLIRContext *ctx)
      : OpBuilder(ctx), tilingInfo_(info), kernelInfo_(kernelInfo) {}

  explicit StmtExprBuilder(ModuleOp m, MLIRContext *ctx)
      : OpBuilder(ctx), module_(m) {}

  /// Create an `Expr` holding a int64_t constant value.
  Expr createConstExpr(int64_t cst);

  /// Create a `DimSymbol` that represents the `dimIdx`-th dimension size of
  /// the interested value of the current to-be-scheduled kernel.
  /// If the `tensorIdx`-th arg is not reshaped in origin kernel, directly
  /// create the dim symbol from the `tensorIdx`-th arg. Otherwise, create the
  /// dim symbol from the result value reshaped from the `tensorIdx`-th arg.
  Expr createDimSymbolExpr(size_t tensorIdx, size_t dimIdx);

  /// Create `DimSymbol`s that represents the dimension size of
  /// the interested value of the current to-be-scheduled kernel from `startDim`
  /// to `endDim` (with step = 1).
  SmallVector<Expr> createDimSymbolExprs(size_t tensorIdx, size_t startDim,
                                         size_t endDim);

  /// Create a `DimSymbol` that represent the `dimIdx`-th dimension size
  /// if dynamic shape or create `Expr` holding a int64_t constant value if
  /// static shape.
  Expr createExpr(Value val, int64_t idx, IRMapping &mapper);

  /// Create a `CallStmt` to the existing function named `funcName` with the
  /// given `operands`.
  ///
  /// Assertion will be raised when:
  ///   1) the function is not found;
  ///   2) the number of operands does not match the function's arity.
  CallStmt createCallStmt(FlatSymbolRefAttr funcName,
                          SmallVector<Value> operands);

  /// Create a `CallStmt` to an external function named `funcName` with the
  /// given `operands`. If function is not present, a function declaration
  /// will be created based on the operand type.
  ///
  /// Assertion will be raised when:
  ///   1) the function is not found;
  ///   2) the function could not be created;
  CallStmt createExternCallStmt(FlatSymbolRefAttr funcName,
                                SmallVector<Value> operands,
                                StringAttr externLibraryPath);

  // Create an `AssertOp` that verifies a predicate is true
  void createConstraintVerification(const Expr &predicate,
                                    llvm::StringRef errorMessage);

private:
  /// Create a `DimSymbol` that represent the `dimIdx`-th dimension size of
  /// the `tensorValue` of the current to-be-scheduled kernel.
  /// The `tensorValue` can be tensor argument or tensor value reshaped from
  /// tensor argument.
  Expr createDimSymbolExpr(Value tensorValue, size_t dimIdx);

  /// Recursively clone op operands first, then clone current op.
  /// Return the cloned op.
  Operation *recursivelyCloneOp(Operation *op, IRMapping &mapper);
  Value recursivelyCloneValue(Value val, IRMapping &mapper);

private:
  ModuleOp module_;

  TilingInfo *tilingInfo_{nullptr};

  KernelInfo *kernelInfo_{nullptr};

  DenseMap<std::pair<Value, int64_t>, Expr> exprMap_;
};

/// Tiling Key is a compile-time constant value of type `int64_t`. It should be
/// a unique identifier of a tiling case. The exact meaning of each key is
/// determined by the scheduler.
using TilingKey = int64_t;

//===----------------------------------------------------------------------===//
// TilingData
//===----------------------------------------------------------------------===//

/// The `TilingData` class represents a binding between host tiling data, device
/// kernel argument, and `ValueHandles` used by auto-scheduler's schedule
/// operations.
///
/// For example, consider the following IR :
/// \code
/// func.func private @host_tiling(...) -> (i64, i64)
/// func.func private @device_kernel(..., %tiling_data0 : i64, %tiling_data1:
/// i64)
/// \endcode
///
/// During schedule, if one which to use a tiling data's value to perform
/// scheduling, he/she can get a "handle" that points to device kernel argument
/// tied to that tiling data.
///
/// For instance, the following code snippet:
/// \code
///   TilingData *ubTileSize = tilingInfo->getTilingData(0);
///   ValueHandle* ubTilingDataHandle = getTilingDataHandle(*ubTileSize, ...);
///   tileUsingFor(targetOp, ubTilingDataHandle, ...);
/// \endcode
///
/// will generate the following schedule sequence:
/// \code
/// %arg_handle = transform.func.get_func_argument %arg0[N] ...
/// transform.structured.tile_using_for %target_op tile_sizes [%arg_handle] ...
/// \endcode
///
/// which finally produces the scheduled kernel:
/// \code
/// func.func @device_kernel(..., %tiling_data0 : i64, %tiling_data1: i64) {
///   scf.for %arg7 = %c0 to %4 step %tiling_data0 iter_args(...)
///   ...
/// }
/// \endcode
///
/// The underlying tiling data storage is either an `Expr` or a constant value
/// of `int64_t` type.
struct TilingData {
public:
  using TilingDataTy = std::variant<std::unique_ptr<Expr>, int64_t>;

  TilingData() = default;
  explicit TilingData(Expr &&data, Type t)
      : data_(TilingDataTy(std::make_unique<Expr>(data))), t_(t) {}

  /// Returns whether the tiling data is constant.
  bool isConst() const;

  /// Get `Expr` corresponding to the tiling data. Raise exception if the data
  /// is constantized.
  Expr *getExpr() const;

  /// Get constantized value of the tiling data. Rase exception if the data
  /// is not constantized.
  int64_t getConst() const;

  /// Query the tiling data type.
  Type getType() { return t_; }

  /// Getters and setters for the value handle pointer.
  ValueHandle *getHandle() const { return vh_; }
  void setHandle(ValueHandle *vh) { vh_ = vh; }

  /// Set tiling data to expression or to constant value.
  void setData(int64_t newData);
  void setData(Expr &&newData);

  /// Set a static, heuristic tiling value for the given tiling key.
  ///
  /// This is useful in dynamic shape cases because we might know
  /// for certain that the tiling size for a particular tiling case
  /// is going to be a constant value. But because the tiling computation
  /// is common for all the tiling cases, we cannot reduce it to a constant
  /// value at compile time, and hinders further optimizations.
  void setHeuristicValueForKey(TilingKey key, int64_t hint);
  std::optional<int64_t> getHeuristicValueForKey(TilingKey key) const;

  /// Remove all heuristics.
  void resetHeuristics();

  /// Getters and setters for the tiling data's position index within kernel
  /// function's input argument.s
  size_t getPos() const { return pos_; }
  void setPos(size_t pos) { pos_ = pos; }

private:
  /// Tiling data storage.
  TilingDataTy data_;
  /// Type of the tiling data.
  Type t_;
  /// Position within the to-be-scheduled function.
  size_t pos_{0};
  /// Bound value handle pointer using during scheduling.
  ValueHandle *vh_{nullptr};
  /// Mapping from tiling key to the heuristic tiling factor.
  DenseMap<TilingKey, int64_t> heuristicForKey_{};
};

//===----------------------------------------------------------------------===//
// TilingCases
//===----------------------------------------------------------------------===//

/// Tiling Cases are a collection of unique Tiling Keys. Each key corresponds
/// to a schedule implementation.
struct TilingCases {
  using iterator = SetVector<TilingKey>::iterator;
  using const_iterator = SetVector<TilingKey>::const_iterator;
  iterator begin() const { return cases.begin(); }
  iterator end() const { return cases.end(); }

  /// Add a tiling case key.
  ///
  /// \param caseKey
  /// \return Whether the case key is successfully recorded.
  ///         Return `failure` when there is a duplicate tiling case.
  LogicalResult addKey(TilingKey caseKey);

  /// Get reference to the tiling cases.
  SetVector<TilingKey> getRef() { return cases; }

  /// Index into the tiling cases.
  TilingKey operator[](size_t n) const { return cases[n]; }

  void removeIf(int64_t keepKey) {
    cases.remove_if([&](int64_t key) { return key != keepKey; });
  }

  /// Return the number of tiling case keys.
  size_t size() { return cases.size(); }

private:
  SetVector<TilingKey> cases;
};

//===----------------------------------------------------------------------===//
// TilingStruct
//===----------------------------------------------------------------------===//

using TilingDataPtr = std::unique_ptr<TilingData>;

/// Tiling Struct is a series of Tiling Data. The order of tiling data
/// corresponds to the order of values returned by the host tiling function.
struct TilingStruct {
  using iterator = SmallVectorImpl<TilingDataPtr>::iterator;

public:
  TilingStruct() = default;

  /// Initialize tiling struct with `size` amount of empty tiling data.
  explicit TilingStruct(size_t size);

  iterator begin() { return data_.begin(); }
  iterator end() { return data_.end(); }

  /// Get the number of tiling data in the struct.
  size_t size() const { return data_.size(); }

  /// Push back tiling data.
  void push_back(TilingData &&tilingData);

  /// Access tiling data at position `index`.
  TilingDataPtr &operator[](size_t index);
  const TilingDataPtr &operator[](size_t index) const;

private:
  SmallVector<TilingDataPtr> data_;
};

//===----------------------------------------------------------------------===//
// TilingComputeFn
//===----------------------------------------------------------------------===//

using TilingFnResultTy = FailureOr<std::pair<TilingCases, TilingStruct>>;
/// Tiling computation function is a lambda function that computes the tiling
/// data using information from kernel information.
/// It returns `TilingCases` and `TilingStruct`.
using TilingComputeFn =
    std::function<TilingFnResultTy(KernelInfo *, StmtExprBuilder *)>;

//===----------------------------------------------------------------------===//
// TilingInfo
//===----------------------------------------------------------------------===//

/// Data structure for holding tiling information.
class TilingInfo {
public:
  using tiling_data_iterator = SmallVectorImpl<TilingDataPtr>::iterator;

  TilingInfo() = default;
  virtual ~TilingInfo() = default;
  TilingInfo &operator=(TilingInfo const &) = delete;

  explicit TilingInfo(size_t tilingSize) : struct_(TilingStruct(tilingSize)) {}

  TilingInfo(TilingInfo &&other) {
    this->struct_ = std::move(other.struct_);
    this->caseKeys_ = std::move(other.caseKeys_);
    this->hostTilingFunc_ = std::move(other.hostTilingFunc_);
    this->tilingComputeFn_ = std::move(other.tilingComputeFn_);
    this->tilingKey2Kernel_ = std::move(other.tilingKey2Kernel_);
  }

  tiling_data_iterator tilingDataBegin() { return struct_.begin(); }
  tiling_data_iterator tilingDataEnd() { return struct_.end(); }

  /// Return whether tiling struct is empty.
  bool empty() const { return size() == 0; }

  /// Get the number of tiling data.
  size_t size() const { return struct_.size(); }

  /// Get pointers to all tiling data.
  SmallVector<TilingData *> getTilingStruct();

  /// Get pointer to tiling key.
  TilingData *getTilingKey() const;

  /// Get pointer to the `idx`-th tiling data.
  TilingData *getTilingData(unsigned idx) const;

  /// Get all tiling cases.
  TilingCases getTilingCases() { return caseKeys_; }

  /// Getter and setter to host tiling function.
  func::FuncOp getHostTilingFunc() { return hostTilingFunc_; }
  void setHostTilingFunc(func::FuncOp f) { hostTilingFunc_ = f; }

  /// Get the `idx`-th function argument of host tiling func.
  BlockArgument getHostTilingFuncArg(size_t idx);

  /// Return whether all tiling data are static.
  bool isTilingFullyStatic();

  /// Remove all tiling keys from tiling cases except for `keepKey`.
  void pruneTilingExcept(int64_t keepKey);

  /// Try to simply host tiling function.
  LogicalResult trySimplifyTilingFunc();

  /// Evaluate tiling computation function to generate IR values.
  FailureOr<SmallVector<Value>>
  evaluateTilingComputation(TilingComputeFn fn, KernelInfo *kernelInfo,
                            StmtExprBuilder *builder);

  /// Get mapping from tiling key to device kernel function.
  llvm::DenseMap<TilingKey, func::FuncOp> getTilingKey2KernelMap();

  /// Record assocation between tiling key and to-be-scheduled kernel function.
  void recordKernelFunc(TilingKey key, func::FuncOp f);

private:
  /// Reference to tiling computation lambda function.
  TilingComputeFn tilingComputeFn_;
  /// Tiling data.
  TilingStruct struct_{};
  /// Tiling cases.
  TilingCases caseKeys_;
  /// Pointer to host tiling function.
  func::FuncOp hostTilingFunc_{nullptr};
  /// Tiling key to scheduled device kernel.
  llvm::DenseMap<TilingKey, func::FuncOp> tilingKey2Kernel_{};
};

namespace tiling {

/// Construct an array of `Expr`s that represents the accumulated number of
/// elements up to a certain dimension.
///
/// Input:
/// [dim_0, dim_1, ... dim_{N-1}]
///
/// Output:
/// [dim_0,
///  dim_0 * dim_1, ...,
///  dim_0 * ... * dim_{N-2} * dim_{N-1}]
SmallVector<Expr> getAccumulatedDims(SmallVector<Expr> dims);
} // namespace tiling

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_TILINGUTILS_H
