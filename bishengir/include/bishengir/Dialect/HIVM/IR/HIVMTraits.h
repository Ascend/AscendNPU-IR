//===- HIVMTraits.h - HIVM dialect trait definitions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hivm {
/// Forward declarations.
enum class TCoreType : uint32_t;
enum class PIPE : uint32_t;
enum class IteratorType : uint32_t;

namespace detail {
/// Return whether the op is an elemwise n-ary op.
/// An HIVM operation is elemwise nary op if it has
/// \c OpTrait::ElementwiseNaryOpTrait<N> trait.
bool isElemwiseNaryOpImpl(Operation *op);

/// Return the iterator types of HIVM Elemwise Ops.
SmallVector<IteratorType> getIteratorTypesArrayForElemwiseOp(Operation *op);

/// Return the permutation loop dims in vtranspose,
/// and transpose attr otherwise
ArrayRef<int64_t> getPermutationArray(Operation *op);

/// Return the broadcast loop dims in vBrc,
/// and broadcast attr otherwise
ArrayRef<int64_t> getBroadcastArray(Operation *op);

/// Returns the list of inline-broadcasted axes for the operand.
SmallVector<int64_t> getInlinedBroadcastableAxes(const Operation *op,
                                                 const OpOperand *opOperand);

/// Return the iterator types of HIVM Elemwise Ops.
LogicalResult
setIteratorTypesArrayForElemwiseOp(Operation *op,
                                   const IteratorType &iteratorType,
                                   const DenseI64ArrayAttr &arrayAttr);
} // namespace detail
} // namespace hivm

namespace OpTrait {
namespace impl {
LogicalResult verifyElementwiseNaryOpTrait(Operation *op, int numOperands);
LogicalResult verifyHIVMOpSameOperandsAndResultRank(Operation *op);
LogicalResult verifyBroadcastableOTF(Operation *op);
LogicalResult verifyTransposableOTF(Operation *op);
LogicalResult verifyVectorOnlyTrait(Operation *op, int idx);
LogicalResult verifyScalarOnlyTrait(Operation *op, int idx);
} // namespace impl

//===----------------------------------------------------------------------===//
// Operation Traits
//===----------------------------------------------------------------------===//

/// This class provides the API for HIVM Elemwise Vector Operations.
template <int N> class ElementwiseNaryOpTrait {
public:
  static_assert(N >= 1, "N-ary Ops should have at least one operand");
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, ElementwiseNaryOpTrait<N>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      return impl::verifyElementwiseNaryOpTrait(op, N);
    }
  };
};

/// This class provides the API for HIVM Operations that have library
/// functions that support operands with a static, predefined maximum rank.
template <int MaxRank> class OpLibraryMaxRankTrait {
public:
  template <typename ConcreteType>
  class Impl
      : public TraitBase<ConcreteType, OpLibraryMaxRankTrait<MaxRank>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) { return success(); }
    static int getOpLibraryMaxRankImpl() { return MaxRank; }
  };
};

/// This class provides the API for HIVM Operations that have library
/// functions that support operands with a dynamic maximum rank.
class InferMaxRankTrait : public OpLibraryMaxRankTrait<0> {};

/// This class provides the API for HIVM Operations that have library
/// functions that have no restrictions on the operand rank.
class NoMaxRankTrait : public OpLibraryMaxRankTrait<-1> {};

/// This class provides the API for HIVM Operations that doesn't have a library
/// function implemented.
template <typename ConcreteType>
class NoLibraryFunctionTrait
    : public TraitBase<ConcreteType, NoLibraryFunctionTrait> {
public:
  static int getOpLibraryMaxRankImpl() {
    llvm_unreachable("This op has no library function.");
  }
  static std::string getOpLibraryCallName(std::optional<bool> isOpsAligned) {
    llvm_unreachable("This op has no library function.");
  }
};

template <typename ConcreteType>
class BroadcastableOTF : public TraitBase<ConcreteType, BroadcastableOTF> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyBroadcastableOTF(op);
  }
};

template <typename ConcreteType>
class TransposableOTF : public TraitBase<ConcreteType, TransposableOTF> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyTransposableOTF(op);
  }
};

template <typename ConcreteType>
class SinglePipeOpTrait : public TraitBase<ConcreteType, SinglePipeOpTrait> {};

template <hivm::PIPE Pipe> class OpPipeTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, OpPipeTrait<Pipe>::Impl> {
  public:
    static hivm::PIPE getPipe() { return Pipe; }
  };
};

template <typename ConcreteType>
class MacroOpTrait : public TraitBase<ConcreteType, MacroOpTrait> {};

template <hivm::PIPE InPipe, hivm::PIPE OutPipe> class MacroOpPipeTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType,
                                MacroOpPipeTrait<InPipe, OutPipe>::Impl> {
  public:
    static hivm::PIPE getInPipe() { return InPipe; }
    static hivm::PIPE getOutPipe() { return OutPipe; }
  };
};

template <hivm::TCoreType CoreType> class CoreTypeTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, CoreTypeTrait<CoreType>::Impl> {};
};

template <typename ConcreteType>
class CommutativeOpTrait : public TraitBase<ConcreteType, CommutativeOpTrait> {
};

//===----------------------------------------------------------------------===//
// Operand/Result Traits
//===----------------------------------------------------------------------===//

template <typename ConcreteType>
class HIVMOpSameOperandsAndResultRank
    : public TraitBase<ConcreteType, HIVMOpSameOperandsAndResultRank> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyHIVMOpSameOperandsAndResultRank(op);
  }
};

template <int idx> class ScalarOnlyTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, ScalarOnlyTrait<idx>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      return impl::verifyScalarOnlyTrait(op, idx);
    }
  };
};

template <int idx> class VectorOnlyTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, VectorOnlyTrait<idx>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      return impl::verifyVectorOnlyTrait(op, idx);
    }
  };
};

/// @see: HIVMTraits.td
template <typename ConcreteType>
class UniformReassociationFlattenTrait
    : public TraitBase<ConcreteType, UniformReassociationFlattenTrait> {};

/// @see: HIVMTraits.td
template <typename ConcreteType>
class CollapsibleConsecutiveTargetDimsTrait
    : public TraitBase<ConcreteType, CollapsibleConsecutiveTargetDimsTrait> {};
} // namespace OpTrait
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMTRAITS_H
