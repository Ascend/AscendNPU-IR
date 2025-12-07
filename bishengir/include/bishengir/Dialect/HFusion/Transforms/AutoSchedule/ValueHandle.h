//===- ValueHandle.h ----------------------------------------- --*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VALUEHANDLE_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VALUEHANDLE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir {
namespace hfusion {

/// Different types of handles.
enum class ValueHandleKind : int32_t {
  // Default type. Unknown type.
  kUnknown = 0,
  //  Temporary handles that are not meant to be reused.
  kRegular,
  // Handles whose corresponding IR value is named.
  kNamed,
  // Handles to function arguments.
  kFuncArg
};

/// Different status that value handles can take.
enum class HandleStatus : int32_t {
  // Default type. Unknown status.
  kUnknown = 0,
  // Handle is still valid and can be used.
  kValid,
  // Handles that is usable once it's matched. Only applicable to
  // named value handles.
  kNeedsRematch,
  // Unusable handles.
  kInvalid
};

//===----------------------------------------------------------------------===//
// ValueHandle definition
//===----------------------------------------------------------------------===//

struct ValueHandle {
  ValueHandle(Value handle, HandleStatus status, ValueHandleKind kind)
      : handle_(handle), status_(status), kind_(kind) {}

  ValueHandle &operator=(ValueHandle const &) = delete;
  virtual ~ValueHandle() = default;

  static bool classof(const ValueHandle *) { return true; }

  /// Get the underlying handle.
  ///
  /// \note Implementation is different for different kind of handles.
  template <class... Args> Value get(Args &&...args) {
    return getImpl(std::forward<Args>(args)...);
  }

  virtual Value getImpl();
  virtual Value getImpl(Value matchTarget, OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Getter and setter methods.
  //===--------------------------------------------------------------------===//

  ValueHandleKind getValueHandleKind() const { return kind_; }
  HandleStatus getStatus() { return status_; }

  void setStatus(HandleStatus s);
  /// Update handle's underling value to `v` and set its status to valid.
  void setHandle(Value v);
  void invalidate();

protected:
  Value handle_;
  HandleStatus status_{HandleStatus::kUnknown};
  ValueHandleKind kind_{ValueHandleKind::kUnknown};
};

struct RegularValueHandle : public ValueHandle {
  RegularValueHandle(Value handle, HandleStatus status)
      : ValueHandle(handle, status, ValueHandleKind::kRegular) {}

  static bool classof(const ValueHandle *T) {
    return T->getValueHandleKind() == ValueHandleKind::kRegular;
  }

  Value getImpl() override;
};

enum class IdentifierType : uint8_t {
  kUnknown = 0,
  /// Identifier is added to operation's attribute.
  /// For example: \c linalg.elemwise_binary { __identifier__ }
  kAttribute,
  /// Identifier is MLIR operation's assembly name:
  /// \c {dialect_namespace}.{operation_name}
  /// For example: \c linalg.broadcast
  kOperation
};

namespace detail {

/// Helper struct to store information about identifiers. Used inside
/// AutoScheduler.
class Identifier {
public:
  explicit Identifier(IdentifierType type) : type(type) {}
  virtual ~Identifier() = default;

  static bool classof(const Identifier *) { return true; }
  IdentifierType getIdentifierKind() const { return type; }
  virtual StringRef getUniqueIdentifier() const = 0;

  IdentifierType type{IdentifierType::kUnknown};
};

class OperationIdentifier : public Identifier {
public:
  OperationIdentifier(StringRef opName)
      : Identifier(IdentifierType::kOperation), opName_(opName) {}

  static bool classof(const Identifier *T) {
    return T->getIdentifierKind() == IdentifierType::kOperation;
  }

  StringRef getUniqueIdentifier() const override;
  StringRef getName() const { return opName_; }

private:
  StringRef opName_;
};

class AttributeIdentifier : public Identifier {
public:
  // Constructor for one required named, unit attribute.
  AttributeIdentifier(StringRef name);
  // Constructor for one required named attribute.
  AttributeIdentifier(StringRef name, Attribute attr);
  // Constructor for an array of required attributes and optional atributes.
  AttributeIdentifier(const DenseMap<StringRef, Attribute> &requiredAttrs,
                      const DenseMap<StringRef, Attribute> &optionalAttrs);

  static bool classof(const Identifier *T) {
    return T->getIdentifierKind() == IdentifierType::kAttribute;
  }

  /// For attribute identifiers that only have a key, the unique identifier is
  /// the key itself.
  /// For those that have an attribute value, the unique identifier is
  /// constructed as "key + attribute's assembly format".
  StringRef getUniqueIdentifier() const override;
  DictionaryAttr getAttrs(OpBuilder &builder, bool required) const;

private:
  DenseMap<StringRef, Attribute> requiredAttrs_;
  DenseMap<StringRef, Attribute> optionalAttrs_;
  std::string uniqueIdentifier_;
};

/// Options for constructing named value handles.
struct NamedValueHandleArgs {
  StringRef name;
  IdentifierType type{IdentifierType::kUnknown};
  /// Whether to annotate the input \c target before recording.
  bool needsAnnotate{true};
  /// Whether to reverse the order of payload objects in \c target.
  bool needsReverse{false};
  /// Whether the \c name is a unique identifier. If true, then we won't
  /// do automatic renaming to avoid duplicate identifiers.
  bool isNameUnique{false};
};
} // namespace detail

struct NamedValueHandle : public ValueHandle {
  NamedValueHandle(Value handle, std::string name, IdentifierType type,
                   HandleStatus status, bool needsReverse)
      : ValueHandle(handle, status, ValueHandleKind::kNamed),
        name_(std::move(name)), type_(type), needsReverse_(needsReverse) {}

  static bool classof(const ValueHandle *T) {
    return T->getValueHandleKind() == ValueHandleKind::kNamed;
  }

  Value getImpl(Value matchTarget, OpBuilder &opBuilder) override;

  std::string getName() const { return this->name_; }

private:
  /// Unique identifier of this ValueHandle.
  std::string name_;
  /// Identifier type.
  IdentifierType type_{IdentifierType::kUnknown};
  /// Whether we need to reverse the handles upon matching and re-matching.
  bool needsReverse_{false};
};

struct FuncArgHandle : public ValueHandle {
  FuncArgHandle(Value handle, size_t funcArgNum, HandleStatus status)
      : ValueHandle(handle, status, ValueHandleKind::kFuncArg),
        funcArgNum_(funcArgNum) {}

  static bool classof(const ValueHandle *T) {
    return T->getValueHandleKind() == ValueHandleKind::kFuncArg;
  }

  Value getImpl(Value funcValue, OpBuilder &opBuilder) override;

private:
  // TODO: vector of fun arg num
  size_t funcArgNum_;
};

using ValueHandles = SmallVector<ValueHandle *>;

/// This class represents either a ValueHandle or a constantize attribute.
class ValueHandleFoldResult : public PointerUnion<ValueHandle *, Attribute> {
  using PointerUnion<ValueHandle *, Attribute>::PointerUnion;

public:
  explicit ValueHandleFoldResult(int64_t cst, MLIRContext *ctx);

  std::optional<int64_t> getConstInteger();
  std::optional<ValueHandle *> getValueHandle();
};

using ValueHandleFoldResults = SmallVector<ValueHandleFoldResult>;

//===----------------------------------------------------------------------===//
// HandleRecord definition
//===----------------------------------------------------------------------===//

class HandleRecord {
public:
  HandleRecord() = default;
  HandleRecord(HandleRecord &&O) noexcept
      : allocatedHandles_(std::move(O.allocatedHandles_)),
        attributeCount_(std::move(O.attributeCount_)),
        attribute2ValueHandle_(std::move(O.attribute2ValueHandle_)) {}
  // The move assignment operator is defined as deleted pending further
  // motivation.
  HandleRecord &operator=(HandleRecord &&) = delete;

  // The copy constructor and copy assignment operator is defined as deleted
  // pending further motivation.
  HandleRecord(const HandleRecord &) = delete;
  HandleRecord &operator=(const HandleRecord &) = delete;

  ~HandleRecord() { clear(); }

  /// Record handle.
  template <class T> T *record(T &&handle) {
    static_assert(std::is_base_of<ValueHandle, T>::value,
                  "must be ValueHandle");
    allocatedHandles_.push_back(std::move(std::make_unique<T>(handle)));
    return (T *)allocatedHandles_.back().get();
  }

  /// Fetch handles if possible by their identifier.
  std::optional<NamedValueHandle *>
  tryFetchRecordImpl(const detail::Identifier &identifier);

  /// Construct a new attribute name from `oldAttrName` if it's already present
  /// in the handle record and add it to the record.
  std::string getAndRecordAttrName(StringRef oldAttrName);

  /// Reset all recorded handles.
  /// \note Different value handle kind have different implementation.
  void resetAllHandles();

  /// Clear all data structures.
  void clear();

private:
  std::vector<std::unique_ptr<ValueHandle>> allocatedHandles_{};
  llvm::StringMap<size_t> attributeCount_{};
  llvm::StringMap<NamedValueHandle *> attribute2ValueHandle_{};
};

/// Set the status of all value handles.
void setStatusTo(ValueHandles &vhs, HandleStatus status);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOSCHEDULE_VALUEHANDLE_H
