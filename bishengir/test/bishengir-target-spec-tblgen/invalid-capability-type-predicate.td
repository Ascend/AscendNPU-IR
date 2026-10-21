// clang-format off
// RUN: not bishengir-target-spec-tblgen -DEMPTY_PREDICATE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not bishengir-target-spec-tblgen -DWHITESPACE_PREDICATE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not bishengir-target-spec-tblgen -DUNSET_PREDICATE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: not bishengir-target-spec-tblgen -DOP_PREDICATE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SELF
// RUN: not bishengir-target-spec-tblgen -DOPERAND_PREDICATE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SELF
// RUN: not bishengir-target-spec-tblgen -DBUILDER_PREDICATE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SELF
// RUN: not bishengir-target-spec-tblgen -DUNUSED_PREDICATE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=SELF
// clang-format on

include "bishengir/Target/Capability/CapabilityBase.td"

def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def Item : CapabilitySlot<[ElementType]>;
def ItemType : SlotField<Item, ElementType>;
def Key : CapabilityKey<RequestSchema<[Item]>>;

#ifdef EMPTY_PREDICATE
def BadType : Type<CPred<"">, "empty predicate">;
#endif
#ifdef WHITESPACE_PREDICATE
def BadType : Type<CPred<"   ">, "whitespace predicate">;
#endif
#ifdef UNSET_PREDICATE
def BadType : Type<?, "unset predicate">;
#endif
#ifdef OP_PREDICATE
def BadType : Type<CPred<"$_op->getNumResults() == 1">, "op-dependent type">;
#endif
#ifdef OPERAND_PREDICATE
def BadType : Type<CPred<"$0.getType().isF16()">, "operand-dependent type">;
#endif
#ifdef BUILDER_PREDICATE
def BadType
    : Type<CPred<"$_self == $_builder.getF16Type()">, "builder-dependent type">;
#endif
#ifdef UNUSED_PREDICATE
// Every declared signature is validated, even if no row uses it.
def BadType : Type<CPred<"$_op->getNumResults() == 1">, "unused bad type">;
def Unused : TypeSignature<[ItemType], [BadType]>;
#else
def Bad : SupportCase<Key, [TypeSignature<[ItemType], [BadType]>]>;
#endif

// clang-format off
// EMPTY: error: TypeSignature requires a non-empty ODS Type predicate
// SELF: error: TypeSignature predicate may only reference $_self
// clang-format on
