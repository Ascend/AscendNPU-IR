// clang-format off
// RUN: not bishengir-target-spec-tblgen -DMISSING -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not bishengir-target-spec-tblgen -DDUP -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=DUP
// RUN: not bishengir-target-spec-tblgen -DUNKNOWN -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not bishengir-target-spec-tblgen -DKIND -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=KIND
// RUN: not bishengir-target-spec-tblgen -DDOMAIN -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=DOMAIN
// RUN: not bishengir-target-spec-tblgen -DPROJECTION -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PROJECTION
// RUN: not bishengir-target-spec-tblgen -DOPTIONALATTR -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=OPTIONALATTR
// RUN: not bishengir-target-spec-tblgen -DOPTIONALOPER -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=OPTIONALOPER
// RUN: not bishengir-target-spec-tblgen -DDUPBINDING -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=DUPBINDING
// clang-format on

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "bishengir/Target/Capability/CapabilityBase.td"

def BindDialect : Dialect {
  let name = "capbind";
  let cppNamespace = "::mlir::capbind";
}

def BindModeA : I32EnumAttrCase<"A", 0>;
def BindModeB : I32EnumAttrCase<"B", 1>;
def BindModeEnum : I32EnumAttr<"BindMode", "bind mode", [BindModeA, BindModeB]> {
  let cppNamespace = "::mlir::capbind";
}
def BindModeAttr : EnumAttr<BindDialect, BindModeEnum, "bind_mode">;

def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def Src : CapabilitySlot<[ElementType]>;
def Dst : CapabilitySlot<[ElementType]>;
def SrcType : SlotField<Src, ElementType>;
def DstType : SlotField<Dst, ElementType>;
def Mode : EnumAttrCapabilityParameter<BindModeAttr>;
def Flag : BoolAttrCapabilityParameter<>;
def Schema : RequestSchema<[Src, Dst], [Mode, Flag]>;
def Payload : CapabilityKey<Schema>;

def BindOp : Op<BindDialect, "payload", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyRankedTensor:$src,
                       AnyMemRef:$dst,
                       Optional<AnyRankedTensor>:$optional_src,
                       DefaultValuedAttr<BindModeAttr, "BindMode::A">:$mode,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$flag,
                       OptionalAttr<BoolAttr>:$optional_flag);
  let results = (outs AnyRankedTensor:$result);
}

#ifdef MISSING
// MISSING: capability binding: 'Payload' does not cover 'flag'
def MissingBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>
]>;
#endif

#ifdef DUP
// DUP: capability binding: 'Payload' covers 'mode' more than once
def DuplicateFieldBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef UNKNOWN
// UNKNOWN: capability binding: no operand named 'missing' on 'BindOp'
def UnknownSourceBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"missing">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef KIND
// KIND: capability binding: 'mode' on 'BindOp' is an attribute, not an operand
def WrongKindBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<OperandBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef DOMAIN
// DOMAIN: capability binding: attribute 'flag' does not match the domain of 'mode'
def DomainBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Flag>
]>;
#endif

#ifdef PROJECTION
// PROJECTION: capability binding: element_type projection requires a type-domain field
def ProjectionBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef OPTIONALATTR
// OPTIONALATTR: capability binding: optional attribute 'optional_flag' has no ODS default
def OptionalAttrBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"optional_flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef OPTIONALOPER
// OPTIONALOPER: capability binding: optional operand 'optional_src' cannot feed a required field
def OptionalOperandBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"optional_src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif

#ifdef DUPBINDING
// DUPBINDING: capability binding: duplicate binding for 'Payload'
def FirstBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
def SecondBinding : CapabilityBinding<Payload, BindOp, [
  CapabilityExtractor<OperandBindingSource<"src">, ElementTypeProjection, SrcType>,
  CapabilityExtractor<OperandBindingSource<"dst">, ElementTypeProjection, DstType>,
  CapabilityExtractor<AttributeBindingSource<"mode">, IdentityProjection, Mode>,
  CapabilityExtractor<AttributeBindingSource<"flag">, IdentityProjection, Flag>
]>;
#endif
