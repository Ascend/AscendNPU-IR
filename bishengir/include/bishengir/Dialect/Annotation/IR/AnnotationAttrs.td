//===- AnnotationAttrs.td - Annotation dialect attributes --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for Annotation dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONATTRS_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "bishengir/Dialect/Annotation/IR/AnnotationBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class Annotation_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::annotation";
}

class Annotation_I32EnumAttr<string mnemonic, Annotation_I32Enum enumInfo>
  : EnumAttr<Annotation_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Annotation_EffectWrite : I32EnumAttrCase<"Write", 0, "write">;
def Annotation_EffectRead : I32EnumAttrCase<"Read", 1, "read">;

def Annotation_EffectModeEnum : Annotation_I32Enum<
  "EffectMode", "Memory Effect Mode for Annotation", [
    Annotation_EffectWrite,
    Annotation_EffectRead,
  ]>;

def Annotation_EffectModeAttr : Annotation_I32EnumAttr<"effect_mode", Annotation_EffectModeEnum> {
  let description = [{
    annotation memory effect mode attribute.
  }];
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONATTRS_TD