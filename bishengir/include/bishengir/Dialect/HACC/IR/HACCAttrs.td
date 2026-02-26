//===- HACCAttrs.td - HACC attributes definitions ----------*- tablegen -*-===//
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
//
// Defines MLIR HACC dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD

include "bishengir/Dialect/HACC/IR/HACCBase.td"
include "bishengir/Dialect/HACC/IR/HACCInterfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/DLTI/DLTIAttrs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

//===----------------------------------------------------------------------===//
// HACC Enums and Attributes
//===----------------------------------------------------------------------===//

class HACC_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hacc";
}

class HACC_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HACC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Function Type
//===----------------------------------------------------------------------===//

def HACC_HOST : I32EnumAttrCase<"HOST", 1>;
def HACC_DEVICE : I32EnumAttrCase<"DEVICE", 2>;
def HACC_FuncTypeEnum : HACC_I32Enum<"HACCFuncType", "HACC Function Category", [
  HACC_HOST,
  HACC_DEVICE
]> {}

def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {
  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC function type.
  }];
}

//===----------------------------------------------------------------------===//
// HACC To LLVM Translation Attribute
//===----------------------------------------------------------------------===//

def HACC_ToLLVMIRTranslateAttr : HACC_I32Enum<"HACCToLLVMIRTranslateAttr", "", [
  // For Device Entry Functions
  I32EnumAttrCase<"ENTRY", 0, "hacc.entry">,
  // For Mix Device Entry Functions
  I32EnumAttrCase<"MIX_ENTRY", 1, "hacc.mix_entry">,
  I32EnumAttrCase<"ALWAYS_INLINE", 2, "hacc.always_inline">,
]> {
  let cppNamespace = "::mlir::hacc";
}

//===----------------------------------------------------------------------===//
// Kernel Argument Type
//===----------------------------------------------------------------------===//

def HACC_kFFTSBaseAddr : I32EnumAttrCase<"kFFTSBaseAddr", 0, "ffts_base_address">;
def HACC_kInput : I32EnumAttrCase<"kInput", 1, "input">;
def HACC_kOutput : I32EnumAttrCase<"kOutput", 2, "output">;
def HACC_kInputAndOutput : I32EnumAttrCase<"kInputAndOutput", 3, "input_and_output">;
def HACC_kWorkspace : I32EnumAttrCase<"kWorkspace", 4, "workspace">;
def HACC_kSyncBlockLock : I32EnumAttrCase<"kSyncBlockLock", 5, "sync_block_lock">;
def HACC_kTilingKey : I32EnumAttrCase<"kTilingKey", 6, "tiling_key">;
def HACC_kTilingData : I32EnumAttrCase<"kTilingData", 7, "tiling_data">;
def HACC_kTilingStruct : I32EnumAttrCase<"kTilingStruct", 8, "tiling_struct">;
def HACC_kMeshArg : I32EnumAttrCase<"kMeshArg", 9, "mesh_arg">;
def HACC_kSanitizerAddr : I32EnumAttrCase<"kSanitizerAddr", 10, "sanitizer_addr">;

def HACC_KernelArgTypeEnum : HACC_I32Enum<"KernelArgType", "HACC Kernel Arg Category", [
  HACC_kFFTSBaseAddr,
  HACC_kInput,
  HACC_kOutput,
  HACC_kInputAndOutput,
  HACC_kWorkspace,
  HACC_kSyncBlockLock,
  HACC_kTilingKey,
  HACC_kTilingData,
  HACC_kTilingStruct,
  HACC_kMeshArg,
  HACC_kSanitizerAddr,
]> {}

def HACC_KernelArgTypeAttr : HACC_Attr<"KernelArgType", "arg_type"> {
  let parameters = (ins EnumParameter<HACC_KernelArgTypeEnum>:$arg_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC kernel arg type.
  }];
}

//===----------------------------------------------------------------------===//
// Host Function Type
//===----------------------------------------------------------------------===//

// TODO: Refactor into HACC_ToLLVMIRTranslateAttr
def HACC_kEntry : I32EnumAttrCase<"kEntry", 1, "host_entry">;
def HACC_kTilingFunction : I32EnumAttrCase<"kTilingFunction", 2, "tiling_function">;
def HACC_kInferOutputShapeFunction : I32EnumAttrCase<"kInferOutputShapeFunction", 3, "infer_output_shape_function">;
def HACC_kInferWorkspaceShapeFunction : I32EnumAttrCase<"kInferWorkspaceShapeFunction", 4, "infer_workspace_shape_function">;
def HACC_kInferSyncBlockLockNumFunction : I32EnumAttrCase<"kInferSyncBlockLockNumFunction", 5, "infer_sync_block_lock_num_function">;
def HACC_kInferSyncBlockLockInitFunction : I32EnumAttrCase<"kInferSyncBlockLockInitFunction", 6, "infer_sync_block_lock_init_function">;
def HACC_kGetTilingStructSizeFunction : I32EnumAttrCase<"kGetTilingStructSizeFunction", 7, "get_tiling_struct_size_function">;
def HACC_kInferTaskTypeFunction : I32EnumAttrCase<"kInferTaskTypeFunction", 8, "infer_task_type_function">;
def HACC_HostFuncTypeEnum : HACC_I32Enum<"HostFuncType", "HACC Host function type", [
  HACC_kEntry,
  HACC_kTilingFunction,
  HACC_kInferOutputShapeFunction,
  HACC_kInferWorkspaceShapeFunction,
  HACC_kInferSyncBlockLockNumFunction,
  HACC_kInferSyncBlockLockInitFunction,
  HACC_kGetTilingStructSizeFunction,
  HACC_kInferTaskTypeFunction,
]> {}

def HACC_HostFuncTypeAttr : HACC_Attr<"HostFuncType", "host_func_type"> {
  let parameters = (ins EnumParameter<HACC_HostFuncTypeEnum>:$host_func_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC host func type.
  }];
}

//===----------------------------------------------------------------------===//
// Host and Device Function Binding
//===----------------------------------------------------------------------===//

class HACC_FuncRefAttr<string attrName, string attrMnemonic>
    : HACC_Attr<attrName, attrMnemonic> {
  let parameters = (ins AttrParameter<"::mlir::FlatSymbolRefAttr",
                                      "function symbol name">:$funcName);
  let assemblyFormat = "`<` $funcName `>`";
  let builders = [TypeBuilder<(ins "StringRef":$funcName), [{
      return Base::get($_ctxt,
        ::mlir::FlatSymbolRefAttr::get($_ctxt, std::move(funcName)));
  }]>];
  let extraClassDeclaration = [{
    StringRef getFuncNameStr() {
      return getFuncName().getValue();
    }
  }];
}

def HACC_TilingFunctionAttr
    : HACC_FuncRefAttr<"TilingFunction", "tiling_function"> {
  let description = [{
    HACC tiling function attribute. This is used to indicate the host
    tiling function of the device function.
  }];
}

def HACC_InferOutputShapeFunctionAttr
    : HACC_FuncRefAttr<"InferOutputShapeFunction",
                       "infer_output_shape_function"> {
  let description = [{
    HACC infer output shape function attribute. This is used to indicate the host
    infer output shape function of the device function.
  }];
}

def HACC_InferWorkspaceShapeFunctionAttr
    : HACC_FuncRefAttr<"InferWorkspaceShapeFunction",
                       "infer_workspace_shape_function"> {
  let description = [{
    HACC infer workspace shape function attribute. This is used to indicate the host
    infer workspace shape function of the device function.
  }];
}

def HACC_InferSyncBlockLockNumFunctionAttr
    : HACC_FuncRefAttr<"InferSyncBlockLockNumFunction",
                       "infer_sync_block_lock_num_function"> {
  let description = [{
    HACC infer sync block lock num function attribute. This is used to indicate the host
    infer sync block lock num function of the device function.

    For 1 atomic op, all the blocks share 1 <1xi64> type memref on gm, which is called a lock,
    and this function is used to infer how many locks the kernel needs.
  }];
}

def HACC_InferSyncBlockLockInitFunctionAttr
    : HACC_FuncRefAttr<"InferSyncBlockLockInitFunction",
                       "infer_sync_block_lock_init_function"> {
  let description = [{
    HACC infer sync block lock init function attribute. This is used to indicate the host
    get sync block lock init value for the device function.

    Every lock needs to be initialized before running the kernel,
    this function is used to infer the initial value of the lock.
  }];
}

def HACC_GetTilingStructSizeFunctionAttr
    : HACC_FuncRefAttr<"GetTilingStructSizeFunction",
                       "get_tiling_struct_size_function"> {
  let description = [{
    HACC get tiling struct size function attribute. This is used to indicate the host
    get tiling struct size function of the device function.
  }];
}

def HACC_InferTaskTypeFunctionAttr
    : HACC_FuncRefAttr<"InferTaskTypeFunction",
                       "infer_task_type_function"> {
  let description = [{
    HACC infer core type and mix ratio.
  }];
}

//===----------------------------------------------------------------------===//
// Device Target and Specification
//===----------------------------------------------------------------------===//

def HACC_TargetAttr : HACC_Attr<"Target", "target"> {
  let description = [{
    This attribute is used to indicate the target device.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "target device">:$target
  );
  let assemblyFormat = "`<` $target `>`";
}

def HACC_DeviceSpecEnum :
  HACC_I32Enum<"DeviceSpec", "HACC device spec", [
  I32EnumAttrCase<"AI_CORE_COUNT", 0>,
  I32EnumAttrCase<"CUBE_CORE_COUNT", 1>,
  I32EnumAttrCase<"VECTOR_CORE_COUNT", 2>,
  I32EnumAttrCase<"UB_SIZE", 3>,
  I32EnumAttrCase<"L1_SIZE", 4>,
  I32EnumAttrCase<"L0A_SIZE", 5>,
  I32EnumAttrCase<"L0B_SIZE", 6>,
  I32EnumAttrCase<"L0C_SIZE", 7>,
  I32EnumAttrCase<"UB_ALIGN_SIZE", 8>,
  I32EnumAttrCase<"L1_ALIGN_SIZE", 9>,
  I32EnumAttrCase<"L0C_ALIGN_SIZE", 10>
]> {
  let description = [{
    HACC `DeviceSpec` enum class defines the list of hardware specification
    that might be useful to compiler.
  }];
}

def HACCTargetDeviceSpecTrait
    : NativeAttrTrait<"HACCTargetDeviceSpecInterface::Trait"> {
  let cppNamespace = "::mlir::hacc";
}

def HACC_TargetDeviceSpecAttr :
    HACC_Attr<"TargetDeviceSpec", "target_device_spec",
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
              [TargetDeviceSpecTrait, HACCTargetDeviceSpecTrait]> {
#else
              [DeclareAttrInterfaceMethods<TargetDeviceSpecInterface>, DeclareAttrInterfaceMethods<DLTIQueryInterface>, HACCTargetDeviceSpecInterface]> {
#endif
  let summary = "An attribute to represent NPU target device specification.";
  let description = [{
    Each device specification describes a single device and its
    hardware properties. Each device specification can contain any number
    of optional hardware properties (e.g., "UB_SIZE" below).

    Example:
    ```
    #hacc.target_device_spec<
      #dlti.dl_entry<"UB_SIZE", 196608 : i32>>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "single spec entry">:$entries
  );
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";
}

//===----------------------------------------------------------------------===//
// Misc.
//===----------------------------------------------------------------------===//

def HACC_ExportAsDAGAttr : HACC_Attr<"ExportAsDAG", "export_as_dag"> {
  let description = [{
    Export function as DAG.
  }];
}

def HACC_DummyFuncAttr : HACC_Attr<"DummyFunc", "dummy_func"> {
  let description = [{
    HACC dummy func type.
  }];
}

def HACC_RenameFuncAttr : HACC_Attr<"RenameFunc", "rename_func"> {
  let parameters = (ins AttrParameter<"::mlir::FlatSymbolRefAttr",
                                      "function symbol name">:$targetName);
  let assemblyFormat = "`<` $targetName `>`";
  let description = [{
    This attribute is used to indicate that current function shall be
    renamed to the target function name.
  }];
}

def HACC_InputIdxAttr : HACC_Attr<"InputIdx", "input_idx"> {
  let parameters = (ins "unsigned":$argIdx);
  let assemblyFormat = "`<` params `>`";
}

def HACC_OutputIdxAttr : HACC_Attr<"OutputIdx", "output_idx"> {
  let parameters = (ins "unsigned":$argIdx);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    On NPU device, the kernel function's calling convention is that the output
    is passed in as a input parameter.

    This attribute is used to indicate which output value does the current
    function argument corresponds to.
  }];
}

def HACC_ExternalFunctionPathAttr
    : HACC_Attr<"ExternalFunctionPath", "external_function_path"> {
  let description = [{
    This attribute is used to indicate the location of external functions.
  }];
}

def HACC_CachedIOAttr
    : HACC_Attr<"CachedIO", "cached_io"> {
  let description = [{
    This attribute is used to indicate the value has been cached io.
  }];
}

def HACC_NoIOAliasAttr
    : HACC_Attr<"NoIOAlias", "no_io_alias"> {
  let description = [{
    This attribute is used to indicate the function inputs/outputs are
    strictly not alias.
  }];
}

def HACC_BlockDimAttr : HACC_Attr<"BlockDim", "block_dim"> {
  let description = [{
    HACC block dimension attribute for function.
  }];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
