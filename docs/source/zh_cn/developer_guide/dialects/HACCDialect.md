# hacc方言

异构异步计算调用（HACC）方言。

## 属性

### BlockDimAttr

**语法：** `#hacc.block_dim`

**功能：** HACC函数块维度属性。

### CachedIOAttr

**语法：** `#hacc.cached_io`

**功能：** 标记当前值为缓存IO。

### DummyFuncAttr

**语法：** `#hacc.dummy_func`

**功能：** HACC虚拟函数标识属性。

### ExportAsDAGAttr

**语法：** `#hacc.export_as_dag`

**功能：** 将当前函数导出为计算DAG。

### ExternalFunctionPathAttr

**语法：** `#hacc.external_function_path`

**功能：** 标注外部函数文件路径。

### HACCFuncTypeAttr

**语法：**

```mlir
#hacc.function_kind<
  ::mlir::hacc::HACCFuncType   # function_kind
>
```

**功能：** 标识HACC函数类型。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| function_kind | `::mlir::hacc::HACCFuncType` | 枚举类型HACCFuncType |

### GetTilingStructSizeFunctionAttr

**语法：**

```mlir
#hacc.get_tiling_struct_size_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧接口函数，用于获取设备函数所需Tiling结构体占用大小。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### HostFuncTypeAttr

**语法：**

```mlir
#hacc.host_func_type<
  ::mlir::hacc::HostFuncType   # host_func_type
>
```

**功能：** 标识HACC主机函数细分类型。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| host_func_type | `::mlir::hacc::HostFuncType` | 枚举类型HostFuncType |

### InferOutputShapeFunctionAttr

**语法：**

```mlir
#hacc.infer_output_shape_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧接口函数，用于推导设备函数输出张量形状。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### InferSyncBlockLockInitFunctionAttr

**语法：**

```mlir
#hacc.infer_sync_block_lock_init_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧接口函数，用于计算同步块锁初始值。内核执行前所有锁必须完成初始化，由该函数提供初始数值。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### InferSyncBlockLockNumFunctionAttr

**语法：**

```mlir
#hacc.infer_sync_block_lock_num_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧接口函数，用于计算内核所需同步锁数量。原子操作下所有计算块共享一块`<1xi64>`类型全局内存作为锁资源，该函数统计锁总数。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### InferTaskTypeFunctionAttr

**语法：**

```mlir
#hacc.infer_task_type_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定函数，用于推导任务类型与混合计算比例。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### InferWorkspaceShapeFunctionAttr

**语法：**

```mlir
#hacc.infer_workspace_shape_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧接口函数，用于推导设备函数工作空间尺寸。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 目标函数符号名 |

### InputIdxAttr

**语法：**

```mlir
#hacc.input_idx<
  unsigned   # argIdx
>
```

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| argIdx | `unsigned` | 输入参数索引 |

### KernelArgTypeAttr

**语法：**

```mlir
#hacc.arg_type<
  ::mlir::hacc::KernelArgType   # arg_type
>
```

**功能：** 标识内核参数类型。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| arg_type | `::mlir::hacc::KernelArgType` | KernelArgType枚举值 |

### NoIOAliasAttr

**语法：** `#hacc.no_io_alias`

**功能：** 标记函数输入、输出内存无任何别名重叠。

### OutputIdxAttr

**语法：**

```mlir
#hacc.output_idx<
  unsigned   # argIdx
>
```

**功能：** NPU内核调用约定将输出数据以输入参数形式传入，该属性标记当前参数对应的输出序号。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| argIdx | `unsigned` | 输出对应索引 |

### RenameFuncAttr

**语法：**

```mlir
#hacc.rename_func<
  ::mlir::FlatSymbolRefAttr   # targetName
>
```

**功能：** 指定当前函数的重命名目标名称。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| targetName | `::mlir::FlatSymbolRefAttr` | 函数符号名 |

### TargetAttr

**语法：**

```mlir
#hacc.target<
  StringAttr   # target
>
```

**功能：** 指定编译目标硬件设备。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| target | `StringAttr` | 目标设备标识字符串 |

### TargetDeviceSpecAttr

**语法：**

```mlir
#hacc.target_device_spec<
  ::llvm::ArrayRef<DataLayoutEntryInterface>   # entries
>
```

**功能：** 承载NPU硬件规格配置，单条目描述一类硬件参数，可配置多条硬件属性。

**示例：**

```mlir
#hacc.target_device_spec<
  #dlti.dl_entry<"UB_SIZE", 196608 : i32>>
```

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| entries | `::llvm::ArrayRef<DataLayoutEntryInterface>` | 单个规范条目 |

### TilingFunctionAttr

**语法：**

```mlir
#hacc.tiling_function<
  ::mlir::FlatSymbolRefAttr   # funcName
>
```

**功能：** 绑定主机侧分块调度函数，用于设备函数任务切分。

**参数：**

| 参数名 | C++类型 | 说明 |
| :-------: | :-------: | ----------- |
| funcName | `::mlir::FlatSymbolRefAttr` | 分块函数符号名 |

## 枚举

### DeviceSpec

HACC设备硬件规格枚举

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| AI_CORE_COUNT | `0` | AI_CORE_COUNT |
| CUBE_CORE_COUNT | `1` | CUBE_CORE_COUNT |
| VECTOR_CORE_COUNT | `2` | VECTOR_CORE_COUNT |
| UB_SIZE | `3` | UB_SIZE |
| L1_SIZE | `4` | L1_SIZE |
| L0A_SIZE | `5` | L0A_SIZE |
| L0B_SIZE | `6` | L0B_SIZE |
| L0C_SIZE | `7` | L0C_SIZE |
| UB_ALIGN_SIZE | `8` | UB_ALIGN_SIZE |
| L1_ALIGN_SIZE | `9` | L1_ALIGN_SIZE |
| L0C_ALIGN_SIZE | `10` | L0C_ALIGN_SIZE |

### HACCFuncType

HACC函数大类枚举

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| HOST | `1` | HOST |
| DEVICE | `2` | DEVICE |

### HostFuncType

HACC主机函数细分类型枚举

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| kEntry | `1` | host_entry |
| kTilingFunction | `2` | tiling_function |
| kInferOutputShapeFunction | `3` | infer_output_shape_function |
| kInferWorkspaceShapeFunction | `4` | infer_workspace_shape_function |
| kInferSyncBlockLockNumFunction | `5` | infer_sync_block_lock_num_function |
| kInferSyncBlockLockInitFunction | `6` | infer_sync_block_lock_init_function |
| kGetTilingStructSizeFunction | `7` | get_tiling_struct_size_function |
| kInferTaskTypeFunction | `8` | infer_task_type_function |

### KernelArgType

内核参数类型枚举

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| kFFTSBaseAddr | `0` | ffts_base_address |
| kInput | `1` | input |
| kOutput | `2` | output |
| kInputAndOutput | `3` | input_and_output |
| kWorkspace | `4` | workspace |
| kSyncBlockLock | `5` | sync_block_lock |
| kTilingKey | `6` | tiling_key |
| kTilingData | `7` | tiling_data |
| kTilingStruct | `8` | tiling_struct |
| kMeshArg | `9` | mesh_arg |
| kSanitizerAddr | `10` | sanitizer_addr |

### HACCToLLVMIRTranslateAttr

取值范围：32位无符号整数0、1、2

| 枚举符号 | 数值 | 字符串标识 |
| :----: | :---: | ------ |
| ENTRY | `0` | hacc.entry |
| MIX_ENTRY | `1` | hacc.mix_entry |
| ALWAYS_INLINE | `2` | hacc.always_inline |
