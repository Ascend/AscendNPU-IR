# memref_ext方言

扩展memref方言，提供内存相关扩展操作。

## 操作定义

### memref_ext.alloc_workspace (::bishengir::memref_ext::AllocWorkspaceOp)

**功能：** 工作空间内存分配操作，用于申请一块工作空间内存区域。

**语法：**

```mlir
operation ::= `memref_ext.alloc_workspace` `(`$dynamicSize`)` (`from` $workspaceArg^)? ( `offset` `=` `[`$offset^`]`)?
              attr-dict `:` (`from` type($workspaceArg)^ `to`)? type($memref)
```

**示例：**

```mlir
memref_ext.alloc_workspace() : memref<100xi8>
memref_ext.alloc_workspace(%dynamic) : memref<2x?xi32>
memref_ext.alloc_workspace(%dynamic) from %arg offset = [%offset] : from memref<?xi8> to memref<2x?xi32>
```

**特性：** `AttrSizedOperandSegments`

**操作数：**

| 操作数 | 说明 |
| :-----: | ----------- |
| `workspaceArg` | 任意类型内存视图 |
| `dynamicSize` | 变长index动态尺寸 |
| `offset` | 变长index偏移量 |

**结果：**

| 结果 | 说明 |
| :----: | ----------- |
| `memref` | 分配完成的任意类型内存视图 |

## 枚举

### CmpFPredicate

**取值范围：** 64位无符号整数0~15

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| AlwaysFalse | `0` | false |
| OEQ | `1` | oeq |
| OGT | `2` | ogt |
| OGE | `3` | oge |
| OLT | `4` | olt |
| OLE | `5` | ole |
| ONE | `6` | one |
| ORD | `7` | ord |
| UEQ | `8` | ueq |
| UGT | `9` | ugt |
| UGE | `10` | uge |
| ULT | `11` | ult |
| ULE | `12` | ule |
| UNE | `13` | une |
| UNO | `14` | uno |
| AlwaysTrue | `15` | true |

### CmpIPredicate

**取值范围：** 64位无符号整数0~9

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| eq | `0` | eq |
| ne | `1` | ne |
| slt | `2` | slt |
| sle | `3` | sle |
| sgt | `4` | sgt |
| sge | `5` | sge |
| ult | `6` | ult |
| ule | `7` | ule |
| ugt | `8` | ugt |
| uge | `9` | uge |

### IntegerOverflowFlags

**功能：** 整数溢出算术标识

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| none | `0` | none |
| nsw | `1` | nsw |
| nuw | `2` | nuw |

### RoundingMode

**功能：** 浮点舍入模式

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| to_nearest_even | `0` | to_nearest_even |
| downward | `1` | downward |
| upward | `2` | upward |
| toward_zero | `3` | toward_zero |
| to_nearest_away | `4` | to_nearest_away |

### AtomicRMWKind

**取值范围：** 64位无符号整数0~14

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| addf | `0` | addf |
| addi | `1` | addi |
| assign | `2` | assign |
| maximumf | `3` | maximumf |
| maxs | `4` | maxs |
| maxu | `5` | maxu |
| minimumf | `6` | minimumf |
| mins | `7` | mins |
| minu | `8` | minu |
| mulf | `9` | mulf |
| muli | `10` | muli |
| ori | `11` | ori |
| andi | `12` | andi |
| maxnumf | `13` | maxnumf |
| minnumf | `14` | minnumf |

### FastMathFlags

**功能：** 浮点快速运算标识

| 枚举符号 | 数值 | 标识字符串 |
| :----: | :---: | ------ |
| none | `0` | none |
| reassoc | `1` | reassoc |
| nnan | `2` | nnan |
| ninf | `4` | ninf |
| nsz | `8` | nsz |
| arcp | `16` | arcp |
| contract | `32` | contract |
| afn | `64` | afn |
| fast | `127` | fast |
