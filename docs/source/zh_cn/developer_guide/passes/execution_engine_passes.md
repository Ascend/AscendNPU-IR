# ExecutionEngine Passes

## -convert-hivm-to-hfusion

**功能**：将HIVM向量操作转换回HFusion、上游方言。

将HIVM操作的向量侧转换回上游方言（主要是Linalg和HFusion）。

所做的修改：

- 跳过Host与AIC函数：Cube侧操作保持不变。
- 保持`hivm.hir.load`操作不转换。
- 对每个函数以贪心方式重写其余的向量操作。

约束：

- 操作必须具有tensor或buffer语义。

**选项**：

- `-convert-to-named-op`：将hir操作转换为命名操作。

## -execution-engine-convert-hfusion-to-upstream

**功能**：将HFusion操作转换为上游方言的等价操作，供执行引擎使用。

将选定的HFusion操作（如HistogramOp、IsFiniteOp）转换为上游方言，以支持CPU执行与精度调试。

所做的修改：

- 将复杂的HFusion操作分解为标准的Linalg、SCF、Arith、Math和Tensor操作。
- 支持tensor形状折叠、动态形状解析与mask语义。

约束与假设：

- 目前要求操作具有纯tensor语义。

## -execution-engine-convert-hivm-to-upstream

**功能**：将HIVM操作转换为上游方言的等价操作。

主要使用Linalg将选定的HIVM操作转换为上游方言。主要目标是使用可通过上游Pass（而非内部Pass）下降到LLVM IR的方言。

所做的修改：

- 从类型中移除HIVM属性（如内存空间）。
- 将HIVM操作转换为其上游方言的等价操作。

约束与假设：

- 操作必须具有tensor或buffer语义。
- 同步操作视为NoOp。
- 不支持PointerCastOp。

**选项**：

- `-convert-to-named-op`：将hir操作转换为命名操作。

## -execution-engine-create-host-main

**功能**：用main函数包装唯一的Host入口函数。

用另一个函数包装唯一的Host入口函数，以便通过不同的runner工具函数测试kernel函数。

所做的修改：

- 创建包装函数来初始化函数的输入，并将输入与生成的输出打印到文件。
- 为包装过程中使用的函数创建声明。

约束与假设：

- 必须恰好存在一个带有`hacc.function_kind = #hacc.function_kind<HOST>`和`hacc.host_func_type = #hacc.host_func_type<host_entry>`属性的函数。
- 该唯一的Host入口函数不得以包装函数的名字命名。
- 该函数的参数与返回结果必须是tensor或memref。

输入

```mlir
 func.func @kernel(...) -> (...) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
   // do something
   %result = ...
   return %result
 }
 ```

输出

```mlir
 func.func @kernel(...)
 func.func @main(...) {
   // prepare arguments
   %input = ...
   // invoke kernel
   %result = call @kernel(%input, ...)
   // dump output
 }
```

**选项**：

- `-wrapper-name`：为唯一的Host入口函数生成的包装函数名称。
