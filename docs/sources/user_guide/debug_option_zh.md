# 调试调测

## 调试：DEBUG OP类

有多种生态编程语言对接AscendNPU IR，当前仅以Triton为例进行介绍，剩余还有TileLang/FlagTree/DLCompiler/TLE 方式类试可以参考Triton进行对接

目前与调试调测相关的triton op主要有如下四类：

* **​static_assert：​**编译时静态断言
* **​static_print：​**编译时静态打印
* **​device_assert：​**运行时设备断言
* **​device_print：​**运行时设备打印

### static_assert

#### 接口描述

```
# condition: bool - 编译时可计算的布尔表达式
# message: str - 可选，断言失败时显示的消息
triton.language.static_assert(condition: bool, message: str = "") -> None
```

#### 使用示例

```
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_tensor,
    output_tensor,
    BLOCK_SIZE: tl.constexpr
):
    tl.static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive")
```

### static_print

#### 接口描述

```
# message: str - 要打印的消息，可以包含编译时常量
triton.language.static_print(message: str) -> None
```

#### 使用示例

```
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_tensor,
    output_tensor,
    BLOCK_SIZE: tl.constexpr
):
    tl.static_print(f"  BLOCK_SIZE = {BLOCK_SIZE}")
```

### device_assert

注：使用此功能前需要设置环境变量export TRITON_DEBUG=1 export TRITON_DEVICE_PRINT=1

#### 接口描述

```
# condition: bool - 要断言的条件, 必须是一个布尔张量
# message: str - 可选，断言失败时显示的消息

# triton language 接口
triton.language.device_assert(condition: bool, message: str = "") -> None
```

#### 使用示例

```
import triton
import triton.language as tl

@triton.jit
def kernel_name(x_ptr, y):
    x_ptrs = x_ptr + tl.arange(0, 8)
    x = tl.load(x_ptrs)
    tl.device_assert(x > 0, "x must be positive")
```

#### 断言效果：

![](figs/P1.png)

### device_print

注：使用此功能前需要设置环境变量export TRITON_DEVICE_PRINT=1

#### 接口描述

```
# prefix: str - 打印在值之前的前缀，必须是字符串
# *args - 要打印的值可以是任何张量或标量
# hex: bool - 是否将所有值以十六进制而非十进制形式打印

# triton language 接口
triton.language.device_print(prefix, *args, hex=False) -> None
```

#### 使用示例

```
import triton
import triton.language as tl

@triton.jit
def kernel_name(x_ptr, y):
    x_ptrs = x_ptr + tl.arange(0, 8)
    x = tl.load(x_ptrs)
    tl.device_print("x", x)
    tl.device_print("y and 16", y, 16, hex=True)
```

#### 打印效果：

![](figs/P2.png)

## 调试：工具类

### mssanitizer

命令行异常检测工具用于triton算子内存检测/竞争检测/未初始化检测等

注：使用此功能前需要设置环境变量 export TRITON_ENABLE_SANITIZER=true

#### 使用方式

```
# 直接拉起triton算子运行即可
mssanitizer python test.py
```

#### 效果展示

![](figs/P3.png)

### msprof

命令行模型调优工具用于triton算子性能数据的采集和解析

#### 使用方式

```
# 整网上板调优
# --output - 收集到的性能数据的存放路径，默认在当前目录下保存性能数据
# --application - 整网执行命令
msprof --output=xxx --application=""

# 单算子上板调优
# --output - 收集到的性能数据的存放路径，默认在当前目录下保存性能数据
# --application - 单算子执行命令
# --kernel-name - 指定要采集的算子名称，支持使用算子名前缀进行模糊匹配
# --aic-metrics - 使能算子性能指标的采集能力和算子采集能力指标（Roofline/Occupancy/MemoryDetail等）
msprof op --output=xxx --application="" --kernel-name=xxx --aic-metrics=xxx

# 单算子仿真调优
# --core-id - 指定部分逻辑核的id，解析部分核的仿真数据
# --kernel-name - 指定要采集的算子名称，支持使用算子名前缀进行模糊匹配
# --soc-version - 指定仿真器类型
# --output - 收集到的性能数据的存放路径，默认在当前目录下保存性能数据
msprof op simulator --core-id=xxx --kernel-name=xxx --soc-version=Ascendxxx --output=xxx
```

#### AscendNPU IR编译选项

#### 常用性能分析图

性能流水图数据可在以下文件中获取

trace.json：支持在chrome://tracing/上生成指令流水图
 ![](figs/P4.png)

visualize_data.bin：支持在Mind Studio Insight可视化呈现指令在昇腾AI处理器上的运行情况
![](figs/P5.png)

#### 其他性能分析图

详见[Mindstudio算子开发工具](https://www.hiascend.com/document/detail/zh/mindstudio/830/ODtools/Operatordevelopmenttools/atlasopdev_16_0136.html)

### AscendNPU IR编译选项

#### enable-sanitizer

开启TRITON_ENABLE_SANITIZER环境变量后enable-sanitizer置为true，将debug 信息（行号等）以及 metadata（桩函数名称等）传给毕昇编译器，毕昇编译器根据这些信息添加桩函数调用 (CALL) 完成插桩，而桩函数的实现在 mssanitizer 工具里面

#### enable-debug-info

用于传递对应的triton kernel行号信息