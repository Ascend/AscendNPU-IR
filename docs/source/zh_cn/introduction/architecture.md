# 架构设计

## 目标定位

毕昇编译器AscendNPU IR是基于MLIR生态构建的昇腾硬件高层抽象表达，它会自下而上对昇腾硬件底层指令、核内资源、核间资源、SOC资源逐层进行抽象编译优化，多层抽象间分层解耦、开源开放，允许生态编程、三方框架权衡性能与易用性的需求灵活对接，为生态框架提供面向昇腾的统一编译接入层和硬件完备表达优化能力。

![image](../../images/introduction/architecture1_zh.png)

## 逻辑架构

AscendNPU IR中自研设计的方言有HFusion、HIVM、HACC、Annotation、Scope，其中HFusion方言负责硬件相对无关的优化，HIVM则负责精细化感知NPU硬件细节，将High level的编程语言转换成NPU的底层指令，HACC方言负责异构硬件抽象表达，Annotation和Scope则负责对于特定Operand或者Operation标记`compiler hint`信息。

![image](../../images/introduction/architecture2_zh.png)

### HFusion方言

HFusion（Hybrid Fusion）方言是基于MLIR社区Linalg方言的扩展集，HFusion方言继承了Linalg方言的所有`operations`并且自行扩展了Linalg社区还未支持的`operations`，要注意HFusion方言处理的`operations`均是`named operations`，这样可以最大化保留高层语义方便编译器处理。HFusion方言主要包括转换层、预处理、融合处理三层能力：

1. **转换层**：HFusion方言是生态对接关键的一层，当前支持与Arith、Math、Torch等方言关键Operations的`Conversion`对接，后续会逐步完善补齐生态对接能力。

2. **预处理**：硬件细节相对无关优化层，支持Tensor表达式化简、`BF16`/`Bool`数据类型合法化、复杂Op组合实现等常见Device函数优化。

3. **融合处理**：能够自动融合生成Device Kernel算子及Host Tiling函数。

### HIVM方言

HIVM（Hybrid ISA Virtual Machine）：面向昇腾硬件对计算、搬运、同步等操作进行抽象，提供Tile级Operation支持任意维度、大小的Tensor或者`Memref`操作类型，屏蔽昇腾硬件底层指令参数。HIVM层编译优化主要分为以下三层：

1. **CV核映射编译**：感知NPU CV核分离硬件架构自动对Mix Kernel（既包括`cube`操作又包括`vector`操作的核函数）进行CV融合编译优化，通过分析`cube`和`vector`操作间的数据依赖关系自动插入`store`和`load`来进行CV核数据交互，计算中间交互所需的workspace global memory空间大小并生成Host侧推导大小的函数，同时对有CV数据依赖处插入核间同步保证依赖顺序，最后自动拆分MixKernel为单独的AIC核函数和AIV核函数，从而实现CV融合编译功能。在性能优化上，通过CVPipeline pass自动实现调整Cube代码与Vector代码顺序保证CV核流水并行，通过AutoSubTiling自动实现CV配比1:2切分特性。

2. **核内片上内存映射**：感知NPU核内片上内存结构，编译优化自动实现片上内存空间推导、片上内存数据格式推导、片上访存自动对齐、Op临时空间申请以及片上内存地址分配。

3. **核内处理单元映射**：感知NPU核内多级流水处理单元，自动插入流水同步操作保证不同流水线有序执行同时并行流水优化；感知NPU指令细节自动完成基于策略的指令自动映射，使能NPU SIMD高效指令。

### A5芯片上的特性与AscendNPU IR的支持优化

A5芯片继承了310B芯片的RegBase（Register-based）的编程模型，硬件上相比A2和A3芯片Memory-based编程模型增加了寄存器层；在Cube和Vector核之间增加了数据通路，为CV融合提供更多优化空间；增加了Warp Scheduler等组件引入SIMT能力；增加`ND-DMA`等一些新的硬件指令。

AscendNPU IR在HIVM方言中对新的硬件特性提供了支持，包括Arith和Vector方言支持计算与规约类Op。对于纯SIMD编译增加了VF融合、向量化、掩码优化和Combine优化。在A5上新增SIMT编译支持，将社区的TritonGPU方言对接至HIVM，构建昇腾亲和的Layout优化、共享内存分配、核心指令映射优化的算法。AscendNPU IR在A5上除了支持纯SIMD模式和纯SIMT以外，还支持SIMD/SIMT混合编译。

![A5的AscendNPU IR架构](../../images/introduction/architecture_A5_zh.png)

## 代码架构

AscendNPU IR是基于MLIR生态构建的，MLIR原生社区代码是作为第三方引入，代码结构如下所示，`bishengir`（即AscendNPU IR）目录下是AscendNPU IR相关实现，`build-tools`目录下是AscendNPU IR构建所需脚本。AscendNPU IR对于MLIR原生社区的增强会优先在`include/bishengir/Dialect`独立目录下创建对应方言目录，通过独立目录新增文件扩展能力来避免对社区侵入式修改；对于无法隔离的修改，已直接提交到`third-party`下对应的Ascend维护分支中（例如`llvm-project`对应分支为`Ascend/AscendNPU-IR/llvmorg-19.1.7`、torch-mlir对应分支为`Ascend/AscendNPU-IR/main-20250716`），每个修改均有单独`commit`信息，便于后续回合MLIR社区。历史版本中通过`build-tools/patches`目录下的patch文件在构建时应用的方式已废弃。

```text
.
├── bishengir // AscendNPU IR 相关实现
├── build-tools // AscendNPU IR 构建脚本所在目录
│   └── build.sh
└── third-party
    ├── llvm-project // Ascend 维护分支：Ascend/AscendNPU-IR/llvmorg-19.1.7
    ├── shmem
    └── torch-mlir   // Ascend 维护分支：Ascend/AscendNPU-IR/main-20250716
```

`bishengir`目录结构和`mlir`目录结构保持一致：`include`下存放声明文件，包括C++头文件（`.h`、`.hpp`）和TableGen定义文件（`.td`），构建目录`build/include`中包含TableGen自动生成的文件（`.h.inc`、`.cpp.inc`）；`lib`目录存放实现代码（`.cpp`），其目录结构与`include`基本保持一致。

```text
.
├── bishengir // AscendNPU IR 相关实现
│   ├── include
│   │   └── bishengir
│   │       ├── Conversion
│   │       └── Dialect
│   │           ├── 社区方言 // 对于社区方言的扩展增强
│   │           └── 自研方言 // 自定义的方言
├── lib
└── tools
    ├── bishengir-compile // AscendNPU IR 编译器命令行驱动程序
    └── bishengir-opt
```

IR中主要由`Conversion`、`Dialect`、`tools`三部分组成，其中`Conversion`承载不同方言间转换的能力，`Dialect`下是不同方言的定义和实现，`tools`目录下定义编译工具链。
`Conversion`中既包括三方生态对接转换（如TorchToHFusion）也包括AscendNPU IR内部方言间转换（如HFusionToHIVM）；`Dialect`下既包括自研方言也包括社区方言；`tools`中`bishengir-compile`是AscendNPU IR编译器的命令行驱动程序。

## 编译流程

AscendNPU IR对应工具链是`bishengir-compile`，会负责把高抽象层级的Tile级Op编译成感知NPU硬件架构的low level op，该工具链输入和输出均是MLIR。`hivmc`工具会负责把low level的MLIR转成LLVM IR并基于LLVM IR进行底层指令编译优化，最终生成算子二进制。
![image](../../images/introduction/architecture3_zh.png)
