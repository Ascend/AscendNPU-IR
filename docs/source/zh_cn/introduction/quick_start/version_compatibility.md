# 版本配套说明

本文说明AscendNPU IR依赖的CANN软件栈以及硬件驱动环境。

为了保证编译与运行的稳定性，请严格按照本文档提供的版本配套关系进行环境配置。

## CANN配套关系

以下版本组合为经过验证的推荐配置，建议优先使用：

| AscendNPU IR版本 | Gitcode分支 | 依赖CANN版本 | 硬件支持 |
| --- | --- | --- | --- |
| `v1.2.0` | `release/v1.2.x` | CANN 9.1.0 | <ul><li>Ascend 950PR&950DT系列产品(branch `feature/regbase`)</li><li>Atlas A3系列产品</li><li>Atlas A2系列产品</li></ul> |
| `v1.1.0` | `release/v1.1.x` | CANN 9.0.0 | <ul><li>Ascend 950PR&950DT系列产品(branch `feature_a5`)</li><li>Atlas A3系列产品</li><li>Atlas A2系列产品</li></ul> |
| `v1.0.0` | `release/v1.0.0` | CANN 8.5.0 | <ul><li>Atlas A3系列产品</li><li>Atlas A2系列产品</li></ul> |

**重要说明**：

如需保留当前CANN版本不做升级替换，又想使用AscendNPU IR的新特性，可以从新版CANN工具包中提取AscendNPU IR组件，将当前CANN包中的AscendNPU IR进行版本替换。参考步骤如下：

```bash
# 本示例：在CANN 8.5.0安装中，将其AscendNPU IR替换为CANN 9.0.0包中的版本
# 1. 准备变量：新版CANN包路径（<arch>按实际架构填写，如x86_64）、临时解压目录、当前CANN安装路径
NEW_CANN_PKG="PATH-TO/Ascend-cann_9.0.0_linux-<arch>.run"
TMP_PATH="tmp_pkg_files"
OLD_CANN_PATH="${CANN_850_PATH}/Ascend/cann-8.5.0"

# 2. 解包新版CANN工具包（--noexec表示只解包、不安装）
bash ${NEW_CANN_PKG} --noexec --extract=cann900

# 3. 从解包目录中提取ascendnpu-ir子包
bash cann900/run_package/ascendnpu-ir_*.run --noexec --extract=$TMP_PATH

# 4. 用提取出的新版bishengir替换当前CANN包中的NPUIR，并清理临时文件
cp -r $TMP_PATH/bishengir/* ${OLD_CANN_PATH}/tools/bishengir/
rm -rf $TMP_PATH

# 5. 一般完成上述替换即可；如仍有问题，可按同样方式进一步替换bisheng-compiler
bash cann900/run_package/cann-bisheng-compiler_*.run --noexec --extract=$TMP_PATH
BiShengCompilerPath="${TMP_PATH}/bisheng_compiler" # CANN 9.1.0及之后版本路径为 "tools/bisheng_compiler/"
cp -r $BiShengCompilerPath/* ${OLD_CANN_PATH}/tools/bisheng_compiler/
rm -rf $TMP_PATH
```

## Python配套关系

AscendNPU IR同时包含Python的wheel包，可以通过`pip`安装：

```bash
pip install ascendnpu-ir
```

安装完成后，可以在Python中直接调用编译接口，示例如下：

```python
import ascendnpuir

# 待编译的MLIR字符串
mlir_str = """
module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, %arg1: memref<16xi16, #hivm.address_space<gm>>, %arg2: memref<16xi16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>, memref<16xi16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
"""
output_path = "example.o"
options = [
    "-enable-hivm-compile=true",
]

# 调用编译接口，生成目标文件
res = ascendnpuir.compile(
    mlir_str,
    output_path=output_path,
    option=options,
)
print(f"Compiled to: {output_path}")
```

支持的Python版本范围如下：

| AscendNPU IR版本 | Python版本支持 |
| --- | --- |
| `v1.0.0` | `>=3.9，<=3.12` |
| `v1.1.0` | `>=3.10，<=3.13` |
