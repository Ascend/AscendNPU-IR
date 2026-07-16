# 版本配套说明

本文说明AscendNPU IR依赖的CANN软件栈以及硬件驱动环境。

为了保证编译与运行的稳定性，请严格按照本文档提供的版本配套关系进行环境配置。

## CANN配套关系

以下版本组合为经过验证的推荐配置，建议优先使用：

| AscendNPU IR版本 | Gitcode分支 | 依赖CANN版本 | 硬件支持 |
| --- | --- | --- | --- |
| `v1.0.0` | `release/v1.0.0` | CANN 8.5.0 | Ascend A2/A3 |
| `v1.1.0` | `release/v1.1.0` | CANN 9.0.0 | Ascend A2/A3；<br>950系列（branch `feature_a5`） |

**重要说明**：

如果您的CANN版本不想进行升级和替换，又想使用AscendNPU IR的新特性，可以参考如下方式尝试体验：

```bash
NEW_CANN_PKG="PATH-TO/Ascend-cann-toolkit_9.0.0_linux-aarch64.run"
TMP_PATH="tmp_pkg_files"
OLD_CANN_PATH="${CANN_850_PATH}/Ascend/cann-8.5.0"

bash ${NEW_CANN_PKG} --noexec --extract=cann900
bash cann900/run_package/ascendnpu-ir_*.run --noexec --extract=$TMP_PATH
cp -r $TMP_PATH/bishengir/* ${OLD_CANN_PATH}/tools/bishengir/
rm -rf $TMP_PATH

# 一般执行完上述就会基本可行了，如果还有问题，可以进一步尝试
bash cann900/run_package/cann-bisheng-compiler_*.run --noexec --extract=$TMP_PATH
BiShengCompilerPath="${TMP_PATH}/bisheng_compiler" # CANN 9.1.0 后路径为 "tools/bisheng_compiler/"
cp -r $BiShengCompilerPath/* ${OLD_CANN_PATH}/tools/bisheng_compiler/
rm -rf $TMP_PATH
```

## Python配套关系

AscendNPU IR同时包含Python的wheel包，可以通过`pip`安装：

```bash
pip install ascendnpu-ir
```

支持的Python版本范围如下：

| AscendNPU IR版本 | Python版本支持 |
| --- | --- |
| `v1.0.0` | `>=3.9，<=3.12` |
| `v1.1.0` | `>=3.10，<=3.13` |
