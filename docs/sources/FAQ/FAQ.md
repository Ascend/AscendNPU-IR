# 常见问题答疑


## - 如何获取中间编译态日志

### 问题描述
 AscendNPU-IR编译出来的bishengir-compile转换ttadapter，需要获取到下层mlir的内容，例如hivm等层的mlir转换结果。

### 打印命令

关于这个问题，具体需要看build.sh中的可选项，如果需要获取到编译优化后导出的mlir，需要将build.sh文件中的可选项<ENABLE_IR_PRINT>以及<BISHENGIR_PUBLISH>设置为ON，直接使用下面的编译命令进行编译即可，这个会打印出来AscendNPU IR带有内存、同步等信息的中间mlir。
```python
--bishengir-print-ir-before=hivm-inject-block-sync --bishengir-print-ir-after=hivm-inject-block-sync
```




