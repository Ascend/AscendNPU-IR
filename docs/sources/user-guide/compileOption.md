# 编译选项
## 1.1 参考：编译优化选项

| 编译选项| 含义| 取值范围|
| --- | --- | --- |
| multibuffer | 设置是否启用乒乓流水 | False(默认),True |
| limit_auto_multi_buffer_of_local_buffer | 设置乒乓流水在片中(L1, L0, 及UB)的作用范围"no-limit"表示不限乒乓流水范围"no-l0c"表示只允许L0缓存外启用乒乓流水 | "no-limit","no-l0c"(默认) |
| unit_flag | 设置cube搬出时是否按照block搬出，仅限数据对齐场景下使用 | False(默认),True |
| limit_auto_multi_buffer_only_for_local_buffer | 设置是否在GM workspace中启用CV流水并行，False表示启用  后续会整改接口，提供更可读的选项 | False(默认),True |
| set_workspace_multibuffer | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置CV并行的并行度使用时需确保数据没有依赖若设置为N，则N个CV操作并行执行 | 2 (默认),4 |
| tile_mix_vector_loop | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置当前vector的切分数量，数值可由autotuning得出，均可为最优 | 1 (默认),2,4 |
| tile_mix_cube_loop | 仅在limit_auto_multi_buffer_only_for_local_buffer=false的场景下生效设置当前cube的切分数量，数值可由autotuning得出，均可为最优 | 1 (默认),2,4 |


## 1.2 bishengir-compile编译选项


## 1.3 bishengir-hivm-compile编译选项

