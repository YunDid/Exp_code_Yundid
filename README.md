# Exp_code

这是对 `../exp_all.py` 的第一阶段可读性拆分版本。

目标只有两个：

- 保留原来的实验逻辑与执行顺序
- 让你以后改范式时能直接定位到对应职责文件

当前拆分：

- `main.py`
  - 单一入口
  - 对应原来 `main()` + 顶层运行选择
- `experiment_config.py`
  - 集中放电极选择与 `CONFIG`
  - 当前已不再依赖旧 `exp_all.py`
- `system_api.py`
  - 系统初始化、array 配置、stim unit 映射与上下电
- `recording_api.py`
  - routing/download/offset/clear_events
  - start/stop recording
- `stimulation_api.py`
  - 序列构建、事件标签、unit 连接子集切换
- `protocols.py`
  - `run_random_stim_experiment`
  - `run_test_block`
  - `run_train_block`
  - `run_protocol`

当前剩余的人手调整点：

1. Linux 实验机上的保存路径仍在 `experiment_config.py` 中手工修改
2. 新范式需要时，再决定是否把 test/train/new paradigm 继续拆成独立协议文件

推荐启动方式：

- 在 `Exp_code` 目录内执行：`python3 main.py`
- 或在上一级目录执行：`python3 -m Exp_code.main`

`main.py` 已经做了启动时的 `sys.path` 引导，目的是兼容 `Exp_code` 子目录运行，同时还能找到上一级目录中的 `maxlab` 包。
