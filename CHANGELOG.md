# 更新日志（本地中文定制版）

## v1.5.0 · 2026-10-16

基于上游 VideoCaptioner v1.4.2 之后的 master（95842ec）。

### 新增
- 「本地中文识别」引擎（sherpa-onnx）：SenseVoice（快，CPU 约 22 倍实时）和 Fun-ASR-Nano（更准，支持热词），完全离线。
- N 卡自动加速；没有 N 卡、装的是 CPU 版、或 CUDA 运行库缺失时自动回退 CPU，并说明原因。
- 模型按需自动下载：GitHub 加速源自动切换、断点续传，设置里有「模型管理」。
- 按标点、停顿、每行字数在本地断句，不需要 LLM；英文单词之间自动补空格。
- 任务创建页的「处理方案」卡片，批量页显示同一方案。
- 转录可取消。
- 命令行：`--asr local`（新默认）、`--local-model`、`--device`、`--hotwords`、`--threads`、`--models-dir`、`--download-source`、`--max-chars`；`doctor` 会检查本地识别环境。
- 安装脚本 `install_linux.sh` / `install_windows.bat`（自动选择 GPU/CPU 版），启动脚本 `run_gui.*` / `vc.*`。

### 改动
- 默认值：本地识别、中文、只出字幕（不重新编码视频）、合成时用软字幕。
- 所有产物统一放在 `work-dir/<视频名>/`，不再往原视频旁边写字幕或覆盖输入文件。
- 修改 LLM 地址或 Key 后立即生效，不用重启。
- LLM 校正有失败批次时给出警告，不再提示「优化完成」。

### 移除
- 启动时联网检查版本、远程公告、远程强制更新开关。
- 捐赠和推广入口。

### 修复
- Linux 上缺少 `xdg-open` 时打开文件夹导致崩溃。
- 字幕处理的取消操作停不掉 LLM 优化器。

详见 [docs/local-fork/AUDIT_zh.md](docs/local-fork/AUDIT_zh.md)。
