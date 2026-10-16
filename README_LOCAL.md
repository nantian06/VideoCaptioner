# 本地中文字幕版（VideoCaptioner 定制版）v1.5.0

基于 [VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner)（GPL-3.0）改的个人定制版，主要面向**中文直播录像、夹杂少量英文**的场景：

- 新增「本地中文识别」引擎：**SenseVoice**（快）和 **Fun-ASR-Nano**（更准），完全离线，不上传音频。
- 有 N 卡自动用 GPU，没有显卡或 CUDA 环境不完整时**自动回退 CPU**，不会报错退出。
- 模型第一次使用时自动下载，带国内加速源和断点续传。
- 保留原版的 LLM 断句/校正/翻译、字幕样式、视频合成（软/硬字幕）、批量处理、视频下载、命令行配音等功能，并按使用习惯重新梳理了流程（见 [docs/local-fork/AUDIT_zh.md](docs/local-fork/AUDIT_zh.md)）。
- 去掉了联网检查更新、远程公告/强制更新开关、捐赠和推广入口。

支持 **Linux（x86_64）和 Windows 10/11**。

---

## 一、安装

需要：能上网（装依赖、下模型），磁盘 CPU 版约 2GB、GPU 版约 5GB（不含模型）。

### Linux

```bash
unzip local-asr-captioner-v1.5.0.zip && cd local-asr-captioner
./install_linux.sh          # 自动判断：有 N 卡驱动装 GPU 版，否则装 CPU 版
# 或者手动指定
./install_linux.sh --gpu
./install_linux.sh --cpu
```

脚本会做这些事：
1. 缺 ffmpeg 时用 apt/dnf/pacman 安装；Debian/Ubuntu 上补齐 Qt 界面需要的 xcb 库和中文字体（会要 sudo 密码）。
2. 安装 [uv](https://docs.astral.sh/uv/)，建一个 Python 3.12 虚拟环境（`.venv`），装好所有依赖。
3. GPU 版：再下载 sherpa-onnx 的 CUDA 版 wheel（先试 huggingface.co，再试 hf-mirror.com），并用 pip 安装 CUDA 12 / cuDNN 9 运行库（`nvidia-*-cu12`，约 2-3GB）。**不需要系统里装 CUDA Toolkit**，只要显卡驱动支持 CUDA 12（驱动版本 ≥ 525）。
4. 运行 `videocaptioner doctor` 自检，告诉你装的是 CPU 版还是 GPU 版、检测到哪张卡。

GPU 版任何一步失败，脚本都会自动恢复成 CPU 版，软件照样能用。

### Windows

双击 `install_windows.bat`（自动判断），或在命令行里 `install_windows.bat --gpu` / `--cpu`。步骤和 Linux 相同；ffmpeg 缺失时用 winget 安装。

### 网络不好时

| 情况 | 办法 |
|---|---|
| PyPI 慢 | 安装前设置 `UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` |
| CUDA 版 wheel 下不动 | 浏览器从 <https://k2-fsa.github.io/sherpa/onnx/cuda.html> 下载 `sherpa_onnx-1.13.8+cuda12.cudnn9…cp312…` 对应系统的 whl，然后 `VC_CUDA_WHEEL=/路径/xxx.whl ./install_linux.sh --gpu`（Windows：`set VC_CUDA_WHEEL=D:\xxx.whl` 后再运行脚本） |

> ⚠️ 装好以后不要自己运行 `uv sync` 或 `uv run`，它们会按锁文件把 GPU 版换回 CPU 版。要更新或切换，重新运行安装脚本即可。

### 启动

- 界面：Linux `./run_gui.sh`，Windows 双击 `run_gui.bat`
- 命令行：Linux `./vc.sh …`，Windows `vc.bat …`

---

## 二、模型

| 模型 | 用途 | 大小 | 说明 |
|---|---|---|---|
| silero VAD | 切分人声片段 | 不到 1MB | 必需，自动下载 |
| SenseVoice int8 | CPU 默认 | 约 230MB（下载包 160MB） | 中/英/日/韩/粤，速度快 |
| SenseVoice fp32 | GPU 默认 | 约 900MB（下载包 1GB） | 同上，GPU 上用全精度 |
| Fun-ASR-Nano int8 | CPU 用 | 约 970MB（下载包 800MB） | 识别更准，CPU 上较慢 |
| Fun-ASR-Nano fp16 | GPU 用 | 下载包约 1GB | 同上 |

- 选好模型后**第一次识别时自动下载**；也可以在 **设置 → 转录配置 → 模型管理** 里提前下载、查看哪些已装、打开模型文件夹。
- Fun-ASR-Nano 会同时用到 SenseVoice（用它的时间戳给 Nano 的文字定位），所以两个都会下载。
- 默认存放位置：程序目录下 `AppData/models/local_asr/`。可以在设置的「模型目录」或命令行 `--models-dir` 改到别的盘。
- **下载源**（设置里「下载源」，命令行 `--download-source`）：
  - `auto`（默认）：依次试 ghfast.top → gh-proxy.com → GitHub，速度太慢自动换源，支持断点续传；
  - `github` / `ghfast` / `ghproxy`：只用某一个；
  - `custom`：填你自己的 GitHub 加速前缀。
- 想手动下载：从 <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models> 下载下面几个文件，`silero_vad.onnx` 直接放进模型目录，`.tar.bz2` 解压到模型目录：
  - `silero_vad.onnx`
  - `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`
  - `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2`（GPU 用）
  - `sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2`
  - `sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2`（GPU 用）

---

## 三、使用

### 界面

1. **主页 → 任务创建**：拖入视频/音频，或粘贴视频链接。
2. 下面的 **「处理方案」卡片**一眼看清这次会做什么，可以直接改：
   - **识别**：引擎（默认「本地中文识别」）、模型、设备；点「识别设置」进详细设置。
   - **断句**：「按标点和停顿」（本地，不需要 LLM，默认）或「LLM 智能断句」。
   - **字幕校正** / **翻译**：需要 LLM 的会提示去填 API Key。
   - **视频**：只出字幕（默认）/ 软字幕（秒出，不重新编码）/ 硬字幕（烧进画面，长视频很慢）。
3. 点 **开始处理**。所有产物放在 `work-dir/<视频名>/` 一个文件夹里，不会改动或覆盖原视频旁边的文件。
4. 识别过程中可以点 **取消转录**。

**批量处理**页顶部会显示当前方案，点「修改方案」回到主页调整。

识别设置里几个常用项：
- **语言**：默认中文（中英夹杂也是选中文，英文照样能识别）。选「自动」也按中文处理；纯日语/韩语/粤语请明确选择。
- **热词**：主播名、游戏名、专有名词，用逗号分隔，只对 Fun-ASR-Nano 生效。
- **每行最多字数**：默认 25（中文字算 1，英文单词算 1.5）。

### 命令行

```bash
./vc.sh doctor                                     # 检查环境、GPU、模型
./vc.sh transcribe 直播.mp4                         # 默认：本地 SenseVoice，自动选设备
./vc.sh transcribe 直播.mp4 --local-model funasr-nano --hotwords "原神,米哈游"
./vc.sh transcribe 直播.mp4 --device cpu --threads 8 --max-chars 20
./vc.sh transcribe 直播.mp4 --download-source ghfast --models-dir /data/models
./vc.sh process 直播.mp4 --no-optimize --no-translate --no-split --subtitle-mode soft
                                                   # 一条龙：本地识别断句 + 软字幕视频，不需要 LLM
./vc.sh --help
```

配音（TTS）目前只有命令行：`./vc.sh dub --help`。

---

## 四、速度参考

本机实测（4 核 CPU，无显卡，175 秒的中英混合直播片段）：

| 模型 | 用时 | 实时倍率 |
|---|---|---|
| SenseVoice int8 | 约 8 秒 | 约 22 倍 |
| Fun-ASR-Nano int8 | 约 40 秒 | 约 4 倍 |

Fun-ASR-Nano 在专有名词、英文单词和同音字上明显更准（例如「频繁」「时态」「MONDAY」），有 N 卡建议用它。

---

## 五、说明与已知限制

- **GPU 路径没有在真实 N 卡上测过**（开发环境没有显卡）。已经测过的是：没有 N 卡时自动用 CPU；装了 CPU 版却选 GPU 时给出提示并回退；CUDA 探测在子进程里进行，失败不会拖垮主程序。如果你的 N 卡识别异常，运行 `./vc.sh doctor` 把输出发给我。
- hf-mirror.com、ghfast.top、gh-proxy.com 等国内源是第三方服务，可用性随时变化；GitHub 直连在开发环境测通过。
- Windows 安装脚本没有在 Windows 上实际运行过，Linux 脚本在 Ubuntu 24.04 上测过 CPU 安装和「GPU 下载失败 → 回退 CPU」两条路径。
- 原版的「B 接口 / J 接口」云端识别仍然保留，会上传音频，且有原版的时长/次数限制。
- 本项目沿用原项目的 GPL-3.0 许可证。模型的许可证见各模型目录里的 LICENSE / README。
