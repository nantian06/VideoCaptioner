"""本地中文识别（SenseVoice / Fun-ASR-Nano）的设置界面和模型管理对话框。"""

import threading
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBoxSettingCard,
    InfoBar,
    InfoBarPosition,
    MessageBoxBase,
    ProgressBar,
    PushButton,
    PushSettingCard,
    SettingCardGroup,
    SingleDirectionScrollArea,
    StrongBodyLabel,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.core.asr.local import models as M
from videocaptioner.core.entities import (
    LOCAL_ASR_MODEL_LABELS,
    LOCAL_DEVICE_LABELS,
    LocalASRModelEnum,
    LocalDeviceEnum,
    TranscribeLanguageEnum,
)
from videocaptioner.core.utils.platform_utils import open_folder
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.LineEditSettingCard import LineEditSettingCard
from videocaptioner.ui.components.SpinBoxSettingCard import (
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)

DOWNLOAD_SOURCE_LABELS = [
    "自动（先国内加速，再 GitHub）",
    "GitHub 直连",
    "国内加速 ghfast.top",
    "国内加速 gh-proxy.com",
    "自定义镜像前缀",
]


def models_root() -> Path:
    return M.get_models_root(cfg.local_models_dir.value or None)


def needed_specs(device: str = "cpu") -> List[M.ModelSpec]:
    """当前设置需要的模型（VAD + SenseVoice [+ Nano]）。"""
    root = models_root()
    specs = [M.MODEL_SPECS["silero-vad"], M.pick_variant("sensevoice", device, root)]
    if cfg.local_model.value == LocalASRModelEnum.FUNASR_NANO:
        specs.append(M.pick_variant("funasr-nano", device, root))
    return specs


class ModelDownloadThread(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, specs: List[M.ModelSpec]):
        super().__init__()
        self.specs = specs
        self.stop_event = threading.Event()

    def run(self):
        try:
            M.ensure_models(
                self.specs,
                models_root(),
                lambda p, m: self.progress.emit(p, m),
                cfg.local_download_source.value,
                cfg.local_custom_mirror.value,
                self.stop_event,
            )
            self.finished_ok.emit()
        except M.DownloadCancelled:
            self.error.emit(self.tr("已取消下载（下次会从断点继续）"))
        except Exception as e:
            self.error.emit(str(e))


class DeviceCheckThread(QThread):
    result = pyqtSignal(str, str)

    def run(self):
        from videocaptioner.core.asr.local.device import resolve_device

        vad = M.model_path(M.MODEL_SPECS["silero-vad"], "model", models_root())
        d = resolve_device(cfg.local_device.value.value, vad if vad.exists() else None)
        self.result.emit(d.device, d.reason)


class LocalModelManagerDialog(MessageBoxBase):
    """模型管理：查看已下载模型、下载缺失模型、检测 GPU。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.download_thread: Optional[ModelDownloadThread] = None
        self.titleLabel = StrongBodyLabel(self.tr("本地识别模型管理"), self)
        self.viewLayout.addWidget(self.titleLabel)

        self.device_label = BodyLabel(self.tr("正在检测显卡…"), self)
        self.device_label.setWordWrap(True)
        self.viewLayout.addWidget(self.device_label)

        self.rows_widget = QWidget(self)
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 8, 0, 8)
        self.viewLayout.addWidget(self.rows_widget)

        self.progress_label = CaptionLabel("", self)
        self.progress_label.setWordWrap(True)
        self.progress_bar = ProgressBar(self)
        self.progress_label.hide()
        self.progress_bar.hide()
        self.viewLayout.addWidget(self.progress_label)
        self.viewLayout.addWidget(self.progress_bar)

        btns = QHBoxLayout()
        self.download_needed_btn = PushButton(FIF.DOWNLOAD, self.tr("下载当前设置需要的模型"), self)
        self.open_dir_btn = PushButton(FIF.FOLDER, self.tr("打开模型文件夹"), self)
        btns.addWidget(self.download_needed_btn)
        btns.addWidget(self.open_dir_btn)
        self.viewLayout.addLayout(btns)

        hint = CaptionLabel(
            self.tr(
                "首次识别时也会自动下载。下载慢可在设置里换下载源，"
                "或按 README 手动下载 .tar.bz2 解压到模型文件夹。"
            ),
            self,
        )
        hint.setWordWrap(True)
        self.viewLayout.addWidget(hint)

        self.download_needed_btn.clicked.connect(self._download_needed)
        self.open_dir_btn.clicked.connect(lambda: open_folder(str(models_root())))

        self.yesButton.hide()
        self.cancelButton.setText(self.tr("关闭"))
        self.widget.setMinimumWidth(560)
        self._device = "cpu"
        self._refresh_rows()

        self.device_thread = DeviceCheckThread()
        self.device_thread.result.connect(self._on_device)
        self.device_thread.start()

    def _on_device(self, device: str, reason: str):
        self._device = device
        prefix = "🟢 " if device == "cuda" else "⚪ "
        self.device_label.setText(prefix + reason)
        self._refresh_rows()

    def _refresh_rows(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        root = models_root()
        needed = {s.key for s in needed_specs(self._device)}
        for spec in M.MODEL_SPECS.values():
            row = QWidget(self.rows_widget)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 2, 0, 2)
            installed = M.is_installed(spec, root)
            mark = "✅" if installed else ("⬇️" if spec.key in needed else "　")
            text = f"{mark} {spec.title}  ·  约 {spec.size_mb} MB"
            if spec.key in needed:
                text += self.tr("  ·  当前需要")
            label = BodyLabel(text, row)
            lay.addWidget(label, 1)
            if not installed:
                btn = PushButton(self.tr("下载"), row)
                btn.clicked.connect(lambda _=False, s=spec: self._start_download([s]))
                lay.addWidget(btn)
            self.rows_layout.addWidget(row)

    def _download_needed(self):
        missing = M.missing_specs(needed_specs(self._device), models_root())
        if not missing:
            InfoBar.success(self.tr("无需下载"), self.tr("当前设置需要的模型都已就绪"), parent=self, position=InfoBarPosition.TOP)
            return
        self._start_download(missing)

    def _start_download(self, specs: List[M.ModelSpec]):
        if self.download_thread is not None and self.download_thread.isRunning():
            return
        self.progress_label.show()
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.download_needed_btn.setEnabled(False)
        self.download_thread = ModelDownloadThread(specs)
        self.download_thread.progress.connect(self._on_progress)
        self.download_thread.finished_ok.connect(self._on_done)
        self.download_thread.error.connect(self._on_error)
        self.download_thread.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_done(self):
        self.download_needed_btn.setEnabled(True)
        self.progress_label.setText(self.tr("下载完成"))
        self._refresh_rows()

    def _on_error(self, msg: str):
        self.download_needed_btn.setEnabled(True)
        self.progress_bar.error()
        self.progress_label.setText(msg)
        self._refresh_rows()

    def reject(self):
        if self.download_thread is not None and self.download_thread.isRunning():
            self.download_thread.stop_event.set()
            self.download_thread.wait(3000)
        super().reject()


def create_basic_cards(parent_group: SettingCardGroup):
    """本地识别的常用设置卡片（设置页和转录页共用）。"""
    model_card = ComboBoxSettingCard(
        cfg.local_model,
        FIF.ROBOT,
        "识别模型",
        "SenseVoice 速度快；Fun-ASR-Nano 对专有名词和口音更准，但慢 5 倍左右",
        [LOCAL_ASR_MODEL_LABELS[m] for m in LocalASRModelEnum],
        parent_group,
    )
    device_card = ComboBoxSettingCard(
        cfg.local_device,
        FIF.IOT,
        "运行设备",
        "自动：能用 N 卡就用 GPU，否则用 CPU",
        [LOCAL_DEVICE_LABELS[d] for d in LocalDeviceEnum],
        parent_group,
    )
    language_card = ComboBoxSettingCard(
        cfg.transcribe_language,
        FIF.LANGUAGE,
        "主要语言",
        "以中文为主选「中文」（夹杂的英文照样能识别），可以避免把中文认成日文假名",
        [lang.value for lang in TranscribeLanguageEnum],
        parent_group,
    )
    language_card.comboBox.setMaxVisibleItems(8)
    hotword_card = LineEditSettingCard(
        cfg.local_hotwords,
        FIF.TAG,
        "热词",
        "用逗号分隔，例如主播名、商品名、圈内术语（仅 Fun-ASR-Nano 生效）",
        "例如：小明,蜜雪冰城,GPT",
        parent_group,
    )
    manage_card = PushSettingCard(
        "模型管理",
        FIF.DOWNLOAD,
        "模型与显卡",
        "查看/下载模型，检测 N 卡是否可用",
        parent_group,
    )
    for card in (model_card, device_card, language_card):
        card.comboBox.setMinimumWidth(240)
    hotword_card.lineEdit.setMinimumWidth(240)
    return [model_card, device_card, language_card, hotword_card, manage_card]


class LocalASRSettingWidget(QWidget):
    """转录页里「本地中文识别」的设置面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical, parent=self)  # type: ignore
        self.scrollArea.setStyleSheet("QScrollArea{background: transparent; border: none}")
        self.container = QWidget(self)
        self.container.setStyleSheet("QWidget{background: transparent}")
        self.containerLayout = QVBoxLayout(self.container)

        self.basic_group = SettingCardGroup(self.tr("本地中文识别（离线，不上传音频）"), self)
        self.basic_cards = create_basic_cards(self.basic_group)
        for card in self.basic_cards:
            self.basic_group.addSettingCard(card)
        self.basic_cards[-1].clicked.connect(self._show_manager)

        self.adv_group = SettingCardGroup(self.tr("高级"), self)
        self.max_chars_card = SpinBoxSettingCard(
            cfg.local_max_chars,
            FIF.ALIGNMENT,
            self.tr("每条字幕最多字数"),
            self.tr("不用 LLM 断句时，按标点和停顿切句的字数上限（英文一个词按 1.5 字算）"),
            minimum=8,
            maximum=60,
        )
        self.vad_card = DoubleSpinBoxSettingCard(
            cfg.local_vad_threshold,
            FIF.VOLUME,  # type: ignore
            self.tr("人声检测阈值"),
            self.tr("调高能少收背景音乐/杂音，调低能少漏小声说话，默认 0.5"),
            minimum=0.1,
            maximum=0.9,
            decimals=2,
            step=0.05,
        )
        self.threads_card = SpinBoxSettingCard(
            cfg.local_num_threads,
            FIF.SPEED_HIGH,
            self.tr("CPU 线程数"),
            self.tr("0 表示自动"),
            minimum=0,
            maximum=64,
        )
        self.source_card = ComboBoxSettingCard(
            cfg.local_download_source,
            FIF.CLOUD_DOWNLOAD,
            self.tr("模型下载源"),
            self.tr("模型托管在 GitHub；国内建议用加速源"),
            DOWNLOAD_SOURCE_LABELS,
            self.adv_group,
        )
        self.mirror_card = LineEditSettingCard(
            cfg.local_custom_mirror,
            FIF.LINK,
            self.tr("自定义镜像前缀"),
            self.tr("下载源选「自定义」时使用，会拼在 GitHub 地址前面"),
            "https://你的加速地址/",
            self.adv_group,
        )
        self.dir_card = PushSettingCard(
            self.tr("选择文件夹"),
            FIF.FOLDER,
            self.tr("模型文件夹"),
            str(models_root()),
            self.adv_group,
        )
        for card in (
            self.max_chars_card,
            self.vad_card,
            self.threads_card,
            self.source_card,
            self.mirror_card,
            self.dir_card,
        ):
            self.adv_group.addSettingCard(card)
        self.source_card.comboBox.setMinimumWidth(240)
        self.dir_card.clicked.connect(self._choose_dir)

        self.containerLayout.addWidget(self.basic_group)
        self.containerLayout.addWidget(self.adv_group)
        self.containerLayout.addStretch(1)
        self.scrollArea.setWidget(self.container)
        self.scrollArea.setWidgetResizable(True)
        self.main_layout.addWidget(self.scrollArea)

    def _show_manager(self):
        LocalModelManagerDialog(self.window()).exec_()

    def _choose_dir(self):
        from PyQt5.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, self.tr("选择模型文件夹"), str(models_root()))
        if folder:
            cfg.set(cfg.local_models_dir, folder)
            self.dir_card.setContent(folder)
