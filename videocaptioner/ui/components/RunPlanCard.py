"""「处理方案」面板：开始之前把整条流水线要做的事摆出来，一处修改，全局生效。

原版里识别引擎、断句、校正、翻译、合成方式分散在设置页、各个工具栏和齿轮弹窗里，
点开始后会一路自动跑到烧录视频，中途看不到会发生什么。
这里把每一步都绑定到同一份配置（cfg），并同步通知其它页面（signalBus）。
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    HyperlinkButton,
    PrimaryPushButton,
    StrongBodyLabel,
    SwitchButton,
)

from videocaptioner.core.entities import (
    LOCAL_ASR_MODEL_LABELS,
    LOCAL_DEVICE_LABELS,
    LLMServiceEnum,
    TranscribeModelEnum,
    TranslatorServiceEnum,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.utils.platform_utils import get_available_transcribe_models
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.common.signal_bus import signalBus

SPLIT_MODES = [
    ("rule", "按标点和停顿断句（本地，免 LLM）"),
    ("llm", "LLM 智能断句（需要配置 LLM）"),
]
VIDEO_MODES = [
    ("none", "只出字幕文件（最快）"),
    ("soft", "软字幕：封装进视频，几秒完成"),
    ("hard", "硬字幕：烧进画面，需要重新编码，长视频很慢"),
]

_LLM_KEY_ITEMS = {
    LLMServiceEnum.OPENAI: "openai_api_key",
    LLMServiceEnum.SILICON_CLOUD: "silicon_cloud_api_key",
    LLMServiceEnum.DEEPSEEK: "deepseek_api_key",
    LLMServiceEnum.OLLAMA: "ollama_api_key",
    LLMServiceEnum.LM_STUDIO: "lm_studio_api_key",
    LLMServiceEnum.GEMINI: "gemini_api_key",
    LLMServiceEnum.CHATGLM: "chatglm_api_key",
}


def llm_configured() -> bool:
    item = _LLM_KEY_ITEMS.get(cfg.llm_service.value)
    return bool(item and str(getattr(cfg, item).value).strip())


def llm_needed() -> bool:
    return (
        cfg.need_split.value
        or cfg.need_optimize.value
        or (cfg.need_translate.value and cfg.translator_service.value == TranslatorServiceEnum.OPENAI)
    )


def video_mode() -> str:
    if not cfg.need_video.value:
        return "none"
    return "soft" if cfg.soft_subtitle.value else "hard"


def engine_detail() -> str:
    model = cfg.transcribe_model.value
    if model == TranscribeModelEnum.LOCAL:
        name = LOCAL_ASR_MODEL_LABELS[cfg.local_model.value].split("（")[0]
        dev = LOCAL_DEVICE_LABELS[cfg.local_device.value].split("（")[0]
        return f"{name} · {dev} · {cfg.transcribe_language.value.value}"
    if model in (TranscribeModelEnum.BIJIAN, TranscribeModelEnum.JIANYING, TranscribeModelEnum.WHISPER_API):
        return "云端识别，会上传音频"
    return "本地 Whisper"


def plan_summary() -> str:
    """一行文字描述当前方案（批量页等处显示）。"""
    parts = [f"识别：{cfg.transcribe_model.value.value.replace(' ✨', '')}（{engine_detail()}）"]
    parts.append("断句：LLM" if cfg.need_split.value else "断句：标点+停顿")
    parts.append("校正：开" if cfg.need_optimize.value else "校正：关")
    if cfg.need_translate.value:
        parts.append(f"翻译：{cfg.translator_service.value.value} → {cfg.target_language.value.value}")
    else:
        parts.append("翻译：关")
    parts.append(dict(VIDEO_MODES)[video_mode()].split("：")[0].split("（")[0])
    return "  ·  ".join(parts)


class RunPlanCard(CardWidget):
    """处理方案面板。"""

    startRequested = pyqtSignal()
    openTranscribeSettings = pyqtSignal()
    openLLMSettings = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._file_stem: Optional[str] = None
        self._build()
        self._load()
        self._connect()

    # ---------------- UI ----------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)
        title = StrongBodyLabel(self.tr("处理方案（点开始前确认，改动会同步到其它页面）"), self)
        root.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        root.addLayout(grid)

        def label(text):
            lb = BodyLabel(text, self)
            lb.setMinimumWidth(70)
            return lb

        # ① 识别
        self.engine_combo = ComboBox(self)
        self._engines = get_available_transcribe_models()
        for m in self._engines:
            self.engine_combo.addItem(m.value)
        self.engine_detail = CaptionLabel("", self)
        self.engine_link = HyperlinkButton("", self.tr("识别设置"), self)
        grid.addWidget(label(self.tr("① 语音识别")), 0, 0)
        grid.addWidget(self.engine_combo, 0, 1)
        grid.addWidget(self.engine_detail, 0, 2)
        grid.addWidget(self.engine_link, 0, 3)

        # ② 断句
        self.split_combo = ComboBox(self)
        for _, text in SPLIT_MODES:
            self.split_combo.addItem(text)
        grid.addWidget(label(self.tr("② 断句")), 1, 0)
        grid.addWidget(self.split_combo, 1, 1, 1, 2)

        # ③ 校正
        self.optimize_switch = SwitchButton(self)
        self.optimize_switch.setOnText(self.tr("开"))
        self.optimize_switch.setOffText(self.tr("关"))
        grid.addWidget(label(self.tr("③ 字幕校正")), 2, 0)
        grid.addWidget(self.optimize_switch, 2, 1)
        grid.addWidget(CaptionLabel(self.tr("用 LLM 修错别字、统一术语和标点"), self), 2, 2)

        # ④ 翻译
        self.translate_switch = SwitchButton(self)
        self.translate_switch.setOnText(self.tr("开"))
        self.translate_switch.setOffText(self.tr("关"))
        self.translator_combo = ComboBox(self)
        for t in TranslatorServiceEnum:
            self.translator_combo.addItem(t.value)
        self.target_combo = ComboBox(self)
        for lang in TargetLanguage:
            self.target_combo.addItem(lang.value)
        self.target_combo.setMaxVisibleItems(10)
        trans_box = QHBoxLayout()
        trans_box.setSpacing(8)
        trans_box.addWidget(self.translate_switch)
        trans_box.addWidget(self.translator_combo)
        trans_box.addWidget(BodyLabel("→", self))
        trans_box.addWidget(self.target_combo)
        trans_box.addStretch(1)
        grid.addWidget(label(self.tr("④ 翻译")), 3, 0)
        grid.addLayout(trans_box, 3, 1, 1, 3)

        # ⑤ 视频
        self.video_combo = ComboBox(self)
        for _, text in VIDEO_MODES:
            self.video_combo.addItem(text)
        grid.addWidget(label(self.tr("⑤ 视频")), 4, 0)
        grid.addWidget(self.video_combo, 4, 1, 1, 2)
        grid.setColumnStretch(2, 1)

        # 底部：输出位置 + 提示 + 开始
        bottom = QHBoxLayout()
        info = QVBoxLayout()
        info.setSpacing(2)
        self.output_label = CaptionLabel("", self)
        self.output_label.setMinimumWidth(300)
        self.warn_label = CaptionLabel("", self)
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #E8A33D;")
        self.llm_link = HyperlinkButton("", self.tr("去设置 LLM"), self)
        self.llm_link.hide()
        info.addWidget(self.output_label)
        warn_row = QHBoxLayout()
        warn_row.addWidget(self.warn_label, 1)
        warn_row.addWidget(self.llm_link)
        info.addLayout(warn_row)
        bottom.addLayout(info, 1)
        self.start_button = PrimaryPushButton(self.tr("开始处理"), self)
        self.start_button.setFixedWidth(140)
        self.start_button.setEnabled(False)
        bottom.addWidget(self.start_button, 0, Qt.AlignBottom)  # type: ignore
        root.addLayout(bottom)

        for combo in (self.engine_combo, self.split_combo, self.video_combo):
            combo.setMinimumWidth(260)
        self.translator_combo.setMinimumWidth(120)
        self.target_combo.setMinimumWidth(120)

    # ---------------- cfg <-> UI ----------------
    def _load(self):
        self._syncing = True
        try:
            model = cfg.transcribe_model.value
            if model in self._engines:
                self.engine_combo.setCurrentIndex(self._engines.index(model))
            self.split_combo.setCurrentIndex(1 if cfg.need_split.value else 0)
            self.optimize_switch.setChecked(cfg.need_optimize.value)
            self.translate_switch.setChecked(cfg.need_translate.value)
            self.translator_combo.setCurrentText(cfg.translator_service.value.value)
            self.target_combo.setCurrentText(cfg.target_language.value.value)
            self.translator_combo.setEnabled(cfg.need_translate.value)
            self.target_combo.setEnabled(cfg.need_translate.value)
            self.video_combo.setCurrentIndex([k for k, _ in VIDEO_MODES].index(video_mode()))
            self.engine_detail.setText(engine_detail())
            self._refresh_hints()
        finally:
            self._syncing = False

    def _refresh_hints(self):
        work = Path(cfg.work_dir.value)
        target = work / self._file_stem if self._file_stem else work / "<文件名>"
        self.output_label.setText(self.tr("输出位置：") + str(target))
        warns = []
        if llm_needed() and not llm_configured():
            warns.append(self.tr("你开启了需要 LLM 的步骤，但还没填 LLM 的 API Key。"))
        if cfg.transcribe_model.value in (TranscribeModelEnum.BIJIAN, TranscribeModelEnum.JIANYING):
            warns.append(self.tr("B/J 接口是云端免费服务：会上传音频，且每 12 小时限 6 小时音频。"))
        if video_mode() == "hard":
            warns.append(self.tr("硬字幕会重新编码整段视频，几小时的直播录像要很久。"))
        self.warn_label.setText("\n".join(warns))
        self.warn_label.setVisible(bool(warns))
        self.llm_link.setVisible(llm_needed() and not llm_configured())

    def _connect(self):
        self.engine_combo.currentIndexChanged.connect(self._on_engine)
        self.split_combo.currentIndexChanged.connect(self._on_split)
        self.optimize_switch.checkedChanged.connect(self._on_optimize)
        self.translate_switch.checkedChanged.connect(self._on_translate)
        self.translator_combo.currentIndexChanged.connect(self._on_translator)
        self.target_combo.currentIndexChanged.connect(self._on_target)
        self.video_combo.currentIndexChanged.connect(self._on_video)
        self.start_button.clicked.connect(self.startRequested)
        self.engine_link.clicked.connect(self.openTranscribeSettings)
        self.llm_link.clicked.connect(self.openLLMSettings)

        # 其它页面改了配置，这里跟着变
        for item in (
            cfg.transcribe_model,
            cfg.local_model,
            cfg.local_device,
            cfg.transcribe_language,
            cfg.need_split,
            cfg.need_optimize,
            cfg.need_translate,
            cfg.translator_service,
            cfg.target_language,
            cfg.need_video,
            cfg.soft_subtitle,
            cfg.llm_service,
            cfg.work_dir,
        ):
            item.valueChanged.connect(lambda *_: None if self._syncing else self._load())
        for name in _LLM_KEY_ITEMS.values():
            getattr(cfg, name).valueChanged.connect(lambda *_: self._refresh_hints())

    def _set(self, item, value):
        self._syncing = True
        try:
            cfg.set(item, value)
        finally:
            self._syncing = False

    def _on_engine(self, index: int):
        if self._syncing or index < 0:
            return
        model = self._engines[index]
        self._set(cfg.transcribe_model, model)
        signalBus.transcription_model_changed.emit(model.value)
        self._load()

    def _on_split(self, index: int):
        if self._syncing:
            return
        self._set(cfg.need_split, SPLIT_MODES[index][0] == "llm")
        self._refresh_hints()

    def _on_optimize(self, checked: bool):
        if self._syncing:
            return
        self._set(cfg.need_optimize, checked)
        signalBus.subtitle_optimization_changed.emit(checked)
        self._refresh_hints()

    def _on_translate(self, checked: bool):
        if self._syncing:
            return
        self._set(cfg.need_translate, checked)
        signalBus.subtitle_translation_changed.emit(checked)
        self.translator_combo.setEnabled(checked)
        self.target_combo.setEnabled(checked)
        self._refresh_hints()

    def _on_translator(self, index: int):
        if self._syncing or index < 0:
            return
        self._set(cfg.translator_service, list(TranslatorServiceEnum)[index])
        self._refresh_hints()

    def _on_target(self, index: int):
        if self._syncing or index < 0:
            return
        lang = list(TargetLanguage)[index]
        self._set(cfg.target_language, lang)
        signalBus.target_language_changed.emit(lang.value)

    def _on_video(self, index: int):
        if self._syncing:
            return
        mode = VIDEO_MODES[index][0]
        need_video = mode != "none"
        soft = mode != "hard"
        self._set(cfg.need_video, need_video)
        self._set(cfg.soft_subtitle, soft)
        signalBus.need_video_changed.emit(need_video)
        signalBus.soft_subtitle_changed.emit(soft)
        self._refresh_hints()

    # ---------------- 外部接口 ----------------
    def set_input(self, path_or_url: str):
        text = path_or_url.strip()
        if text and Path(text).is_file():
            self._file_stem = Path(text).stem
        else:
            self._file_stem = None
        self.start_button.setEnabled(bool(text))
        self._refresh_hints()

    def set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.start_button.setText(self.tr("处理中…") if running else self.tr("开始处理"))


class PlanSummaryBar(QWidget):
    """一行方案摘要 + 「修改」链接（批量页用）。"""

    editRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = CaptionLabel("", self)
        self.label.setWordWrap(True)
        self.link = HyperlinkButton("", self.tr("修改方案"), self)
        lay.addWidget(BodyLabel(self.tr("每个文件按此方案处理："), self))
        lay.addWidget(self.label, 1)
        lay.addWidget(self.link)
        self.link.clicked.connect(self.editRequested)
        for item in (
            cfg.transcribe_model,
            cfg.local_model,
            cfg.local_device,
            cfg.transcribe_language,
            cfg.need_split,
            cfg.need_optimize,
            cfg.need_translate,
            cfg.translator_service,
            cfg.target_language,
            cfg.need_video,
            cfg.soft_subtitle,
        ):
            item.valueChanged.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self):
        self.label.setText(plan_summary())
