"""字级时间戳处理：把识别结果切成「字/词」单元、对齐、再按标点和停顿组句。

单元（Unit）：中文一个字、英文一个词、数字一串、标点一个。
- SenseVoice 是 CTC 模型，每个 token 有起始时间，可以直接得到单元时间。
- Fun-ASR-Nano 是大模型解码，自带的时间戳是均匀摊的，不可靠；
  所以用同一段音频的 SenseVoice 结果做时间参考，把 Nano 的文字对齐上去。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Sequence

CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_UNIT_RE = re.compile(
    rf"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)*"  # 英文词
    r"|\d+(?:[.:]\d+)*%?"  # 数字
    rf"|[{CJK}]"  # 中日韩单字
    r"|[^\s]"  # 其它单个符号（标点等）
)
_LATIN_RE = re.compile(r"^[A-Za-zÀ-ɏ'’]+$")
_DIGIT_RE = re.compile(r"^\d+(?:[.:]\d+)*%?$")
_CJK_RE = re.compile(rf"^[{CJK}]$")

SENTENCE_END = set("。！？!?；;…")
SOFT_BREAK = set("，,、：:")
KEEP_AT_END = set("？！?!…")  # 字幕行尾保留的标点


@dataclass
class Unit:
    text: str
    start: float  # 秒
    end: float
    kind: str  # cjk | word | num | punct

    @property
    def is_punct(self) -> bool:
        return self.kind == "punct"


def unit_kind(text: str) -> str:
    if _CJK_RE.match(text):
        return "cjk"
    if _LATIN_RE.match(text):
        return "word"
    if _DIGIT_RE.match(text):
        return "num"
    return "punct"


def split_text(text: str) -> List[tuple]:
    """把一段文字切成 (单元文字, 类型)。"""
    return [(m.group(), unit_kind(m.group())) for m in _UNIT_RE.finditer(text)]


def tokens_to_units(
    tokens: Sequence[str], starts: Sequence[float], seg_end: float
) -> List[Unit]:
    """把 SenseVoice 的 BPE token + 起始时间合并成单元。

    规则：拉丁字母片段如果没有前导空格、且紧跟在另一个拉丁片段后面，就拼成同一个词。
    """
    units: List[Unit] = []
    n = len(tokens)
    for i, (tok, st) in enumerate(zip(tokens, starts)):
        nxt = starts[i + 1] if i + 1 < n else seg_end
        if not tok.strip():
            continue
        lead_space = tok[:1].isspace()
        piece = tok.strip()
        parts = split_text(piece)
        if not parts:
            continue
        # 片段内部可能有多个单元（如多个汉字），时间平均分
        step = max(nxt - st, 0.0) / len(parts)
        for k, (txt, kind) in enumerate(parts):
            s = st + step * k
            e = st + step * (k + 1)
            prev = units[-1] if units else None
            glue = (
                k == 0
                and not lead_space
                and prev is not None
                and prev.kind == kind
                and kind in ("word", "num")
            )
            if glue and prev is not None:
                prev.text += txt
                prev.end = e
            else:
                units.append(Unit(txt, s, e, kind))
    _cap_durations(units, seg_end)
    return units


def _cap_durations(units: List[Unit], seg_end: float, max_cjk: float = 0.6) -> None:
    """CTC 只给起点，终点用下一个单元起点；遇到长停顿时不要把字拖得太长。"""
    for u in units:
        limit = max_cjk if u.kind == "cjk" else max_cjk * max(1, len(u.text) / 3)
        if u.end - u.start > limit:
            u.end = u.start + limit
        u.end = min(u.end, seg_end)
        if u.end < u.start:
            u.end = u.start


def _norm(u_text: str) -> str:
    return u_text.lower()


def _spread(targets: List[tuple], t0: float, t1: float) -> List[Unit]:
    """把一串没有对齐上的单元平均摊到 [t0, t1]。标点不占时长。"""
    weights = [0.0 if k == "punct" else (1.0 if k == "cjk" else max(1.0, len(t) / 3)) for t, k in targets]
    total = sum(weights) or 1.0
    out, cur = [], t0
    span = max(t1 - t0, 0.0)
    for (txt, kind), w in zip(targets, weights):
        d = span * w / total
        out.append(Unit(txt, cur, cur + d, kind))
        cur += d
    return out


def align_text(text: str, ref: List[Unit], seg_start: float, seg_end: float) -> List[Unit]:
    """把任意文字（如 Nano 的识别结果）按参考单元的时间对齐。"""
    targets = split_text(text)
    if not targets:
        return []
    ref_words = [u for u in ref if not u.is_punct]
    if not ref_words:
        return _spread(targets, seg_start, seg_end)

    tgt_idx = [i for i, (_, k) in enumerate(targets) if k != "punct"]
    a = [_norm(u.text) for u in ref_words]
    b = [_norm(targets[i][0]) for i in tgt_idx]
    timed: dict = {}
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for di in range(i2 - i1):
                r = ref_words[i1 + di]
                timed[tgt_idx[j1 + di]] = (r.start, r.end)
        elif op in ("replace", "insert"):
            if op == "replace":
                t0, t1 = ref_words[i1].start, ref_words[i2 - 1].end
            else:
                t0 = ref_words[i1 - 1].end if i1 > 0 else seg_start
                t1 = ref_words[i1].start if i1 < len(ref_words) else seg_end
            spread = _spread([targets[tgt_idx[j]] for j in range(j1, j2)], t0, max(t1, t0))
            for j, u in zip(range(j1, j2), spread):
                timed[tgt_idx[j]] = (u.start, u.end)

    units: List[Unit] = []
    last_end = seg_start
    for i, (txt, kind) in enumerate(targets):
        if i in timed:
            s, e = timed[i]
            last_end = e
        else:  # 标点：贴在前一个单元末尾
            s = e = last_end
        units.append(Unit(txt, s, e, kind))
    return units


def join_units(units: Sequence[Unit]) -> str:
    """拼回文字：英文词之间加空格，中英之间不加。"""
    out = ""
    prev_kind = ""
    for u in units:
        if out and u.kind in ("word", "num") and (
            prev_kind in ("word", "num") or out[-1] in ".,!?;:"
        ):
            out += " "
        out += u.text
        prev_kind = u.kind
    return out


def _weight(u: Unit) -> float:
    if u.kind == "cjk":
        return 1.0
    if u.kind in ("word", "num"):
        return 1.5
    return 0.0


def _strip_trailing_punct(units: List[Unit]) -> List[Unit]:
    while units and units[-1].is_punct and units[-1].text not in KEEP_AT_END:
        units = units[:-1]
    while units and units[0].is_punct:
        units = units[1:]
    return units


def build_sentences(
    units: List[Unit],
    max_len: int = 25,
    pause_split: float = 0.8,
    min_len: int = 6,
) -> List[List[Unit]]:
    """按「句末标点 > 长停顿 > 逗号 > 字数上限」把单元组成字幕句。

    max_len 以中文字数计，英文一个词按 1.5 个字算。
    """
    sentences: List[List[Unit]] = []
    cur: List[Unit] = []
    length = 0.0

    def flush():
        nonlocal cur, length
        line = _strip_trailing_punct(cur)
        if line:
            sentences.append(line)
        cur, length = [], 0.0

    for u in units:
        if cur and not u.is_punct:
            gap = u.start - cur[-1].end
            if gap >= pause_split and length >= 2:
                flush()
        if not cur and u.is_punct:
            continue
        # 超长：在当前句里找最好的断点（最近的逗号或最大停顿）
        if not u.is_punct and length + _weight(u) > max_len:
            cut = _best_cut(cur, min_len)
            head, tail = cur[:cut], cur[cut:]
            cur = head
            flush()
            cur = tail
            length = sum(_weight(x) for x in cur)
        cur.append(u)
        length += _weight(u)
        if u.is_punct and u.text in SENTENCE_END:
            flush()
        elif u.is_punct and u.text in SOFT_BREAK and length >= max_len * 0.6:
            flush()
    flush()
    return sentences


def _best_cut(cur: List[Unit], min_len: int) -> int:
    """返回切分位置（前半句长度）。优先最靠后的逗号，其次最大停顿。"""
    n = len(cur)
    acc = 0.0
    best_comma = -1
    best_gap, best_gap_pos = -1.0, n
    for i, u in enumerate(cur):
        acc += _weight(u)
        if acc < min_len or i == n - 1:
            continue
        if u.is_punct and u.text in SOFT_BREAK | SENTENCE_END:
            best_comma = i + 1
        nxt = cur[i + 1]
        gap = nxt.start - u.end
        if not nxt.is_punct and gap > best_gap:
            best_gap, best_gap_pos = gap, i + 1
    if best_comma > 0:
        return best_comma
    return best_gap_pos if best_gap_pos < n else n


_WEEKDAY_FIX = re.compile(r"(礼拜|星期|周)([1-6])(?!\d)")
_WEEKDAY_ZH = dict(zip("123456", "一二三四五六"))


def fix_itn(text: str) -> str:
    """SenseVoice 的数字规整会把「礼拜二」变成「礼拜2」，这里改回来。"""
    return _WEEKDAY_FIX.sub(lambda m: m.group(1) + _WEEKDAY_ZH[m.group(2)], text)


def fix_units_itn(units: List[Unit]) -> List[Unit]:
    """单元级的 fix_itn：「礼」「拜」「2」→「礼」「拜」「二」。"""
    for i, u in enumerate(units):
        if u.kind == "num" and u.text in _WEEKDAY_ZH and i > 0:
            prev = "".join(x.text for x in units[max(0, i - 2) : i])
            if prev.endswith(("礼拜", "星期", "周")):
                u.text = _WEEKDAY_ZH[u.text]
                u.kind = "cjk"
    return units
