"""语音转文本 (FunASR)

管线：VAD → ASR → 说话人分离 → 标点恢复
使用阿里达摩院 FunASR 生态（Paraformer + CAM++），CPU 友好，中文优化。
"""

from modules.logger import setup_logger

logger = setup_logger()

# ── 全局缓存 ──
_pipeline = None
_pipeline_no_spk = None


def _get_pipeline(with_diarization=True):
    """延迟加载 FunASR pipeline（全局单例）。"""
    global _pipeline, _pipeline_no_spk

    if with_diarization and _pipeline is not None:
        return _pipeline
    if not with_diarization and _pipeline_no_spk is not None:
        return _pipeline_no_spk

    from funasr import AutoModel

    if with_diarization:
        logger.info("  加载 FunASR 全管线（ASR + VAD + 说话人分离 + 标点恢复）...")
        _pipeline = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
        return _pipeline
    else:
        logger.info("  加载 FunASR（ASR + VAD + 标点恢复）...")
        _pipeline_no_spk = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
        return _pipeline_no_spk


def transcribe(audio_path: str, model_size: str = "base") -> str:
    """纯转录（不含说话人信息）。

    Args:
        audio_path: 音频文件路径
        model_size: 已废弃（FunASR 不需要模型大小参数），保留兼容

    Returns:
        完整转录文本
    """
    pipeline = _get_pipeline(with_diarization=False)
    result = pipeline.generate(input=audio_path, batch_size_s=300)

    if result and len(result) > 0:
        return result[0].get("text", "")
    return ""


def transcribe_with_speakers(
    audio_path: str,
    model_size: str = "base",
    language: str = "zh",
    auth_token: str = "",
) -> list[dict]:
    """完整管线：VAD → ASR → 说话人分离 → 标点恢复。

    Args:
        audio_path: 音频文件路径
        model_size: 已废弃（FunASR 不需要模型大小参数），保留兼容
        language: 已废弃（Paraformer 专为中文设计），保留兼容
        auth_token: 已废弃（FunASR 使用 ModelScope，不需要 HF token），保留兼容

    Returns:
        每段格式: {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "..."}
    """
    pipeline = _get_pipeline(with_diarization=True)

    logger.info("  FunASR 全管线转录中（ASR + VAD + 说话人分离 + 标点恢复）...")
    result = pipeline.generate(input=audio_path, batch_size_s=300)

    if not result or len(result) == 0:
        logger.error("FunASR 转录返回空")
        return []

    raw_segments = _parse_result(result[0])
    logger.info(f"  转录完成：{len(raw_segments)} 句")
    segments = _merge_segments(raw_segments)
    logger.info(f"  合并后：{len(segments)} 段（连续同说话人合并）")
    return segments


def _parse_result(result: dict) -> list[dict]:
    """将 FunASR 输出转换为标准 segments 格式。"""
    segments = []

    # FunASR 的 sentence_info 格式示例：
    # [{"spk": 0, "start": 0.0, "end": 2.5, "text": "..."}, ...]
    sentences = (
        result.get("sentence_info")
        or result.get("sentences")
        or []
    )

    if sentences:
        for sent in sentences:
            spk = sent.get("spk", "UNKNOWN")
            if isinstance(spk, (int, float)):
                spk = f"SPEAKER_{int(spk):02d}"
            elif isinstance(spk, str) and spk.isdigit():
                spk = f"SPEAKER_{int(spk):02d}"

            # FunASR 时间戳为毫秒，转为秒以保持与旧格式兼容
            start = sent.get("start", 0) / 1000
            end = sent.get("end", 0) / 1000

            segments.append({
                "speaker": spk,
                "start": round(start, 2),
                "end": round(end, 2),
                "text": sent.get("text", "").strip(),
            })
        return segments

    # 回退：无分段信息，整段输出
    text = result.get("text", "")
    if text:
        segments.append({
            "speaker": "SPEAKER_00",
            "start": 0,
            "end": 0,
            "text": text.strip(),
        })

    return segments


def _merge_segments(segments: list[dict], gap_threshold: float = 2.0) -> list[dict]:
    """合并连续同说话人的片段，减少 LLM 处理的段数。

    FunASR 输出粒度过细（每句话一段，335句→30段左右）。
    此函数将同一说话人连续且间隔 < gap_threshold 秒的片段合并。
    """
    if not segments:
        return []

    merged = []
    current = dict(segments[0])  # 复制第一段作为当前合并段

    for seg in segments[1:]:
        same_speaker = seg["speaker"] == current["speaker"]
        gap = seg["start"] - current["end"]
        close_in_time = gap <= gap_threshold

        if same_speaker and close_in_time:
            # 合并：扩展结束时间，拼接文本
            current["end"] = seg["end"]
            current["text"] += seg["text"]
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)
    return merged


def format_dialogue(segments: list[dict]) -> str:
    """将带说话人的片段格式化为可读对话文本。"""
    return "\n".join(
        f"[{seg['speaker'].replace('SPEAKER_', '👤 ')}] {seg['text']}"
        for seg in segments
    )
