"""语音转文本 (WhisperX)

管线：转录 → 对齐 → 说话人分离 → 合并
"""

import os

import pandas as pd
import torch
import torchaudio
import whisperx

from modules.logger import setup_logger

logger = setup_logger()

# ── 全局缓存（延迟加载，只初始化一次） ──
_device = None
_align_model = None
_align_metadata = None
_diarization_pipeline = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _get_align_model(language="zh"):
    global _align_model, _align_metadata
    if _align_model is None:
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=language, device=_get_device()
        )
    return _align_model, _align_metadata


def _get_diarization_pipeline(auth_token=""):
    global _diarization_pipeline
    if _diarization_pipeline is None:
        token = auth_token or os.environ.get("HF_TOKEN", "")
        if not token:
            logger.error(
                "需要 HuggingFace 访问令牌。\n"
                "1. 注册 https://huggingface.co\n"
                "2. 接受协议: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. 生成 token: https://huggingface.co/settings/tokens\n"
                "4. 设置环境变量 HF_TOKEN 或传入 auth_token 参数"
            )
            raise ValueError("缺少 HuggingFace 访问令牌")
        from pyannote.audio import Pipeline
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    return _diarization_pipeline


def transcribe(audio_path: str, model_size: str = "base") -> str:
    """纯转录（不含说话人信息）。"""
    device = _get_device()
    model = whisperx.load_model(model_size, device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language="zh")
    return "".join(seg["text"] for seg in result["segments"])


def transcribe_with_speakers(
    audio_path: str,
    model_size: str = "base",
    language: str = "zh",
    auth_token: str = "",
) -> list[dict]:
    """完整管线：转录 → 对齐 → 说话人分离 → 合并。

    Returns:
        每段格式: {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "..."}
    """
    device = _get_device()

    logger.info("  [1/4] 转录中...")
    model = whisperx.load_model(model_size, device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language=language)

    logger.info("  [2/4] 时间轴对齐中...")
    align_model, align_metadata = _get_align_model(language)
    result = whisperx.align(result["segments"], align_model, align_metadata, audio, device)

    logger.info("  [3/4] 说话人分离中...")
    pipeline = _get_diarization_pipeline(auth_token)
    diar_audio = torchaudio.load(audio_path)
    diarize_output = pipeline(
        {"waveform": diar_audio[0], "sample_rate": diar_audio[1]},
        num_speakers=2,
    )

    logger.info("  [4/4] 合并说话人标签...")
    anno = diarize_output.exclusive_speaker_diarization
    df_rows = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in anno.itertracks(yield_label=True)
    ]
    diarization_df = pd.DataFrame(df_rows)
    result = whisperx.assign_word_speakers(diarization_df, result)

    return [
        {
            "speaker": seg.get("speaker", "UNKNOWN"),
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]


def format_dialogue(segments: list[dict]) -> str:
    """将带说话人的片段格式化为可读对话文本。"""
    return "\n".join(
        f"[{seg['speaker'].replace('SPEAKER_', '👤 ')}] {seg['text']}"
        for seg in segments
    )
