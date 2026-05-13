"""AI家长成交分析Agent - 主入口

一键跑通全链路: 语音 -> 观察 -> 诊断

用法:
  python main.py --audio data/input/test.mp3          # 完整管线(含语音转录)
  python main.py --segments data/output/test/segments.json  # 跳过转录,直接分析

中间文件保存在 ./data/output/<录音文件名>/
"""

import argparse
import json
import sys
from pathlib import Path

from modules.logger import setup_logger

logger = setup_logger()

OUT_ROOT = Path("data/output")


def _resolve_case_name(args) -> str:
    """从参数推断案例名(录音文件名,不含扩展名)。"""
    if args.audio:
        return Path(args.audio).stem
    # 从 segments 路径推断: data/output/xxx/segments.json -> xxx
    seg_path = Path(args.segments)
    return seg_path.parent.name


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(case: str, data: dict, filename: str) -> Path:
    out_dir = OUT_ROOT / case
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return p


# ---- 管线 ----

def run_module1(audio_path: str, case: str, model_size: str = "base") -> list[dict]:
    """模块1: 语音转文本 + 说话人分离。"""
    from modules.stt import transcribe_with_speakers

    logger.info("")
    logger.info("=" * 40)
    logger.info("模块1: 语音转文本")
    logger.info("=" * 40)

    audio = Path(audio_path)
    if not audio.exists():
        logger.error(f"音频文件不存在: {audio_path}")
        sys.exit(1)

    segments = transcribe_with_speakers(str(audio), model_size)
    save_json(case, segments, "segments.json")
    logger.info(f"segments 已保存 ({len(segments)} 段)")
    return segments


def run_module2(segments: list[dict], case: str) -> dict:
    """模块2: 对话结构拆解(观察层)。"""
    from dataclasses import asdict
    from modules.dialogue_parser import parse_dialogue

    logger.info("")
    logger.info("=" * 40)
    logger.info("模块2: 对话结构拆解")
    logger.info("=" * 40)

    ds = parse_dialogue(segments)
    output = {
        "teacher_utterances": ds.teacher_utterances,
        "parent_utterances": ds.parent_utterances,
        "parent_profile": asdict(ds.parent_profile),
        "stage": asdict(ds.stage),
        "teacher": asdict(ds.teacher),
    }
    save_json(case, output, "dialogue_structure.json")
    return output


def run_module3(ds_data: dict, case: str) -> dict:
    """模块3: 问题诊断(评价层)。"""
    from dataclasses import asdict
    from modules.dialogue_parser import DialogueStructure, ParentProfile, StageAnalysis, TeacherObservations
    from modules.diagnosis import diagnose

    logger.info("")
    logger.info("=" * 40)
    logger.info("模块3: 问题诊断")
    logger.info("=" * 40)

    ds = DialogueStructure(
        teacher_utterances=ds_data["teacher_utterances"],
        parent_utterances=ds_data["parent_utterances"],
        parent_profile=ParentProfile(**ds_data["parent_profile"]),
        stage=StageAnalysis(**ds_data["stage"]),
        teacher=TeacherObservations(**ds_data["teacher"]),
    )

    report = diagnose(ds)
    output = asdict(report)
    save_json(case, output, "diagnosis_report.json")
    return output


# ---- 报告输出 ----

def _wrap(text: str, width: int = 70) -> list[str]:
    lines = []
    while text:
        lines.append(text[:width])
        text = text[width:]
    return lines


def print_report(d2: dict, d3: dict) -> None:
    parent = d2["parent_profile"]
    stage = d2["stage"]
    teacher = d2["teacher"]

    sep = "=" * 60

    print()
    print(sep)
    print("  AI 家长成交分析报告")
    print(sep)

    print(f"  [家长画像]")
    print(f"    类型: {parent['primary_type']} (confidence: {parent['confidence']})")
    for c in parent.get("concerns", []):
        print(f"    顾虑: {c}")

    print(f"\n  [成交阶段] {stage['stage']} ({stage['coverage']})")
    print(f"    老师风格: {teacher['speaking_style']}")

    print(f"\n  [决策阻碍]")
    for line in _wrap(d3["real_blocker"]):
        print(f"    {line}")

    print(f"\n  [致命失误]")
    for line in _wrap(d3["fatal_mistake"]):
        print(f"    {line}")
    print(f"    为何致命: {d3['why_fatal']}")

    print(f"\n  [替代话术]")
    for line in _wrap(d3.get("better_response", "")):
        print(f"    {line}")

    print(sep)
    print()


# ---- 入口 ----

def main():
    parser = argparse.ArgumentParser(description="AI家长成交分析Agent - 一键全链路分析")
    parser.add_argument("--audio", help="通话录音文件路径 (.mp3/.wav/.m4a)")
    parser.add_argument("--segments", help="已有 segments.json,跳过转录直接分析")
    parser.add_argument("--model", default="base", help="已废弃（FunASR 使用 Paraformer，无需指定模型大小），保留兼容")
    parser.add_argument("--context", help="补充背景文本文件(暂未接入模块管线)")
    args = parser.parse_args()

    if not args.audio and not args.segments:
        parser.error("请提供 --audio(录音文件) 或 --segments(已有 segments.json)")

    case = _resolve_case_name(args)
    logger.info(f"案例名称: {case}")

    # 模块1: 语音转录
    if args.audio:
        segments = run_module1(args.audio, case, args.model)
    else:
        seg_path = Path(args.segments)
        if not seg_path.exists():
            logger.error(f"segments 文件不存在: {args.segments}")
            sys.exit(1)
        segments = load_json(seg_path)
        logger.info(f"加载已有 segments ({len(segments)} 段),跳过转录")

    # 模块2: 对话拆解
    d2 = run_module2(segments, case)

    # 模块3: 问题诊断
    d3 = run_module3(d2, case)

    # 输出报告
    print_report(d2, d3)
    logger.info(f"中间文件保存在 {(OUT_ROOT / case).resolve()}")
    logger.info("全链路完成。")


if __name__ == "__main__":
    main()
