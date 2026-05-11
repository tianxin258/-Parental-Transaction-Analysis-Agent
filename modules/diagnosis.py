"""模块3：问题诊断（评价层）

基于模块2 的观察结果做因果归因 + 找出致命失误 + 替代话术。
一次 LLM 调用，输入是模块2 的输出，不做重新分析。
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from modules.dialogue_parser import call_claude, DialogueStructure
from modules.logger import setup_logger

logger = setup_logger()

# ── 数据结构 ──

@dataclass
class DiagnosisReport:
    real_blocker: str        # 家长真正的决策阻碍（因果解释）
    blocker_evidence: str    # 引用 Module2 证据
    fatal_mistake: str       # 最关键的一个失误
    why_fatal: str           # 为什么对这个家长类型致命
    better_response: str     # 反事实话术


# ── LLM Prompt ──

_DIAGNOSIS_SYSTEM = """你是一个资深销售教练。请基于对话的事实观察，诊断这个销售通话中最关键的问题。

## 你的任务

1. 基于家长画像和对话内容，**推断家长真正的决策阻碍**（不是复述表面标签，而是因果归因——家长为什么犹豫？真正卡在哪里？）
2. 找出老师**最关键的一个失误**——不要列很多，只找最致命的那个
3. 给出**反事实话术**——如果当时这样说，会更好

## 规则

- 所有判断必须引用提供的观察证据，不能凭空猜测
- 不要重新分析家长画像（已经提供），只需做因果解释
- 失误判断要结合家长类型——同样的动作对不同类型家长效果完全不同
- better_response 必须是具体的、可以直接说出口的话术

返回严格 JSON（只输出 JSON，不要有任何解释或前缀）：
{
  "real_blocker": "家长真正卡住的原因（因果归因）",
  "blocker_evidence": "引用观察证据来支撑上述判断",
  "fatal_mistake": "老师最关键的一个失误",
  "why_fatal": "为什么对这个家长类型来说这个失误是致命的",
  "better_response": "当时如果这样说，效果会更好（具体话术）"
}"""


def _format_utterances(utterances: list[dict]) -> str:
    """格式化为编号文本。"""
    return "\n".join(f"[{i}] {u['text'][:200]}" for i, u in enumerate(utterances))


# ── 错误日志 ──

_ERROR_DIR = Path(__file__).resolve().parent.parent / "logs"
_ERROR_LOG = _ERROR_DIR / "llm_errors.json"


def _log_error(context: str, error: str) -> None:
    _ERROR_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "diagnosis",
        "context": context,
        "error": error,
    }
    records = []
    if _ERROR_LOG.exists():
        try:
            records = json.loads(_ERROR_LOG.read_text("utf-8"))
        except json.JSONDecodeError:
            records = []
    records.append(entry)
    _ERROR_LOG.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")


# ── 主入口 ──

def diagnose(ds: DialogueStructure) -> DiagnosisReport:
    """模块3 主入口：基于模块2 的观察结果，诊断最关键的失误。

    Args:
        ds: 模块2 输出的 DialogueStructure

    Returns:
        DiagnosisReport: 家长决策阻碍 + 致命失误 + 替代话术
    """
    logger.info("=" * 50)
    logger.info("模块3：问题诊断（评价层）")
    logger.info("=" * 50)

    # 构建输入上下文
    parent_info = (
        f"家长类型：{ds.parent_profile.primary_type}\n"
        f"家长顾虑：{', '.join(ds.parent_profile.concerns) if ds.parent_profile.concerns else '未知'}\n"
        f"关键话术：{', '.join(ds.parent_profile.key_phrases[:5]) if ds.parent_profile.key_phrases else '无'}"
    )

    stage_info = (
        f"当前阶段：{ds.stage.stage} ({ds.stage.coverage})\n"
        f"判断依据：{'; '.join(ds.stage.evidence[:5]) if ds.stage.evidence else '无'}"
    )

    teacher_info = (
        f"老师沟通风格：{ds.teacher.speaking_style}\n"
        f"行为模式：\n" + "\n".join(f"  - {p}" for p in ds.teacher.patterns) + "\n"
        f"关键话术：{', '.join(ds.teacher.key_phrases[:5]) if ds.teacher.key_phrases else '无'}"
    )

    dialogue_snippet = (
        f"老师发言（共{len(ds.teacher_utterances)}段）：\n{_format_utterances(ds.teacher_utterances)}\n\n"
        f"家长发言（共{len(ds.parent_utterances)}段）：\n{_format_utterances(ds.parent_utterances)}"
    )

    user_content = (
        f"## 家长画像（来自模块2，不需重新分析）\n{parent_info}\n\n"
        f"## 成交阶段\n{stage_info}\n\n"
        f"## 老师行为观察\n{teacher_info}\n\n"
        f"## 完整对话\n{dialogue_snippet}"
    )

    logger.info("诊断中（1 次 LLM 调用）...")
    result = call_claude(_DIAGNOSIS_SYSTEM, user_content, max_tokens=2048)

    if not result:
        logger.error("诊断 LLM 调用失败")
        return DiagnosisReport(
            real_blocker="LLM 调用失败",
            blocker_evidence="",
            fatal_mistake="LLM 调用失败",
            why_fatal="",
            better_response="",
        )

    logger.info(f"  决策阻碍：{result.get('real_blocker', '未知')[:60]}...")
    logger.info(f"  致命失误：{result.get('fatal_mistake', '未知')[:60]}...")
    logger.info("模块3 完成\n")

    return DiagnosisReport(
        real_blocker=result.get("real_blocker", "未知"),
        blocker_evidence=result.get("blocker_evidence", ""),
        fatal_mistake=result.get("fatal_mistake", "未知"),
        why_fatal=result.get("why_fatal", ""),
        better_response=result.get("better_response", ""),
    )
