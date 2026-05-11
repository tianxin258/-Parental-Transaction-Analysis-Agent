"""模块2：对话结构拆解（观察层）

串行推理链：角色分离 → 家长画像 → 阶段判断 → 老师行为观察。
只做事实观察，不做价值判断。价值判断留给模块3。
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from modules.logger import setup_logger

logger = setup_logger()

# ── LLM 配置（环境变量优先） ──

def _load_llm_config() -> tuple[str, str, str]:
    """从环境变量读取 LLM 配置。返回 (base_url, api_key, model)。"""
    return (
        os.environ.get("LLM_BASE_URL", "http://localhost:18789/v1"),
        os.environ.get("LLM_API_KEY", "dummy"),
        os.environ.get("LLM_MODEL", "deepseek-chat"),
    )


_BASE_URL, _API_KEY, _MODEL = _load_llm_config()

# ── 数据结构 ──

@dataclass
class ParentProfile:
    """家长画像（观察层：只描述事实，不评价）。"""
    primary_type: str          # 焦虑型 / 价格敏感 / 拖延型 / 不信任型
    confidence: str            # high / medium / low
    key_phrases: list[str]     # 关键话术引用
    concerns: list[str]        # 核心顾虑

@dataclass
class StageAnalysis:
    """成交阶段判断。"""
    stage: str                 # 建立信任 / 挖需求 / 放大痛点 / 提供方案 / 推成交
    evidence: list[str]        # 判断依据
    coverage: str              # 已完成 / 部分完成 / 已进入

@dataclass
class TeacherObservations:
    """老师行为观察（观察层：只描述行为，不评价对错）。"""
    patterns: list[str]        # 客观行为模式
    speaking_style: str        # 讲解型 / 提问型 / 引导型 / 推销型
    key_phrases: list[str]     # 关键话术

@dataclass
class DialogueStructure:
    """模块2 完整输出。"""
    teacher_utterances: list[dict]
    parent_utterances: list[dict]
    parent_profile: ParentProfile
    stage: StageAnalysis
    teacher: TeacherObservations


# ── LLM 工具 ──

_LLM_DIR = Path(__file__).resolve().parent.parent / "logs"
_ERROR_LOG = _LLM_DIR / "llm_errors.json"


def call_llm(system_prompt: str, user_content: str, temperature: float = 0.1, max_tokens: int = 1024) -> dict:
    """统一 LLM 入口（OpenAI 兼容接口），通过 OpenClaw Gateway 调用。
    要求 JSON 输出，解析失败自动重试一次（2x max_tokens）。"""
    client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)

    def _parse_response(text: str) -> dict:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    def _call_and_parse(tokens: int, conciseness_hint: bool = False) -> dict:
        sp = system_prompt + (" Keep your JSON output concise." if conciseness_hint else "")
        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_response(text)

    try:
        return _call_and_parse(max_tokens)
    except Exception as e:
        first_error = str(e)
        logger.warning(f"LLM 首次调用失败，重试（2x max_tokens）: {first_error[:100]}")
        try:
            return _call_and_parse(max_tokens * 2, conciseness_hint=True)
        except Exception as e2:
            _log_error(system_prompt[:120], f"first: {first_error[:80]} | retry: {str(e2)[:80]}")
            return {}


def _log_error(context: str, error: str) -> None:
    """记录 LLM 错误到日志文件。"""
    _LLM_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    logger.error(f"LLM 调用失败（已记录）: {error[:150]}")


# ── 对话格式化 ──

def _format_segments(segments: list[dict]) -> str:
    """将 segments 列表格式化为编号对话文本，供 LLM 分析。"""
    lines = []
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        lines.append(f"[{i} | {speaker} | {start:.0f}s-{end:.0f}s] {text}")
    return "\n".join(lines)


def _format_utterances(utterances: list[dict]) -> str:
    """将某角色的 utterances 格式化为纯文本列表。"""
    return "\n".join(
        f"[{i}] {u['text']}" for i, u in enumerate(utterances)
    )


# ── 说话时长回退 ──

def _fallback_roles(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """按说话总时长分离角色（LLM 失败时的回退方案）。

    假设：老师说话时长 > 家长说话时长（销售场景中老师更主导对话）。
    """
    speakers: dict[str, list[dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        speakers.setdefault(speaker, []).append(seg)

    def total_duration(utts):
        return sum(u.get("end", 0) - u.get("start", 0) for u in utts)

    sorted_speakers = sorted(speakers.items(), key=lambda kv: total_duration(kv[1]), reverse=True)
    teacher_utts = sorted_speakers[0][1] if len(sorted_speakers) > 0 else []
    parent_utts = sorted_speakers[1][1] if len(sorted_speakers) > 1 else []
    logger.warning("使用说话时长回退方案分离角色")
    return teacher_utts, parent_utts


# ── 步骤1：角色分离 ──

_SEPARATE_SYSTEM = """你是一个对话分析助手。你的任务是根据对话内容的语义特征，判断每段话是"老师/销售顾问"说的还是"家长"说的。

判断依据（按重要性排序）：
1. 说话内容：谁在介绍课程、询问成绩、推荐报课、解释优惠、跟进决策 → 老师
2. 说话内容：谁在描述孩子情况、表达顾虑、说需要和孩子商量、提到经济考虑 → 家长
3. 说话立场：谁在"卖"（老师），谁在"买/考虑买"（家长）

说话时长仅作微弱参考，不作为主要判断依据。

返回严格 JSON（只输出 JSON，不要有任何解释或前缀）：
{"teacher_indices": [0, 1, 3, ...], "parent_indices": [2, 4, ...], "reasoning": "一句话说明判断逻辑"}"""


def _separate_roles(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """步骤1：用语义特征区分老师和家长（LLM 驱动）。"""
    dialogue_text = _format_segments(segments)
    result = call_llm(_SEPARATE_SYSTEM, dialogue_text, max_tokens=4096)

    teacher_idx = result.get("teacher_indices", [])
    parent_idx = result.get("parent_indices", [])

    if not teacher_idx and not parent_idx:
        logger.warning("角色分离 LLM 返回空，回退到说话时长方案")
        return _fallback_roles(segments)

    teacher_utts = [segments[i] for i in teacher_idx if i < len(segments)]
    parent_utts = [segments[i] for i in parent_idx if i < len(segments)]

    logger.info(f"  角色分离完成：老师 {len(teacher_utts)} 段，家长 {len(parent_utts)} 段")
    return teacher_utts, parent_utts


# ── 步骤2：家长画像 ──

_PARENT_SYSTEM = """你是一个对话分析助手。请根据家长的发言内容，提取家长的画像特征。

## 家长类型判断标准

- **焦虑型**：反复追问细节、担心孩子跟不上、强调时间紧迫、语气急切
- **价格敏感**：主动询价、对比价格、提到经济压力、问是否有优惠
- **拖延型**：说需要再考虑、要和孩子商量、不明确表态、回避决策
- **不信任型**：质疑效果、拿其他机构比较、要求保证提分
备注：如果以上类型都不符合该家长的画像特征，你可自行判断，但必须符合事实

## 输出要求

1. 输出**观察事实**，不要评价家长"对/错"
2. key_phrases 必须是原文中出现的表述（或近似原文）
3. 如果证据不足，confidence 设为 "low"

返回严格 JSON（只输出 JSON，不要有任何解释或前缀）：
{
  "primary_type": "焦虑型",
  "confidence": "medium",
  "key_phrases": ["原文话术1", "原文话术2"],
  "concerns": ["顾虑1", "顾虑2"]
}"""


def _analyze_parent(parent_utterances: list[dict]) -> ParentProfile:
    """步骤2：分析家长画像。"""
    if not parent_utterances:
        return ParentProfile(
            primary_type="未知", confidence="low",
            key_phrases=[], concerns=["家长发言过少，无法判断"]
        )

    text = _format_utterances(parent_utterances)
    result = call_llm(_PARENT_SYSTEM, text, max_tokens=2048)

    if not result:
        return ParentProfile(
            primary_type="未知", confidence="low",
            key_phrases=[], concerns=["LLM 调用失败"]
        )

    logger.info(f"  家长画像：{result.get('primary_type', '未知')} (confidence: {result.get('confidence', 'low')})")
    return ParentProfile(
        primary_type=result.get("primary_type", "未知"),
        confidence=result.get("confidence", "low"),
        key_phrases=result.get("key_phrases", []),
        concerns=result.get("concerns", []),
    )


# ── 步骤3：成交阶段判断 ──

_STAGE_SYSTEM = """你是一个销售对话分析助手。请根据完整对话内容，判断当前销售沟通处于哪个阶段。

## 五个阶段

1. **建立信任**：老师自我介绍、确认身份、寒暄、表达关心
2. **挖需求**：询问学生成绩、了解学习情况、问薄弱科目
3. **放大痛点**：强调一轮复习的重要性、指出基础不牢的后果、制造紧迫感
4. **提供方案**：介绍课程安排、说明学习计划、给出具体科目建议
5. **推成交**：报价格、说优惠截止日期、要求当场做决定、约下次跟进时间

## 输出要求

- 一次对话可能跨越多个阶段，请判断**当前推进到的阶段**
- 如果对话尚在早期阶段，coverage 用"已进入"
- 如果该阶段已充分展开，coverage 用"部分完成"
- 如果该阶段已完成并推进到下一阶段，coverage 用"已完成"

返回严格 JSON（只输出 JSON，不要有任何解释或前缀）：
{
  "stage": "放大痛点",
  "evidence": ["依据1", "依据2"], 
  "coverage": "部分完成"
}"""


def _detect_stage(parent_profile: ParentProfile, segments: list[dict]) -> StageAnalysis:
    """步骤3：判断成交阶段（传入家长画像作为上下文）。"""
    profile_text = f"家长类型：{parent_profile.primary_type}，顾虑：{', '.join(parent_profile.concerns) if parent_profile.concerns else '未知'}"
    dialogue_text = _format_segments(segments)
    user_content = f"{profile_text}\n\n完整对话：\n{dialogue_text}"

    result = call_llm(_STAGE_SYSTEM, user_content, max_tokens=2048)

    if not result:
        return StageAnalysis(stage="未知", evidence=[], coverage="未知")

    logger.info(f"  成交阶段：{result.get('stage', '未知')} (coverage: {result.get('coverage', '未知')})")
    return StageAnalysis(
        stage=result.get("stage", "未知"),
        evidence=result.get("evidence", []),
        coverage=result.get("coverage", "未知"),
    )


# ── 步骤4：老师行为观察 ──

_TEACHER_SYSTEM = """你是一个销售对话分析助手。请观察老师/销售顾问在对话中的行为，只做**客观描述**，不做对错评价。

## 禁止使用的评价性语言
- 不要说"做得好""做得不好""应该""不应该""正确""错误""问题在于"
- 不要给建议

## 允许使用的观察性语言
- "老师在XX阶段使用了YY话术"
- "老师通过XX方式回应了家长的YY顾虑"
- "老师采用了XX策略推进对话"

## speaking_style 分类
- **讲解型**：以介绍、解释、说明为主
- **提问型**：以提问、了解情况为主
- **引导型**：通过提问引导家长得出结论
- **推销型**：直接推动成交、强调优惠、制造紧迫感

## key_phrases
提取老师使用频率最高或最关键的话术（不超过5条）

返回严格 JSON（只输出 JSON，不要有任何解释或前缀）：
{
  "patterns": ["观察到的行为模式1", "观察到的行为模式2"],
  "speaking_style": "引导型",
  "key_phrases": ["关键话术1", "关键话术2"]
}"""


def _observe_teacher(
    stage: StageAnalysis,
    parent_profile: ParentProfile,
    teacher_utterances: list[dict],
) -> TeacherObservations:
    """步骤4：观察老师行为（只描述，不评价）。"""
    if not teacher_utterances:
        return TeacherObservations(patterns=[], speaking_style="未知", key_phrases=[])

    context = (
        f"当前成交阶段：{stage.stage}\n"
        f"家长类型：{parent_profile.primary_type}\n"
        f"家长顾虑：{', '.join(parent_profile.concerns) if parent_profile.concerns else '未知'}\n\n"
        f"老师发言：\n{_format_utterances(teacher_utterances)}"
    )

    result = call_llm(_TEACHER_SYSTEM, context, max_tokens=4096)

    if not result:
        return TeacherObservations(patterns=[], speaking_style="未知", key_phrases=[])

    logger.info(f"  老师风格：{result.get('speaking_style', '未知')}，模式数：{len(result.get('patterns', []))}")
    return TeacherObservations(
        patterns=result.get("patterns", []),
        speaking_style=result.get("speaking_style", "未知"),
        key_phrases=result.get("key_phrases", []),
    )


# ── 主入口 ──

def parse_dialogue(segments: list[dict]) -> DialogueStructure:
    """模块2 主入口：串行推理链，一次调用完成全部拆解。

    Args:
        segments: 模块1 输出的带说话人标签的对话片段列表
                  格式：[{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "..."}, ...]

    Returns:
        DialogueStructure: 包含角色分离、家长画像、阶段判断、老师观察
    """
    logger.info("=" * 50)
    logger.info("模块2：对话结构拆解（观察层）")
    logger.info("=" * 50)

    # Step 1: 角色分离
    logger.info("--- Step 1: 角色分离 ---")
    teacher_utts, parent_utts = _separate_roles(segments)

    # Step 2: 家长画像
    logger.info("--- Step 2: 家长画像 ---")
    parent_profile = _analyze_parent(parent_utts)

    # Step 3: 成交阶段判断（传入家长画像）
    logger.info("--- Step 3: 成交阶段 ---")
    stage = _detect_stage(parent_profile, segments)

    # Step 4: 老师行为观察（传入阶段+家长画像）
    logger.info("--- Step 4: 老师行为观察 ---")
    teacher = _observe_teacher(stage, parent_profile, teacher_utts)

    logger.info("模块2 完成\n")
    return DialogueStructure(
        teacher_utterances=teacher_utts,
        parent_utterances=parent_utts,
        parent_profile=parent_profile,
        stage=stage,
        teacher=teacher,
    )
