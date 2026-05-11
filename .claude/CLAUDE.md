# AI家长成交分析Agent

教育销售顾问的**销售复盘 + 用户洞察 Agent**。

## Tool 调用（给 Claude Code）

当用户要求分析一通电话录音时，执行：

```bash
cd "E:\documents\AI-practice\AI家长成交分析Agent" && conda run -n ai_agent python main.py --audio "<音频路径>"
```

如果已有 segments.json，跳过转录（更快）：

```bash
cd "E:\documents\AI-practice\AI家长成交分析Agent" && conda run -n ai_agent python main.py --segments "data/output/<案例名>/segments.json"
```

运行完成后，直接读取 `data/output/<案例名>/diagnosis_report.json` 向用户解读诊断结果。

**注意事项**：
- 虚拟环境：`ai_agent`（conda）
- 转录耗时约 6 分钟（Whisper base + 说话人分离）
- 模块2+3 耗时约 30 秒（5 次 LLM 调用）
- 模块3 偶尔 JSON 截断，重试即可

---

## 项目本质

不只是语音分析，而是**可落地的成交分析系统**。

## 双通道输入设计

系统接收 **两个输入通道**，同时用于分析：

| 通道 | 内容 | 作用 |
|------|------|------|
| **语音** 🎤 | 老师与家长的通话录音 | 提取对话内容，分析沟通模式和成交阶段 |
| **文字背景** 📝 | 家长信息、孩子情况、历史沟通记录等 | 补充上下文，让诊断更有依据 |

> 两者缺一不可：语音还原「说了什么」，文字告诉你「为什么这么说」。

## 系统架构（MVP）

```
                ┌─────────────────┐
  Audio ──────→ │  WhisperX 管线  │ ─→ segments ─┐
                └─────────────────┘              │
                                                 ├──→ [模块2 观察层] → [模块3 评价层]
                ┌─────────────────┐              │
  Text (背景) ─→│  背景信息        │ ─→ 上下文 ──┘
                └─────────────────┘
```

**架构分层**：

| 层 | 模块 | 职责 |
|------|------|------|
| **数据层** | 模块1 (stt.py) | 语音 → 结构化文本（segments） |
| **观察层** | 模块2 (dialogue_parser.py) | 事实提取：角色/家长画像/成交阶段/老师行为模式 |
| **评价层** | 模块3 (diagnosis.py) | 价值判断：因果归因 + 致命失误 + 替代话术 |

### 模块1：语音转文本 (WhisperX 管线)
- 完整管线：转录 → 时间轴对齐 → 说话人分离(pyannote) → 合并
- 说话人分离需要 HuggingFace token，已设置系统环境变量 `HF_TOKEN`

### 模块2：对话结构拆解（观察层）

**设计原则**：
- **Observation vs Judgment 分离**：模块2 只做事实观察，模块3 做价值判断
- **串行推理链**：家长画像 → 阶段判断 → 老师行为观察（非并行分类器）
- **语义驱动**：角色分离基于语义特征（是否在讲课程、引导提问、解释方案），说话时长仅作回退信号
- **LLM 后端**：默认模型，统一 `_call_claude()` 入口

**数据流**：
```
segments (Module 1)
    ├── Step 1: separate_roles (语义特征)
    ├── Step 2: analyze_parent → ParentProfile
    ├── Step 3: detect_stage(parent_profile) → StageAnalysis
    └── Step 4: observe_teacher(stage, profile) → TeacherObservations
```

**数据结构**：
- `ParentProfile`: primary_type / confidence / key_phrases / concerns
- `StageAnalysis`: stage / evidence / coverage
- `TeacherObservations`: patterns (客观行为) / speaking_style / key_phrases
- `DialogueStructure`: 上述三者 + teacher_utterances + parent_utterances

**容错**：LLM 失败 → fallback 默认值 + 记录 `logs/llm_errors.json`，不中断流程

### 模块3：问题诊断（评价层）
- 基于模块2 的观察结果，一次 LLM 调用完成诊断
- 输出：`DiagnosisReport` — 决策阻碍、致命失误（含因果解释）、替代话术

## 输出格式标准

```
【决策阻碍】
- 家长真正的决策卡点（因果归因）

【致命失误】
- 最关键的一个错误 + 为什么对这个家长类型致命

【替代话术】
- 当时如果这样说，效果会更好（具体话术）
```

## 项目结构

```
AI家长成交分析Agent/
├── modules/
│   ├── stt.py              # 语音转文本 (Whisper)
│   ├── dialogue_parser.py  # 对话结构拆解
│   └── diagnosis.py        # 问题诊断
├── data/
│   ├── input/              # 放入待分析的音频
│   └── output/<录音名>/    # 每个录音独立输出目录
│       ├── segments.json
│       ├── dialogue_structure.json
│       └── diagnosis_report.json
├── main.py                 # 主入口
├── requirements.txt        # 依赖
└── .claude/
    └── CLAUDE.md           # 本文件
```

## 使用方式

```bash
# 完整管线（语音转录 + 对话拆解 + 问题诊断）
python main.py --audio data/input/test.mp3

# 跳过转录，直接分析已有 segments
python main.py --segments data/output/test/segments.json

# 指定 Whisper 模型大小（默认 base）
python main.py --audio data/input/test.mp3 --model medium
```

输出保存在 `data/output/<录音文件名>/`：

```
data/output/test/
├── segments.json              # 模块1：说话人分离结果
├── dialogue_structure.json    # 模块2：家长画像 + 阶段 + 老师观察
└── diagnosis_report.json      # 模块3：决策阻碍 + 致命失误 + 替代话术
```

## 后续升级方向

- Agent 化：LangChain / CrewAI 自动流程
- 记忆系统：CRM + AI，记录家长历史
- 评分系统：信任度 / 成交概率 / 风险评分
- 背景信息通道：目前仅设计了输入接口，尚未实现结构化解析

## 核心理念

1. **结构驱动 > 技术驱动**：优势在于懂家长的决策心理和成交逻辑
2. **观察与评价分离**：先看清事实（模块2），再判断好坏（模块3），避免评价污染观察
3. **串行优于并行**：上下文在步骤间传递，后续判断建立在前序发现之上
4. **容错优先**：单步失败不影响整体流程，fallback 值保证系统始终有输出