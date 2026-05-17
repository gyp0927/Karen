"""Result aggregators for different parallel scenarios."""

from __future__ import annotations

import re

from .base import SubTaskResult


def merge_code_reviews(results: list[SubTaskResult]) -> str:
    """Aggregate code review results from multiple sub-agents.

    Deduplicate by file:line, sort by severity, keep highest severity per issue.
    """
    # 严重级别映射：数值越小越严重
    severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "严重": 0, "中等": 1, "轻微": 2, "建议": 3}

    # 收集所有问题：(severity_key, location, description, fix, source_task)
    issues: list[tuple[int, str, str, str, str]] = []

    for r in results:
        if not r.success:
            continue
        # 匹配 P0/P1/P2/P3 格式的问题行
        for match in re.finditer(
            r"[-*]\s*(?:\*\*)?\[?(P[0-3]|严重|中等|轻微|建议)\]?(?:\*\*)?\s*(.*?)$", r.output, re.MULTILINE
        ):
            severity = match.group(1)
            desc = match.group(2).strip()
            sev_key = severity_order.get(severity, 99)
            # 尝试提取位置信息
            loc_match = re.search(r"(\S+\.\w+):(\d+)", desc)
            location = f"{loc_match.group(1)}:{loc_match.group(2)}" if loc_match else "未知位置"
            issues.append((sev_key, location, desc, "", r.task_id))

    # 按 location 去重，保留最高严重级别
    seen: dict[str, tuple[int, str, str, str]] = {}
    for sev_key, location, desc, fix, source in issues:
        if location not in seen or sev_key < seen[location][0]:
            seen[location] = (sev_key, desc, fix, source)

    # 按严重级别排序
    deduped = sorted(seen.values(), key=lambda x: x[0])

    # 统计
    counts: dict[int, int] = {}
    for sev_key, _, _, _ in deduped:
        counts[sev_key] = counts.get(sev_key, 0) + 1

    # 生成报告
    sev_names = {0: "P0（严重）", 1: "P1（中等）", 2: "P2（轻微）", 3: "P3（建议）"}
    lines: list[str] = ["## 代码审查报告（多专家并行审查）"]
    lines.append("")
    lines.append("### 统计")
    for k in sorted(counts.keys()):
        lines.append(f"- {sev_names.get(k, f'P{k}')}: {counts[k]} 个问题")
    lines.append("")

    if deduped:
        lines.append("### 详情")
        for sev_key, desc, fix, source in deduped:
            lines.append(f"- **{sev_names.get(sev_key, f'P{sev_key}')}** {desc}")
            if fix:
                lines.append(f"  - 修复建议: {fix}")
    else:
        lines.append("未发现明显问题。")

    return "\n".join(lines)


def merge_research(results: list[SubTaskResult]) -> str:
    """Aggregate research results from multiple sources.

    Remove duplicate facts, merge complementary information,
    mark information conflicts.
    """
    task_outputs: dict[str, str] = {}
    for r in results:
        if r.success:
            task_outputs[r.task_id] = r.output

    if not task_outputs:
        return "所有研究子任务均失败。"

    lines: list[str] = ["## 深度研究报告（多源并行分析）"]
    lines.append("")

    # 按来源组织
    angle_names = {
        "tech-research": "技术分析",
        "practical-research": "实践分析",
        "critical-research": "批判分析",
    }

    for task_id, output in task_outputs.items():
        angle = angle_names.get(task_id, task_id)
        lines.append(f"### {angle}")
        lines.append(output.strip())
        lines.append("")

    # 简单的冲突检测：检查是否有互斥的关键词对
    conflict_pairs = [
        ("推荐", "不推荐"),
        ("优点", "缺点"),
        ("优势", "劣势"),
        ("高效", "低效"),
    ]

    full_text = " ".join(task_outputs.values())
    conflicts: list[str] = []
    for pos, neg in conflict_pairs:
        if pos in full_text and neg in full_text:
            conflicts.append(f"- 发现'{pos}'与'{neg}'的表述同时存在，请注意权衡")

    if conflicts:
        lines.append("### 注意")
        lines.append("不同角度分析中可能存在观点冲突：")
        lines.extend(conflicts)

    return "\n".join(lines)


def vote_boolean(results: list[SubTaskResult]) -> str:
    """Vote on a yes/no question.

    Returns majority decision with confidence score.
    """
    positive_keywords = ["是", "yes", "true", "正确", "可以", "推荐", "支持"]
    negative_keywords = ["否", "no", "false", "错误", "不可以", "不推荐", "反对"]

    votes: list[bool | None] = []
    for r in results:
        if not r.success:
            votes.append(None)
            continue
        text = r.output.lower()
        pos = sum(1 for kw in positive_keywords if kw in text)
        neg = sum(1 for kw in negative_keywords if kw in text)
        if pos > neg:
            votes.append(True)
        elif neg > pos:
            votes.append(False)
        else:
            votes.append(None)

    valid = [v for v in votes if v is not None]
    if not valid:
        return "无法达成有效表决（所有子任务失败或无明确意见）。"

    yes_count = sum(1 for v in valid if v)
    no_count = len(valid) - yes_count
    confidence = max(yes_count, no_count) / len(valid)

    if yes_count > no_count:
        return f"表决结果：**是**（{yes_count}/{len(valid)} 赞成，置信度 {confidence:.0%}）"
    if no_count > yes_count:
        return f"表决结果：**否**（{no_count}/{len(valid)} 反对，置信度 {confidence:.0%}）"
    return f"表决结果：**平局**（{yes_count} 赞成 vs {no_count} 反对）"


def rank_by_confidence(results: list[SubTaskResult]) -> str:
    """Rank results by confidence score in metadata.

    Returns top result with explanation.
    """
    scored = [(r, r.metadata.get("score", 0.0)) for r in results if r.success]
    if not scored:
        return "所有子任务均失败。"

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    score = scored[0][1]

    lines = [
        f"## 最优结果（置信度: {score}）",
        "",
        f"来源: {best.task_id}",
        "",
        best.output,
    ]

    if len(scored) > 1:
        lines.append("")
        lines.append("### 其他候选")
        for r, s in scored[1:]:
            lines.append(f"- {r.task_id}（置信度: {s}）")

    return "\n".join(lines)
