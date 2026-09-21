#!/usr/bin/env python3
"""Build the local review index, preserving the distinction between validation and visual review."""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from generate_remaining37_semantic_plans import COMPLETED, DATA, LAFAN_TASKS, TASK_OBJECTS, VERSION, write_json
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from holosoma.utils.semantic_contacts import load_semantic_contact_parts, resolve_semantic_contact_links
from monitor_remaining37_semantic_plans import snapshot


def main():
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    constraints_file = root / "task_body_constraints.json"
    constraints = json.loads(constraints_file.read_text()) if constraints_file.is_file() else {}
    tasks = [("omomo", t) for t in TASK_OBJECTS if t not in COMPLETED] + [("lafan", t) for t in LAFAN_TASKS]
    rows = snapshot(root, tasks)
    urdf = Path("src/holosoma/holosoma/data/robots/g1/main_mesh_collision_halfspherehand.urdf")
    links = [e.attrib["name"] for e in ET.parse(urdf).getroot().findall("link")]
    pattern = g1_29dof_wbt_reward_w_object.terms["undesired_contacts"].params["undesired_contacts_body_names"]
    counts = Counter()
    for row in rows:
        if row["status"] != "validated":
            continue
        file = Path(row["output"])
        result = json.loads(file.read_text())
        mode = result.get("generation_metadata", {}).get("planning_mode", "legacy_dynamic")
        counts[mode] += 1
        parts = load_semantic_contact_parts(file)
        mapping = resolve_semantic_contact_links(parts, links)
        exemptions = {name for names in mapping.values() for name in names}
        removed = [name for name in links if re.match(pattern, name) and name in exemptions]
        row.update(
            method=mode,
            event_count=len(result["events"]),
            contact_mapping=mapping,
            removed_authored_penalty_links=removed,
            plan_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            boundary_sources=dict(Counter(e.get("boundary_source", "legacy_dynamic_rule") for e in result["events"])),
            triggers=[e["windows"][0]["trigger_frame"] for e in result["events"]],
            training_ready="not_assessed",
            visual_review="pending",
            body_part_constraints=result.get("generation_metadata", {}).get("body_part_constraints"),
        )
    preserved = []
    current_manifest = audit / "latest_sources/manifest.json"
    if current_manifest.is_file():
        digests = json.loads(current_manifest.read_text())["current_plan_sha256"]
        changed = [name for name, sha in digests.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha]
        preserved.append({"manifest": str(current_manifest), "checked_files": len(digests), "changed": changed})
    ok = sum(r["status"] == "validated" for r in rows)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "validated": ok,
        "total": 37,
        "visual_review": "pending",
        "new_training_started": False,
        "methods": dict(counts),
        "body_part_constraints": constraints,
        "preservation_checks": preserved,
        "results": rows,
    }
    write_json(audit / "review.json", payload)

    def link(path):
        return os.path.relpath(Path(path).resolve(), audit.resolve())

    lines = [
        "# Semantic plan 局部阶段生成报告",
        "",
        f"更新：{payload['updated_at']}；**{ok}/37 本地校验通过**。",
        "",
        "本次范围是18个OMOMO和19个LAFAN；此前完成的两个largebox任务不在生成列表中。后续按用户要求重生成18个OMOMO，19个LAFAN保持原样；见HAND_CONTACT_REVIEW.md。",
        "",
        "**这表示生成及本地结构/时间解析通过，不表示视觉语义已人工确认，也不表示全部任务的RL输入已ready。未启动新版训练。**",
        "",
        "## 双手任务约束（用户指定）",
        "",
        f"已明确为双手任务的 {len(constraints)} 份计划：" + "、".join(constraints) + "。",
        "这些任务中肩、肘/前臂、腕、手出现一侧即补齐另一侧。膝、踝等下肢不自动扩展；未确认的任务不自动归类。",
        "task_body_constraints.json保存任务约束；actions记录body_parts_before_constraints及body_parts_added_by_constraints，segments.visual_plan和模型原始回复不变。",
        "旧候选、失败批次和旧版备份已按用户要求删除。仅保留最新计划、有效输入缓存，以及latest_sources/中的有效模型回复。清理记录见cleanup.json。",
        "",
        "## 修改后的方法",
        "",
        "- 当前生成提示词全部使用英文，放在图像之后。模型根据编号图像给出动作、部位、样本区间和理由；仅保留当前结果实际采用的原始回复。",
        "- OMOMO每阶段分别说明hand/wrist/forearm接触；body_parts取动作关键部位与明确识别出的接触部位的并集，contact_body_parts及新增部位来源单列。uncertain不被当成已确认接触。",
        "- OMOMO用20个真实视频采样组成两张拼图，以便模型看到完整动作。LAFAN按最长20秒分段，每段32个双视角骨架样本；失败片段可细分为10/5秒，保持原覆盖校验，最后不足20秒保留原长。",
        "- 按视觉区间构造互不重叠的局部搜索范围。物体（OMOMO）/pelvis（LAFAN）局部垂直速度达到0.04m/s且方向明显占优时，采用局部对应方向速度峰值；否则明确记录为visual_sample视觉边界。",
        "- 物理规则找不到事件则报错，不使用旧解析器“最近阈值帧”兜底；视觉边界本身是近似标注，不宣称是物理接触时刻。",
        "- 支持共享边界或相邻编号两种图像区间写法，仍拒绝倒序、重叠、未知部位、局部覆盖不足、零长度或重复触发。静止首尾可不标注，区间不保证覆盖每一帧。",
        "- 已通过片段按提示词、采样帧和轨迹信号指纹缓存；额度/网络中断后续跑未完成片段。每个未完成片段最多3次模型请求。旧计划仍按原解析器重放，未改变已完成任务语义。",
        "",
        "## 验证",
        "",
        "- 每份计划：JSON结构、原GT重解析一致性、时间线、原生FPS sampler、实际G1 URDF link映射检查。",
        "- 局部阶段生成历史回归62项、监测器5项通过；本轮英文接触、并集规则及相关回归40项通过。每份当前计划均重新核对GT解析一致性。",
        f"- 当前74份计划JSON文件哈希核对：{sum(p['checked_files'] for p in preserved)}次比对，其中差异 {sum(len(p['changed']) for p in preserved)} 次。",
        "",
        "## largetable 新版入口",
        "",
        f"- [解析后的semantic_plan.json]({link(root / 'omomo/sub1_largetable_028/semantic_plan.json')})",
        f"- [事件规则及模型视觉阶段]({link(root / 'omomo/sub1_largetable_028/semantic_plan.event_plan.json')})",
        "",
        "桌子当前触发帧见review.json对应任务的triggers；历史局部阶段版本已被英文接触版替换。",
        "桌子计划已按双手任务规则补齐左肘、左腕、左手；原始模型只选右侧的回复仍保留用于追溯。动作名称及部位选择的其他语义问题仍待复核。",
        "",
        "## 所有任务",
        "",
        "| 数据集 | 任务/文件 | 状态 | 方法 | 事件数 | 部位数 |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        task = row["task"]
        method = "局部视觉阶段" if row.get("method") == "holosoma.localized_visual_phases.v1" else "保留已有计划"
        target = f"[{task}]({link(row['output'])})" if row["status"] == "validated" else task
        lines.append(
            f"| {row['dataset']} | {target} | {row['status']} | {method} | {row.get('event_count', '—')} | {len(row.get('body_parts', []))} |"
        )
    lines += [
        "",
        "## 审核注意点",
        "",
        "LAFAN长片段的动作命名仍可能混淆相似动作，分段边界也不一定是真实动作切换。图像、sample_frames.json、逐段模型visual_plan和原始回复都保留，可逐段核查。",
        "奖励按整份plan部位并集豁免接触；长任务列出的部位越多，豁免越广。review.json列出实际URDF映射和被剔除的惩罚link，供训练前核对。",
        "",
        "输入/计划位于demo_data/semantic_keyframes/g1_anatomy_v2_20260920；模型回复、监测和本报告位于exp/semantic_plans/g1_anatomy_v2_20260920。旧bundles和frozen计划未替换。",
        "",
        "继续未完成任务（会自动跳过已通过文件）：",
        "```bash",
        "PATH=/usr/bin:$PATH PYTHONPATH=src/holosoma:src/holosoma_retargeting:tools \\",
        "  /home/zongyouyu/miniconda3/envs/hssim/bin/python \\",
        "  tools/regenerate_incomplete_semantic_plans.py --workers 1 --max-repairs 2",
        "```",
        "",
    ]
    (audit / "REVIEW.md").write_text("\n".join(lines))
    (root / "README.md").write_text(
        f"# G1 anatomy semantic plans\n\n{ok}/37通过本地校验；视觉复核待完成，未开始新版训练。\n\n完整报告：`exp/semantic_plans/{VERSION}/REVIEW.md`。\n\n每任务的semantic_plan.json可供审核；semantic_plan.event_plan.json包含可重放规则。localized_visual_inputs保存送模型的编号图片与GT帧对应；localized_phase_cache保存通过验证的模型分段回复。\n"
    )
    print(
        json.dumps(
            {
                "validated": ok,
                "methods": dict(counts),
                "preservation_changes": sum(len(p["changed"]) for p in preserved),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
