# Holosoma Semantic Retargeting / WBT 交接文档

> 更新时间：2026-09-02（Asia/Shanghai）
> 下一位助手：开始工作前请完整阅读本文。仓库中有大量已经完成的因果实验和本地结果，不要从头重跑，也不要删除未提交的实验产物。

## 1. 当前仓库和 Git 状态

- 仓库：`/home/zongyouyu/sxb/holosoma`
- 当前分支：`main`
- 用户目标远程：`git@github.com:SeeXB/holosoma.git`，remote 名称为 `seexb`
- 上游远程：`origin=https://github.com/amazon-far/holosoma.git`
- 已推送的最新提交：`b9e9f1509102aa1a0b9c906584421c14ede549c2`
- 提交标题：`feat: add final semantic retargeting and WBT rewards`
- `seexb/main` 已与上述提交一致。
- `main...origin/main [领先 2]` 是正常的：这里比较的是 Amazon 上游，不代表未推送到 SeeXB。

SSH 注意事项：

- `~/.ssh/config` 当前对 `github.com` 强制使用 `github_Jie_Chu`，该 key 被 GitHub 识别为 `BXZZcj`，没有 SeeXB 仓库写权限。
- `~/.ssh/id_ed25519` 被 GitHub 正确识别为 `SeeXB`。
- 后续推送应使用：

```bash
git -c 'core.sshCommand=ssh -i /home/zongyouyu/.ssh/id_ed25519 -o IdentitiesOnly=yes' push seexb main
```

## 2. 不要上传或删除的本地产物

本轮按用户要求只推送了代码、测试、脚本、配置和文档。以下目录仍完整保留在本地，但已加入 `.gitignore`，没有上传：

```text
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_approach_body_only/
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_semantic_budget/
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/
```

同样不上传：

- `*.mp4`、`*.avi`、`*.mov`、`*.mkv`
- `logs/`、训练 checkpoint、ONNX、W&B 本地缓存
- `.env`
- `MUJOCO_LOG.TXT`

这些文件只是被 Git 忽略，**没有从磁盘删除**。如需更新代码提交，不要用清理未跟踪文件的命令。

## 3. 项目主线的最终决定

当前方法已经收敛为两部分：

1. Semantic-aware OmniRetarget：生成最终 B4 参考轨迹。
2. Task-agnostic Semantic Keyframe Reward：在 WBT RL 中重新分配原始正奖励预算。

不要重新引入已经 Drop 的复杂分支作为主方法：

- Adaptive Compute：Drop。
- Criticality：最终方法不使用，`criticality_used=false`。
- Edge/Criticality/Adaptive 等早期消融只保留作历史实验。
- Legacy Semantic Weight 保留作基线，不是最终方法。

### 3.1 最终 retargeting 方法

最终模式：

```text
uniform2_semantic_weight_full_event_transition_truncated_budget
```

核心定义：

- 普通帧使用 Uniform-2。
- 全事件 semantic spatial weighting 保留。
- 每个事件的 spatial weighting support 从自身 trigger 开始。
- support 在自身 window end 或下一个事件 trigger 处截止，且下一个 trigger 不包含在前一事件中。
- 规则是通用 transition truncation，不允许 hardcode `approach/contact`。
- `approach` 仍然是关键事件，pelvis 仍参与，multiplier 仍为 2。
- approach 实际支持为 frame 26–29；frame 30 开始由 contact 接管。

这条 truncation 规则解决了 FullEvent 方法在 frame 34 出现的穿透，同时避免把问题解释为“approach pelvis 天生不该加权”。根因是串行轨迹中前一事件的 Gaussian tail 跨越了 semantic transition，并与 contact 阶段发生竞争。

### 3.2 B2/B4/B6/B8/B10 选择结果

在保留 transition truncation 后测试了关键帧额外预算 B2/B4/B6/B8/B10。选择规则：

```text
在 registered-feasible 方法中，选择 All-event Part Exact 距离可行最优值不超过 0.1% 时 Actual SQP 最少的方法。
```

最终选择 B4：

- run key：`transition_truncated_b4`
- exact trigger budget：4
- Actual SQP / Solver Calls：454 / 454
- All-event Part Exact：35.474696 mm
- Phase Part Exact：32.436281 mm
- Interaction Part Exact：40.969286 mm
- Global KF Exact：22.530844 mm
- Ordinary：21.820265 mm
- 最大穿透深度：0 mm
- frame 34 signed distance：+9.681426 mm

机器可读结果在本地：

```text
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/final_method.json
```

### 3.3 Semantic keyframe 数据

主 semantic 文件：

```text
src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json
```

相关事件计划和 VLM 审计文件位于同一目录。此前已经修正为：只修补非法字段，禁止 VLM 修改已经通过验证的事件。不要在 prompt 示例里泄露期望字段或左右手答案。

## 4. 最终 RL Semantic Reward

最终奖励不是额外加一个可调 semantic bonus，而是在原始正 tracking reward 的固定预算内重新分配：

```text
R_t = R_penalty(t)
    + W_pos / (W_pos + A_t) * [R_base,pos(t) + A_t * r_sem(t)]
```

其中：

- robot-only：`W_pos=5.0`
- robot + object：`W_pos=7.0`
- `A_t` 来自 transition-truncated semantic temporal gate，范围 `[0,1]`
- object 模式最大 semantic 混合比例 `alpha=1/(7+1)=0.125`
- penalty 完全沿用原路径，不随 semantic gate 缩放
- 当 `A=0` 时严格恢复 Omni baseline
- Part / Relative Geometry / Dynamics 对所有“有效目标”等权平均
- 无有效 external entity/relation 时 Rel 是 INVALID，从分母排除，而不是记为 0
- 没有事件名特化的 reward 分支；事件名只用于诊断

完整推导见：

```text
semantic_fixed_budget_reward_report.md
```

### 4.1 RL 实验预设

- R0：`exp:g1-29dof-wbt-w-object-semantic-r0-u2-omni`
- R1：`exp:g1-29dof-wbt-w-object-semantic-r1-b4-omni`
- R2：`exp:g1-29dof-wbt-w-object-semantic-r2-b4-part`
- R3：`exp:g1-29dof-wbt-w-object-semantic-r3-b4-part-rel`
- R4：`exp:g1-29dof-wbt-w-object-semantic-r4-b4-full`

启动脚本：

```bash
./scripts/train_semantic_wbt.sh r4
```

注意：脚本默认引用本地生成且未上传的 B4 RL motion：

```text
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/rl/transition_truncated_b4_mj_fps50_w_obj.npz
```

在新机器 clone 后必须先生成或拷贝该文件；当前工作站本地已有。

## 5. 已完成的 R4 训练与真实效果

R4 已训练到 30k iteration，训练目录：

```text
logs/WholeBodyTracking/20260818_184353-g1_29dof_wbt_manager-locomotion/
```

最终模型：

```text
logs/WholeBodyTracking/20260818_184353-g1_29dof_wbt_manager-locomotion/model_29999.pt
```

最终 ONNX 在同目录。W&B：

```text
https://wandb.ai/yumou0319-/WholeBodyTracking/runs/secz94cf
```

训练曲线数值上稳定：reward 上升、episode length 增加、KL/value 没有明显发散。但单环境 deterministic/robust evaluation 显示策略并未完成整条参考轨迹：

- 参考轨迹：325 frames，50 Hz，6.5 秒。
- final model robust eval：0/18 完整成功。
- 最长约 126/325 frames，即 2.52 秒。
- 多数 episode 因 `bad_tracking` reset。
- 去掉 per-reset initial pose noise 的测试仍为 0/9，最长约 108/325；该测试仍保留 startup domain randomization，因此不能称为完全 nominal。

最长的本地评估视频：

```text
logs/WholeBodyTracking/20260820_091159-g1_29dof_wbt_manager-wholebodytrackingmanager/renderings_training/episode_19_1787217152.mp4
```

零 reset-noise 视频：

```text
logs/WholeBodyTracking/20260820_091539-g1_29dof_wbt_manager-wholebodytrackingmanager/renderings_training/episode_10_1787217363.mp4
```

参考 MuJoCo 视频：

```text
src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/videos/final_transition_truncated_b4_mujoco.mp4
```

## 6. W&B rollout 为什么看起来像“一堆白色机器人”

这主要是渲染问题，不是 4096 个环境在物理上互相碰撞：

- object scene 使用 `env_spacing=0.0`
- 4096 个 env clone 在相同世界位置渲染重叠
- camera 跟踪 env0，但没有隐藏其他 clone
- `filter_collisions` 仍隔离不同环境的物理碰撞

因此 rollout media 本身不能用于判断 env0 是否学会完整动作。应以单环境评估、motion frame progression、reset reason 和完整轨迹成功率为准。

## 7. 官方 OMOMO demo 脚本的关键审计

文件：

```text
demo_scripts/demo_omomo_wb_tracking.sh
```

它确实是 upstream 官方仓库中的示例，但它是刻意简化的 **robot-only smoke demo**：

- retarget：`--task-type robot_only`
- conversion：`--object_name ground --once`
- train：`exp:g1-29dof-wbt`

虽然序列名是 `sub3_largebox_003`，该 demo 只取人体动作，丢弃动态 largebox，因此没有 object actor、object reward、contact/lift/place 难度。`ground` 对该 robot-only demo 是正确的，但不能拿它和当前 object-interaction R4 直接比较。

正确 object 流程是：

1. retarget 使用 `--task-type object_interaction`；
2. conversion 使用 `--object_name largebox --has_dynamic_object --once`，无显示环境加 `--headless`；
3. train 使用 `exp:g1-29dof-wbt-w-object` 或本项目的 R0–R4 object preset。

本地 B4 转换已按官方 README 命令重新验证：

- source：196 frames，30 fps，qpos 包含 object
- RL output：325 frames，50 fps，包含 object pose/quat/velocity
- 重转换后 keys/shapes 完全相同
- `object_pos_w`、`object_lin_vel_w` 完全相等
- 其他数值最大差约 `1e-5`，仅为重计算误差

因此当前失败不是 conversion flag 缺失。

MuJoCo conversion 与 IsaacSim training 使用的 largebox mesh SHA256 相同；质量、惯量、尺度和摩擦名义值一致，没有发现资产不匹配。

## 8. 当前最值得优先验证的训练问题

不能仅凭 R4 一次训练断言“semantic reward 导致学差”。更强的嫌疑是 object domain randomization 和训练难度设置：

文件：

```text
src/holosoma/holosoma/config_values/wbt/g1/randomization.py
```

重要事实：

- largebox 名义质量只有 `0.1 kg`
- object mass randomization 配置为 `[1.0, 4.0]`
- randomization term 使用 **additive offset**，所以训练质量实际是 `1.1–4.1 kg`
- 即名义质量的 11–41 倍
- object friction 随机到 `0.1–0.6`
- restitution 随机到 `0–1`
- inertia Ixx scale 为 `0.5–1.5`
- robot 还会每 1–3 秒受到随机 push
- 同时存在 base COM、robot material、initial pose/velocity randomization
- motion command 当前 `use_adaptive_timesteps_sampler=True`
- `start_at_timestep_zero_prob=0.0`

这些配置来自 upstream object WBT 路线，不是 semantic reward 新增的，但会让单条 manipulation motion 很难学。官方 robot-only demo 不包含这些 object 难度。

### 推荐的下一轮严格实验

不要直接修改 upstream base config。新增独立的 low-DR / clean object experiment preset，并按以下顺序做：

1. 禁用 random push。
2. 固定 object 为名义 `0.1 kg`，关闭 object mass/material/inertia randomization。
3. 保留一个真正 nominal 的 empty randomization manager；不要把 config 设为 `None`，因为 `BaseTask` 当前要求 manager config 非空。
4. 提高或固定 `start_at_timestep_zero_prob`，先验证策略能 overfit 一条 325-frame motion。
5. 在完全相同配置、seed 和训练预算下比较 R1（B4 Omni reward）与 R4（B4 full semantic reward）。
6. 之后逐项恢复 object mass、friction、push 等 DR，定位性能坍塌点。

必须报告的评估指标：

- 完整 325-frame 成功率
- 平均/最大到达 frame
- reset reason 分布
- robot tracking error
- object pose/rotation error
- contact/lift/place 阶段成功率
- 多 seed 均值和方差

只有 R1 与 R4 的同条件对比才能回答“新 reward 是否导致变差”。历史 reward diagnostic 显示 semantic contribution 相对 base contribution 很小，不能先验认定它是唯一根因。

## 9. 关键代码索引

Semantic retargeting：

```text
src/holosoma_retargeting/holosoma_retargeting/config_types/semantic.py
src/holosoma_retargeting/holosoma_retargeting/semantic_keyframes/runtime.py
src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py
src/holosoma_retargeting/holosoma_retargeting/examples/benchmark_full_event_semantic_budget.py
src/holosoma_retargeting/holosoma_retargeting/examples/benchmark_full_event_transition_truncation.py
src/holosoma_retargeting/holosoma_retargeting/examples/benchmark_transition_truncation_budget_curve.py
```

RL semantic reward：

```text
src/holosoma/holosoma/managers/reward/semantic_keyframes.py
src/holosoma/holosoma/managers/reward/manager.py
src/holosoma/holosoma/managers/reward/terms/wbt.py
src/holosoma/holosoma/config_types/reward.py
src/holosoma/holosoma/config_values/wbt/g1/reward.py
src/holosoma/holosoma/config_values/wbt/g1/experiment.py
src/holosoma/holosoma/config_values/wbt/g1/command.py
scripts/train_semantic_wbt.sh
```

测试：

```text
src/holosoma_retargeting/tests/test_semantic_retargeting_runtime.py
src/holosoma/holosoma/managers/reward/tests/test_semantic_keyframes.py
```

## 10. 已验证状态

在最新提交前运行：

- semantic retargeting focused tests：33 passed
- semantic reward focused tests：21 passed
- 合计：54 passed
- `git diff --check`：通过
- 提交中不含视频、模型、`.env`、日志、NPZ/Numpy 数据

历史更完整验证记录见 `semantic_fixed_budget_reward_report.md`：

- full no-simulator suite：225 passed, 1 skipped, 161 deselected
- live IsaacSim R4 smoke：2 env、1 PPO iteration、48 timesteps，正常退出

## 11. 下一次对话的建议开场

用户大概率会继续追问训练失败，或要求实现 clean/low-DR 对照实验。下一位助手应先：

1. 阅读本文和 `semantic_fixed_budget_reward_report.md`。
2. 检查用户的新请求是“诊断”还是明确要求“修改/开跑”。
3. 如果要求修改，新增独立实验 preset，避免污染 upstream config 和已有 R0–R4 定义。
4. 如果要求训练，先做 1-env/少量 env smoke，再开 tmux/W&B 大训练。
5. 如果要求评估，确保 single-env rendering 隐藏其他 clone，并使用完整轨迹成功率，不只看平均 reward。

不要重复争论或重新实现已经确定的 transition truncation、B4 选择和 fixed-budget reward；除非用户明确要求推翻最终定义。

## 12. 2026-08-26 后续检查

### 12.1 CARI4D 人体—箱体穿模（暂记，尚未修复）

最终 CARI4D SMPL-H 与多帧可见跨度 cuboid 的三维检查确认存在真实穿模，而不只是线框绘制造成的遮挡错觉：196 帧中 172 帧有人体表面顶点进入箱体，最大穿透约 0.121 m。恢复 cuboid 尺寸约 `0.319 × 0.355 × 0.362 m`，正确资产 OBB 约 `0.335 × 0.338 × 0.360 m`，尺寸差不足以解释穿透。CARI4D 接触项只拉手部关节靠近物体，penetration 项则以物体采样点查询人体 SDF；原始 joint optimization 在首次启用 penetration 的 step 1801 曾 OOM，减小 batch 后续跑完成，但最终约束仍未消除人体进入物体的问题。用户要求暂时记住，先不要继续修改 CARI4D。

### 12.2 shared-XY + semantic reward 训练的 24k 实测

目标训练仍在 tmux `b4_semantic_shared_xy_30k` 中运行；训练目录：

```text
logs/WholeBodyTracking/20260824_192645-b4_semantic_shared_xy_default_s42-locomotion/
```

W&B run：`yumou0319-/WholeBodyTracking/11ox0wm7`。2026-08-26 评估时训练约 26.5k/30k，最新已落盘 checkpoint 为 `model_24000.pt`。

对 24k checkpoint 从参考第 0 帧做了 324-step deterministic policy rollout：初始 pose noise=0、push disabled，保留该训练的 seed-42 startup DR；所有 `bad_tracking` 阈值临时设为 999，以免提前终止掩盖真实行为。结果：机器人学会了下蹲、起身和向前移动，但没有抬起并携带箱子，且有效段机器人 XY 位移约 2.455 m，参考 pelvis 仅约 1.566 m，明显过冲。参考箱子应水平移动约 1.443 m、最高抬到 z≈0.845 m；视频中箱子被留在起点。最后自动 motion reset 帧已从展示视频中裁掉。

主要产物：

```text
outputs/eval_b4_semantic_shared_xy_model24000/model24000_fullclip_no_bad_tracking_trimmed.mp4
outputs/eval_b4_semantic_shared_xy_model24000/model24000_fullclip_no_bad_tracking_preview.png
outputs/eval_b4_semantic_shared_xy_model24000/actual_fullclip_seed42_training_dr.npz
```

恢复正常 `bad_tracking` 阈值、保持同一第 0 帧初始化后，episode 只有 9–10 个控制帧。由无早停轨迹对照参考计算可见：reset 第 0 帧不是错位，右手腕 z 误差随后增长，并在约第 7 步超过 `bad_motion_body_pos_threshold=0.25 m`，因此触发 `bad_tracking`。这说明当前高 training average episode length 不能代表从 frame 0 成功完成搬运；24k 完整任务结果仍为失败。

随后又做了一次排除 startup domain randomization 的名义物理参数评估：object mass=`0.1 kg`、robot/object friction=`0.9`、restitution=`0`、inertia scale=`1`，并关闭 base CoM、DOF bias、push 和初始 pose noise；同样将 `bad_tracking` 阈值设为 999，完整观察 324 step。结果仍未成功，而且机器人在中段失衡倒地，箱子被碰离参考运动，没有形成稳定双手抓取、抬升或搬运。对应裁剪视频：

```text
outputs/eval_b4_semantic_shared_xy_model24000/nominal/model24000_nominal_fullclip_no_bad_tracking_trimmed.mp4
```

因此，24k 的失败不能仅归因于某一次 startup DR 抽到了困难箱体参数；但这也不是严格的 reward 消融结论，因为训练本身是在强 DR 下完成的，名义参数对该策略也属于分布中的特定点。2026-08-26 06:06 时训练仍在 `26578/30000`，`Env/average_episode_length=366.667`，最新落盘仍是 24k checkpoint；需等待 28k/最终 checkpoint 后再次做相同的完整任务评估。

## 13. 2026-08-26 换对话前的最终工作状态

### 13.1 Human video → CARI4D → canonical HOI → IsaacLab USD

已经在 Holosoma 侧实现两环境、文件交换式 pipeline；没有把 CARI4D 与 IsaacLab 强耦合，也没有自动安装/升级环境。主要说明见：

```text
docs/cari4d_video_to_isaaclab.md
```

主要入口和模块：

```text
scripts/prepare_video_for_cari4d.sh
scripts/run_video_to_hoi.sh
tools/prepare_cari4d_video.py
tools/run_cari4d_custom.py
tools/export_cari4d_sequence.py
tools/validate_cari4d_export.py
tools/visualize_object_trajectory.py
tools/convert_object_to_usd.py
tools/validate_object_usd.py
tools/hoi_pipeline/
```

通用 RGB video 预处理默认调用 CARI4D 官方 SAM3 text-prompt mask 路线；OMOMO 的 color-key 仅曾用于诊断，用户已要求删除该次结果，不能把它当作正式 pipeline。正式 wrapper 对 masks、Sapiens/keypoints、Hunyuan mesh、SMPL-H、UniDepth、DINOv2 和 VolumetricSMPL 权重做显式 preflight，缺失时直接报错而不 silent fallback。

CARI4D adapter 依据当前 checkout 的真实 serialization：最终 result 为 `run_horefine.py` / `opt_refineout.py` 写出的 `.pth`，关键字段为 `pr.pose_abs`、`pr.smpl_pose`、`pr.smpl_t`、`pr.betas`、`pr.frames`；scale 来自 `estimate_scale.py` JSON 的 `best_scale`，其 `_align.obj` 已经执行 `vertices * best_scale`，exporter 因而不会再 scale 一次。输出保持 `cari4d_metric_camera`（OpenCV `+X right,+Y down,+Z forward`），没有猜测 Isaac 坐标变换。

实际 OMOMO large-box 视频已完整跑过 CARI4D，canonical 目录：

```text
outputs/human_video_sub3_largebox_003/
```

结果为 196 帧、30 FPS；`human/smplh.npz`、`object/trajectory.npz`、`object/initial_pose.npy`、metadata、validation 均已生成。Human/object/video frame IDs 对齐，rotation/quaternion/NaN/Inf 检查通过。IsaacLab 2.3 / Isaac Sim 5.1 的 `MeshConverter` 也已实际生成 `object.usd`，默认 `convexDecomposition`、placeholder mass 1 kg、OBJ/USD bbox 一致。所有这些 output、权重和本地第三方环境均为生成物，不应提交 Git。

视频提取轨迹与 OMOMO GT 只作 evaluation 对比、没有用 GT 参与恢复。报告：

```text
outputs/human_video_sub3_largebox_003/validation/cari4d_vs_intermimic_report.json
```

重要数值：root-centered body22 MPJPE 约 `0.0532 m`，human scale 对齐约 `1.0057`；object translation rigid-alignment RMSE 约 `0.245 m`，similarity scale 约 `0.9255`，object/pelvis distance correlation 约 `0.704`。物体方向更差，relative orientation error mean 约 `31.8°`。因此人体恢复总体可用，物体平移趋势可用但仍有明显噪声，旋转还不能称为精确复现。

### 13.2 物体几何、多帧碰撞盒与穿模遗留项

单帧 Hunyuan 对该 OMOMO 视角恢复错误：normalized extent `[0.8637,0.8116,1.9951]`，UniDepth scale `0.64448` 后 metric extent 约 `[0.5567,0.5231,1.2858] m`；这不是 exporter 漏乘或重复乘 scale，而是单帧 shape 本身把深度/高度恢复错了。用户已决定这条 OMOMO 样本的最终仿真物体直接使用数据集已知的 `largebox.obj`，不要继续纠缠单帧 Hunyuan shape。

多帧诊断工具：

```text
tools/estimate_multiframe_collision_box.py
```

用户选择“旧 Hunyuan/FoundationPose 每帧 pose + 每帧可见跨度中位数”作为该视频的箱体尺寸恢复方法。其 canonical-axis median visible span 为约：

```text
0.3189 × 0.3549 × 0.3618 m
```

与已知正确 largebox OBB `0.3351 × 0.3377 × 0.3601 m` 很接近。注意工具报告也明确说明它是 visible-span diagnostic，而不是严格包围所有点的 collider；不要把不稳定的全局 fusion OBB `0.358 × 0.497 × 0.578 m` 误称为最终尺寸。报告位于：

```text
outputs/human_video_sub3_largebox_003/validation/multiframe_collision/old_hunyuan_fp/report.json
```

最终 SMPL-H + 上述 cuboid 可视化确认了真实穿模，详细结论见 12.1。这个问题只记录、未修复；用户明确要求先转回 RL，不要继续修改 CARI4D optimization。

### 13.3 CARI4D → retargeting 的代码边界

新增 strict `cari4d` bundle loader 与显式 object asset override：

```text
tools/export_cari4d_retargeting_input.py
tools/resolve_semantic_plan_for_retargeting_bundle.py
src/holosoma_retargeting/holosoma_retargeting/src/utils.py
src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py
src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py
src/holosoma_retargeting/holosoma_retargeting/config_types/task.py
```

这些代码只提供 canonical CARI4D trajectory 到后续 retargeting 的稳定输入/schema；尚未宣称已经用视频恢复轨迹完成最终 G1 retargeting 或 RL training。不要把 `human/smplh.npz` 直接当成已经可训练的 G1 motion。

本地 `third_party/CARI4D` 是独立 checkout（约 13 GB，含多个 venv/weights/output），父仓库没有追踪它。里面有为实际运行做过的本地兼容改动，但因用户强调第三方库不应大改，本次父仓库提交不会纳入整个第三方目录；以后若需保留，应该整理为极小 patch 或固定 commit/submodule，而不是直接 vendor 环境和权重。

### 13.4 reset 扰动修复与当前 semantic 训练

已实现 robot/object reset 共享 XY 平移：`NoiseToInitialPoseConfig.share_object_xy_noise_with_root`。object WBT preset 关闭独立 object XY noise，并把同一 `root_pos_delta[:2]` 加到物体，以保持 reset 后手—箱相对关系；z 仍独立控制。相关文件：

```text
src/holosoma/holosoma/config_types/command.py
src/holosoma/holosoma/config_values/wbt/g1/command.py
src/holosoma/holosoma/managers/command/terms/wbt.py
```

关键帧 semantic reward + B4 shared-XY 训练仍在运行：

```text
tmux: b4_semantic_shared_xy_30k
run: logs/WholeBodyTracking/20260824_192645-b4_semantic_shared_xy_default_s42-locomotion/
W&B: yumou0319-/WholeBodyTracking/11ox0wm7
```

换对话前进度为 `26720/30000`，最新已保存 checkpoint 仍为 `model_24000.pt`。24k 两个完整无 `bad_tracking` rollout 都失败，详见 12.2；等待 28k/29999 后必须以同样条件重新评估，不能用 average episode length 代替搬运成功率。

目录名容易混淆：

```text
diagnostics/object_drift/eval_b4_original_reward_30k_nominal/
```

它评估的是 `logs/WholeBodyTracking/20260820_184109-g1_29dof_wbt_manager-locomotion/model_29999.pt` 对 B4 reference 的表现。该模型使用原始 OmniRetarget object-WBT reward（`semantic_keyframe: null`），不是 semantic reward 模型。该 rollout 固定从 frame 0 开始、initial pose noise=0、push/actuator delay/DOF bias 关闭，并将所有 tracking termination threshold 设成 999，目的只是观察不早停时 actual/reference 如何漂移。`nominal` 指这次评估条件，不代表训练时没有强 DR。它记录 326 samples；motion 走完后自动回到 frame 0，比较脚本使用 frame 1–324。summary 中最大 torso Y error 约 `1.381 m`，最大 object Y error约 `0.257 m`。该目录不是成功率评估，也不能拿它证明原奖励策略通过正常终止阈值。

### 13.5 Git 提交约束和本地生成物

本轮 Git 只应包含源码、wrapper、测试、轻量文档/诊断脚本以及本文。明确排除：

- `outputs/`、训练 `logs/`、视频、checkpoint、NPZ 大数据、模型权重；
- 所有后续 eval session、rollout、summary 和视频统一写入 `exp/eval/`；`runs/` 是废弃目录，启动脚本统一放在 `scripts/`。不要再把评测产物写到 `outputs/` 或 `diagnostics/`。物理 rollout 必须关闭实时相机，先记录 NPZ，再离线生成视频。
- 整个本地 `third_party/` checkout 和其中虚拟环境；
- `tools/__pycache__`、`tests/**/__pycache__`；
- `.env` 及任何 Hugging Face token。

工作区还显示已跟踪的 `THIRD_PARTY_LICENSES` 与 `uv.lock` 被删除；这两项与当前功能无关且删除来源不明确，本次不要提交其删除，也不要在没有用户确认时做 destructive restore。下一位助手看到它们仍 dirty 是预期现象。

### 13.6 CARI4D 箱体旋转稳定化（2026-08-27，覆盖 12.1/13.2 的“不要继续修改”旧决定）

用户后来明确授权修改 CARI4D 源码以稳定箱体旋转。本轮在独立 checkout `third_party/CARI4D_pristine`（base commit `71fa7cbe...`）只修改了：

```text
estimater.py
prep/fp_behave.py
```

V1 在每 30 帧 FoundationPose reinit 时选择离上一帧最近的旋转候选，并在最近候选仍超过 10° 时沿用上一帧旋转，消除了 84°–179° 的 symmetry flip；但用户目检仍发现 reinit 之间有左右摆动。V2 在完整 196-frame pose sequence 上增加 21-frame、polyorder=1 的 zero-phase quaternion/SO(3) smoothing，并在最终 joint optimization 使用 `opt_rot=False`，防止 optimizer 重新放大旋转噪声；object translation、human pose、contact 与 penetration 仍照常优化。

V2 正式结果：

```text
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_staging_1024/fp-hy3d-track-sam3-reinit30-rotstable-smooth21/
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_run/output/coconet_sam3_reinit30_rotstable_smooth21_1024/
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_run/output/opt_sam3_reinit30_rotstable_smooth21_1024/
```

最终视频为 H.264/yuv420p、1280×256、30 fps、196 frames；3000-step 日志以 `all done` 退出。rotation delta p95/max 从 V1 的 `6.14°/12.96°` 降到 `2.34°/2.77°`，总旋转路径从 `379.16°` 降到 `146.64°`（OMOMO reference `138.64°`）；decoded silhouette IoU mean 为 `0.612754`，V1 为 `0.616445`。最终 PTH 相对锁定输入的最大旋转差仅 `0.000023°`。

用户验证入口：

```text
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_run/output/rotation_stabilization_v2_old_vs_smooth21_step3000.mp4
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_run/output/rotation_stabilization_v2_object_focus.mp4
exp/omomo_cari4d/sub03_largebox3/cari4d_pristine_run/output/opt_sam3_reinit30_rotstable_smooth21_1024/cari4d-release+step031397_demo_sam3_r30_rs_s21-hy3d3-optv2/OMOMO_Sub03_largebox_003+step003000.mp4
exp/omomo_cari4d/sub03_largebox3/audit_contact_sheets/rotation_smooth21_step3000_object_overlay_every6.png
```

完整路径、hash 和机器可读指标见 `exp/omomo_cari4d/sub03_largebox3/FINAL_REPORT.md`、`sam3/RUN_SUMMARY.json`、`sam3/final_tracking_metrics.json`。两个新增行为均默认关闭，不影响 CARI4D 原行为；启用参数是 `--stabilize_reinit_rotation --smooth_track_rotations --rotation_smoothing_window 21 --rotation_smoothing_polyorder 1`。

### 13.7 Paper-DR robustness eval 与 RL 当前状态（2026-09-02）

本轮已完成 Paper-DR robustness 评测，并将评测入口整理到 `scripts/`：

```text
scripts/eval_paper_dr_robustness.sh
scripts/eval_paper_dr_robustness_instrumentation.py
scripts/multienv_recording_instrumentation.py
scripts/analyze_paper_dr_robustness.py
```

其中 `scripts/multienv_recording_instrumentation.py` 是原先放在废弃 `runs/eval/` 下的 all-environment recorder 的正式归位；Paper-DR wrapper 不再依赖 `runs/`。`locomotion.apply_pushes` 增加了显式 eval push opt-in，默认 eval 行为仍保持关闭，robustness protocol 才会打开论文中的 push。

训练完成的 checkpoint 与 W&B run：

```text
Omni baseline (original_adaptive): logs/WholeBodyTracking/20260827_145241-b4_omni_paperdr_supported_s42-locomotion/model_29999.pt
S1 semantic_uniform:              logs/WholeBodyTracking/20260829_225620-b4_s1_semantic_uniform_paperdr_s42-locomotion/model_29999.pt
S2 semantic_adaptive:             logs/WholeBodyTracking/20260829_225647-b4_s2_semantic_adaptive_paperdr_s42-locomotion/model_29999.pt
W&B run IDs: Omni=424fhe5j, S1=2f73w97w, S2=0ktg4iqz
```

三者使用同一 Omni reward/PPO 与 Paper-DR 物理随机化；S1/S2 的差别是 semantic transition reset sampler（uniform vs adaptive）。标准评测固定 B4 reference `exp/benchmark_results_full_event_transition_truncation/rl/transition_truncated_b4_mj_fps50_w_obj.npz`、seed 42、32 个并行环境、每环境前 10 个完整 episode（320 个 episode/model）、324 个有效 motion frames、6.48 s horizon。评测时开启论文 push（1–3 s 间隔，最大线速度 0.3 m/s、角速度 0.78 rad/s），关闭 initial-pose noise；object termination threshold 为 position 1.0 m、orientation π/4。论文所述 shape ±10% 尚未在当前 simulator setup 实现，`paper_shape_scale_implemented=false`，不应误称为已覆盖。

最终统一报告与 JSON：

```text
exp/eval/paper_dr_robustness/PAPER_DR_ROBUSTNESS_REPORT.md
exp/eval/paper_dr_robustness/paper_dr_robustness_results.json
```

结果（每项 320/320，SR=100%）：

| model | object position RMSE | object orientation RMSE | tracked-body position RMSE | push events |
|---|---:|---:|---:|---:|
| Omni baseline | 0.0810 m | 6.12° | 0.0478 m | 966 |
| S1 semantic_uniform | 0.0775 m | 5.51° | 0.0462 m | 968 |
| S2 semantic_adaptive | 0.0701 m | 5.13° | 0.0445 m | 968 |

三者都没有有效 motion frame 上的早停 tracking failure。原始 terminal snapshot 中少量 `bad_object_pos`/`bad_object_ori` 是 clip 完成后 command 已回到 frame 0、而 PhysX 状态尚未同步造成的一拍 stale flag；分析器已按最后一个有效 motion frame 判定成功，并单独记录 stale 计数（Omni 1、S1 2、S2 1），不能把它们计为失败。

本次提交只应包含源码、测试、评测/启动脚本和上述轻量报告；训练日志、checkpoint、raw rollout NPZ、视频、`outputs/`、`third_party/` 和 `.env` 留在本机并由 `.gitignore` 排除。当前仍存在旧 benchmark 结果迁移产生的大量 tracked deletion，以及 `THIRD_PARTY_LICENSES`、`uv.lock` 删除；这些不属于本次 RL success 提交，未纳入 stage。
