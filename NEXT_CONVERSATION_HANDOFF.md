# Holosoma Semantic Retargeting / WBT 交接文档

## 2026-09-18 最新交接（优先于下方历史记录）

### 当前实验：训练中，尚未到正式最终评测

`sub1_largetable_028` 的三组正式训练于 **2026-09-17 18:17（Asia/Shanghai）** 启动。2026-09-18 23:26 检查：三组日志分别推进到约 **10559 / 10556 / 9490，目标均为 30000 iterations**；state 均为 `training`，日志持续更新，尚未生成最终评测结果。此处是检查时快照，接手时需读取 live state/日志，不要重复启动。

| 组 | 轨迹 / RL | W&B run | 训练 PID |
|---|---|---|---|
| 1 | Original / Original adaptive | [f0316c9b](https://wandb.ai/yumou0319-/WholeBodyTracking/runs/f0316c9b) | 1912123 |
| 2 | Semantic B4 / Original adaptive | [18c44b25](https://wandb.ai/yumou0319-/WholeBodyTracking/runs/18c44b25) | 1912125 |
| 3 | Semantic B4 / Semantic adaptive | [c9b21365](https://wandb.ai/yumou0319-/WholeBodyTracking/runs/c9b21365) | 1912124 |

三组统一 seed 42、4096 environments、原始 reward、Paper-DR、30000 轮，最终 checkpoint 为 `model_29999.pt`。第 1/2 组学习参数相同；短跑 checkpoint 的递归配置比较仅发现 motion 路径和运行/日志标识不同。训练 W&B 于启动时已通过服务端历史读取验证，不是仅创建空 run；该验证不代表未来网络始终正常。

- 持久会话：`tmux attach -t sub1_largetable_028_3groups_20260917`；监视器启动 PID 1912047。
- 实验根目录：`exp/training/sub1_largetable_028/officialpt_compound1mm_20260917/`。
- 配置/启动：根目录 `production.json`、`launch_production.sh`；日志 `supervisor.log`。
- 每组状态：根目录 `production/groupN/state.json`；训练日志 `production/groupN/train.log`；checkpoint 在对应 `train/WholeBodyTracking/` 下。
- 正式评测输出：`exp/eval/sub1_largetable_028/officialpt_compound1mm_20260917/`。
- 实验说明及质量限制：根目录 `EXPERIMENT.md`；输入与网格哈希：`production/input_sha256.json`。

### 已解决的问题与本次代码改动

1. **[已解决] OMOMO pose 打包、资产原点与人体关节列错误。** `prepare_batch_retarget_inputs.py` 修正 InterMimic xyz+xyzw 布局、资产中心化后的平移补偿、reader 往返与单位四元数校验；`smplh_joint_order.py` 统一 native→retarget 52 关节映射，renderer/exporter/semantic loader 校验 layout marker。见历史 13.9–13.11。修复这些字段不等于旧衣架自由物体交互已成功，旧错误轨迹仍不得复用。
2. **[已解决] semantic plan 执行与旧缓存误复用。** 动态计划支持顶层数组，校验重复/坍缩窗口、时间线及映射；resolver 调用动态执行器。batch retarget 记录输入、scene、plan、命令指纹，拒绝无 provenance 或输入已变的旧结果；增加 task/method/tolerance 筛选与诊断入口。
3. **[已解决] largetable Original 第 194 帧不可行。** 单凸包填满桌底空腔，失败姿态 ankle/table 距离为 -5.21 cm；同一姿态改 32 块 VHACD 后为 +20.33 cm。新增 `scripts/build_omomo_compound_collision.py`，生成共用的 MJCF/URDF 碰撞块；两种重定向及三组 RL 共用该模型。Original/B4 均完成 236/236 帧，未关闭约束或放宽原 1 mm 容差。转换后均 392 帧 / 50 Hz，object 轨迹一致、有限数值和单位四元数检查通过；训练有效 horizon 为 391/50 = 7.82 s。实际 Isaac USD 确认每个物体含 32 个启用的 convexHull collider。
4. **[已解决] 同 seed 并发 USD 转换冲突。** 机器人转换目录增加 PID；物体转换使用 `tempfile.mkdtemp` 分配独立目录，避免 IsaacLab 秒级时间+相同随机 seed 造成写入冲突。修改位于 `isaacsim.py` / `object_spawner.py`；不改变 PPO 或物理参数。
5. **[已解决] 训练结束后没有自动评测。** 新增 `scripts/run_three_group_experiment.py`，每组训练成功退出且最终 checkpoint 存在后，立即独立执行评测、汇总、离线视频和 W&B 上传，不等待另外两组。用锁防止重复监视器，保存命令/PID/退出码/阶段，校验输入哈希（含 URDF 引用网格）。训练失败不会误触发评测；失败状态需先调查，不会盲目重训。渲染/上传失败保留已完成物理指标并标记待处理。
6. **[已解决] 评测 GUI 慢与无用尾部 rollout。** wrapper 显式设置 `training.headless=True`；instrumentation 每 100 步输出进度，全环境满 10 回合后通过 PPO callback 正常保存退出；分析器缓存 NPZ 数组，修正 clip 回绕后 stale timeout flags 的判断。离线渲染脚本及多任务结果汇总均保留。

修复与重定向证据：`exp/retargeting/sub1_largetable_028_officialpt_compound_20260917/`，包括 collision manifest、失败姿态对比、两条原始轨迹及 Isaac USD 审计。转换原始 retarget NPZ 时使用 xyz+wxyz 布局，**不要传 `--use-omniretarget-data`**。

### 评测流程与已完成结果

正式评测按每组自己的最终 checkpoint 和训练 reference：seed 42，32 环境，每环境前 10 个已结束回合（失败也计入，共 320），从 frame 0 开始，horizon=`(N-1)/fps`；initial pose noise=0；Paper-DR push 开启（间隔 1–3 s，线速度 0.3 m/s、角速度 0.78 rad/s）；object termination 1 m / π/4，其余正常终止条件保留。shape scaling 未实现，不能声称覆盖。headless 无实时相机，保存全环境 NPZ，再离线生成 env 0 首个 episode 的 actual/reference 视频并上传评测指标。

- **[已完成] largetable 流程短跑**：`smoke_v2` 的三组均完成 2 轮小训练→各 320 回合评测→汇总→视频，supervisor 正常退出。短跑失败率不是正式实验效果。
- **[已完成] sub10_largebox_089 上一轮最终评测**：官方输入/default1mm 三组分别 313/320（97.81%）、318/320（99.38%）、319/320（99.69%）。结果根目录 `exp/eval/sub10_largebox_089/20260916_officialpt_default1mm_v1_paper_dr/`（实际评测于 09-17 执行）；W&B eval [wfr4ll0j](https://wandb.ai/yumou0319-/WholeBodyTracking/runs/wfr4ll0j) 已完成并验证上传。

### 仍未解决或尚未覆盖

- largetable 孤立帧仍有 wrist/table 几何穿透：Original 最大约 9.49 mm，B4 14.07 mm；求解器成功不等于完全无穿透。共享碰撞模型改变了实验物理条件，不能与历史 single-hull 条件混作同一实验。
- 衣架旧任务的 hand/contact 对应与自由物体交互仍有问题，不能因 largetable 成功开训而标成已解决。
- 39 任务 readiness 审计见 `exp/training/readiness_20260917/READINESS.md`、`readiness.json`、`NEXT_TASK.md`；20 条官方 OMOMO PT 可读且校验通过，semantic 为 10 一致、6 待重新 resolve、4 无效。19 条 LAFAN 原输入和既有 Original/Uniform2 可读，但旧 stride20 与 semantic B4 三组协议仍待统一。不要宣称全部 39 个任务都 ready。
- 正式 largetable 三组仍在训练，最终评测待各组完成。独立 tmux 可跨对话持续运行，不保证机器重启后自启动。

### 提交范围与验证

本次验证：redman 环境下 pose packing、input provenance、SMPL-H joint order、semantic keyframes、semantic retarget runtime 共 **60 项测试通过**（6 条第三方弃用 warning）；**41 个 Python/Shell 文件语法检查通过**。仿真端完整三组短跑已于 09-17 验证，本次文档/提交操作不重启训练。

本次提交覆盖当前源码、脚本、回归测试和本文。生成的模型/视频/NPZ、训练日志、W&B 缓存、OMOMO 网格、分卷数据、`exp/` 本地产物不纳入。既有 652 个 tracked deletion（旧 benchmark/diagnostics 迁移及 `THIRD_PARTY_LICENSES`、`uv.lock` 等）保留在工作区，未作为本次代码删除提交；不要 `git add -A` 或清理工作区。

推送目标为用户仓库 `seexb/main`（SeeXB/holosoma），不是 Amazon `origin/main`。提交前 HEAD 为 `d3036835`，远端检查为 `692ba2a1`，本地另有 `7bcea974`、`d3036835` 两个已有提交随本次一并推进。最终提交号以 `git log -1` 和 `git ls-remote seexb refs/heads/main` 为准，避免在提交文件中写自身哈希。

> 更新时间：2026-09-18（Asia/Shanghai）；下方编号章节为历史记录，当前状态以上方最新交接为准。
> 下一位助手：开始工作前请完整阅读本文。仓库中有大量已经完成的因果实验和本地结果，不要从头重跑，也不要删除未提交的实验产物。

## 1. 仓库和 Git 状态（历史快照，最新见顶部）

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

### 13.8 Clothesstand 三组训练暂停与 8k/12k 评测（2026-09-05）

用户要求先停止等待训练，测试 10k 附近效果后再决定训练长度。三个 `sub9_clothesstand_058` run 均以 4096 env、seed 42、目标 30000 iterations 从零训练，在线 W&B 已启用。运行约 43.5 小时后按用户要求 SIGINT 停止，最后日志 iteration 为 17456 / 17458 / 15755；磁盘可恢复 checkpoint 为 16000 / 16000 / 12000。未保存的后续 iterations 没有恢复点。训练并非 OOM 停止，目前未重启。

Run 目录前缀为 `logs/WholeBodyTracking/20260903_093625-sub9_clothesstand_058_`，后缀依次为 `originaltraj_originalrl_s42-locomotion`、`semanticb4traj_originalrl_s42-locomotion`、`semanticb4traj_semanticadaptive_s42-locomotion`。训练 W&B IDs 为 `sspfcd7i` / `n4j5unuh` / `i52zffor`。

保存间隔为 4000，因此不存在精确 10000 checkpoint。最终对三组各自的 `model_08000.pt` 与 `model_12000.pt` 完成评测，不能把这些结果称为精确 10k。沿用 Paper-DR 的 32 环境 × 前 10 完整 episode、seed 42、push 开启、initial-pose noise=0、object 阈值 1 m / 45°。本任务 reference 是 209 帧、50 fps，完整评测 horizon 为 4.16 秒（208 个有效 motion frames），不能照抄 largebox 的 6.48 秒。第 1 组用 Original reference，第 2/3 组用 B4 reference。

六个 checkpoint 均为 **0/320 成功**。8k/12k 平均 episode 时长分别为：第 1 组 0.292/0.269 s、第 2 组 0.256/0.276 s、第 3 组 0.631/0.605 s。主要失败为 `bad_motion_body_pos` 和 `bad_object_ori`；1920 个计分 episode 共只有 1 次 push，绝大多数在初次 push 前就早停。目前不支持“以后只训练 10k 就够了”的结论，尚未修改训练预算；下一步应定位动作起始段/接触失败原因，尚未建立根因。

最终产物位于 `exp/eval/sub9_clothesstand_058/20260905_headless_isolated_8k_12k/`，包括 `ASSESSMENT.md`、`CHECKPOINT_EVAL_REPORT.md`、`checkpoint_results.json`，每个 `groupN_08000` / `groupN_12000` 目录中有 raw NPZ、eval log 和离线 actual/reference 对比视频（env 0 首个 episode、0.5 倍速、末帧停留 1 秒）。评测 W&B: `yumou0319-/WholeBodyTracking/k23m695h`。

本轮修正了两个评测执行问题：旧 wrapper 只传 `eval_overrides.headless=True`，实际 `training.headless` 仍为 False，导致 GUI 评测极慢；现已显式传 `training.headless=True`。独立进程全部 LOCAL_RANK=0 时强制 USD 转换共享 `converted_rank0` 会产生临时 layer 冲突，曾使一组评测崩溃；现转换目录包含 PID。最终六组均在 headless、无相机、独立转换目录下重新执行并正常退出。旧训练是否受共享目录影响尚未查明，不要当作已经证实的失败根因。导入器仍存在非致命 visual-reference warnings，不应误称所有 warnings 已消失。

新增入口 `scripts/eval_sub9_clothesstand_checkpoints.sh`、`scripts/analyze_clothesstand_checkpoints.py`、`scripts/render_clothesstand_checkpoint_eval.py`。Paper-DR instrumentation 每 100 步报告进度，全部环境满 10 个 episode 后通过 callback stop 正常保存退出，PPO eval loop 现支持该 stop 标记；统计器缓存 NPZ 数组避免反复解压，并只对真正回绕到 frame 0 的 timeout 忽略 stale flags。当时源码/脚本/本文尚未提交；现已纳入 2026-09-18 的代码提交范围，其他产物变动保留。

### 13.9 2026-09-05：衣架失败诊断发现本批 OMOMO pose 打包错误（字段问题已解决，见 13.10–13.11）

用户要求检查仿真/参考回放是否能够真实交互。本轮为诊断，没有修改生产输入打包器、URDF 或训练轨迹，没有重启训练。主要证据与后续顺序见 `exp/eval/sub9_clothesstand_058/20260905_reference_physics/DIAGNOSIS.md`。

**确定错误**：`tools/prepare_batch_retarget_inputs.py:66` 将 GT `[qw,qx,qy,qz,x,y,z]` 写成 `[qx,qy,qz,x,y,z,qw]`，但 `src/utils.py:45` 的 InterMimic reader 实际期待 `[x,y,z,qx,qy,qz,qw]`。读取结果变为 `[qw,x,y,z,qx,qy,qz]`。正确 writer permutation 应为 `[4,5,6,1,2,3,0]`。当时尚未实施；后续已修复并通过回归检查，见 13.10–13.11。

只读审计入口 `scripts/audit_omomo_object_pose_roundtrip.py`：全部 20/20 OMOMO 都受影响，三种方法 60 份 retarget NPZ 的物体四元数逐元素等于错误输入；human joints 正确，内存中正确 permutation 的完整往返误差为零。衣架输入的物体平移误差第一帧 1.760736 m（重定向尺度处理前），错误四元数范数 1.198779–1.857313。完整 `object_pose_roundtrip_audit.json` 已落盘。此前含这些结果的 39-task 汇总不能继续作为有效方法比较，需要重算受影响 OMOMO 与 aggregate；19 条 LAFAN 和此前独立成功的 sub3_largebox_003 不属于这次错误路径。

物理测试入口 `scripts/diagnose_clothesstand_reference_physics.py`，使用原训练场景、名义 mass 0.1 kg、关闭 DR/pushes、6 环境，不调用 RL、不提前重置。物体只初始化一次。有效输出 `default_checked/`：强制机器人每 5 ms 跟随参考而物体自由时，Original/B4 在 0.46/0.44 s 超过物体 45° 朝向阈值；仅 PD 跟踪在 0.38/0.40 s 超限。`decomposition_checked/` 中仅诊断实例改凸分解，强制跟踪仍均在 0.40 s 超限；生产转换器不变。原 USD 实际 `convexHull`，静/动摩擦实际 1.0/1.0。RL 为半球手，retarget 为 rubber hand；初始共同 wrist FK 差约 3.33 mm、torso 10.74 mm。参考第一帧衣架 OBJ 最低顶点 z=-0.105718 m。

注意 `default/` 初次新增诊断错用 raw body 0(world)，其数值无效，已有 INVALID.md；`default_validated/` 为 FK 检查拒绝的无效尝试。只用 `default_checked/` 和 `decomposition_checked/` 的结果。生产训练加载器原本按名称找 pelvis，并无该诊断脚本错误。新增脚本已修正并加 root/FK 校验；关闭 IsaacSim 必须使用已有 close_simulation_app workaround。视频用 MuJoCo 原 retarget 视觉模型离线展示 IsaacSim 实测状态，不是碰撞网格渲染。

本轮新增诊断脚本与报告尚未 commit/push；保留其他 dirty 文件。两次有效仿真均已正常退出，最终无 train/eval/diagnose 进程残留，显存约 1.4 GiB、可用内存约 42 GiB。

### 13.10 2026-09-06：字段与资产原点修复完成，最终交互尚未跑通

用户授权修复并检查交互。完整报告：`exp/eval/sub9_clothesstand_058/20260906_pose_fixed/FIX_AND_INTERACTION_CHECK.md`。

已修改 `tools/prepare_batch_retarget_inputs.py`：正确 PT permutation、实际 reader 往返校验；另发现**网格中心化后物体 pose 未同步补偿**，现新增 `canonicalize_object_poses`，用 `t_asset=t_raw+R(s_frame*c_source-c_asset)`，并验证 canonical OBJ 与 source 顶点对应/统一缩放。衣架原点遗漏约 0.352 m。Reader/转换器拒绝非法四元数；FK 转换器支持 `--scene-xml-file`；batch runner 支持任务筛选/显式诊断 step size，并以输入/场景/plan/命令指纹阻止误复用旧结果。

最终 **v3** 输入为 `exp/retargeting/omomo_pose_origin_fixed_20260906/input/`，20/20 完成并通过校验，根目录有 `roundtrip_audit.json`。固定尺度造成的最大顶点误差上界衣架 0.779 mm、20 条最大 tripod 4.685 mm，远小于原点错误。

**重要：交互未修复成功。** v3 输入、原单凸包、step=0.2 下 Original/Uniform-2/B4 在零基 frame 1/3/3 不可行；共享 64 块 VHACD 碰撞、step=0.2 时在 frame 7/8/8 不可行；compound+step=1.0 仅诊断也在 frame 72/9/9 不可行。所有日志/summary 保留在 v3 根目录及 compound、compound_step1_diagnostic 子目录。没有最终完整原设置轨迹，因而没有最终 v3 的完整自由物体测试结果，不要声称“已经拿得住”或“可开始训练”。

v3 `constraint_audit_final.json` 是 Original frame 1 最终失败的子问题：分别移除 foot_sticking/nonpenetration/step_bound 任一组变 optimal，移除 joint_limits 仍 infeasible，说明组合冲突。单凸包下右手/拇指距离约 -244/-260 mm，不是已证实的真实扫描表面穿透。不要沿用更早中间版本“已排除 foot”结论。下一步是初始化、接触定义、retarget/RL 手几何一致和约束可行性，不应偷偷关闭防穿透或变更 B4 方法来让程序跑完。

**中间版本不得混用**：`exp/retargeting/omomo_pose_fixed_20260906/` 是仅修字段的 v2，原点仍错。在这阶段 compound+step1 下 Original/Uniform-2 曾求解完成并做了物理测试（0.46/0.52s 朝向超限），其视频/NPZ在本 eval 根目录，已由 `STAGE_A_ONLY.md` 明确标记。这些不是最终修复后结果；原始 `omomo_batch` 则是 v1。旧训练 motions 已加 `DO_NOT_USE_FOR_NEW_TRAINING.md`；旧训练入口仍引用它们，不能直接启动。没有覆盖旧结果/资产/checkpoint，没有重启 RL，没有重跑全部39任务，也没有 commit/push。

42项测试通过（7项 packing/origin + 2项 cache + 33项 semantic runtime，redman环境）；hssim缺cvxpy不能用于完整runtime suite。py_compile/diff-check通过。诊断脚本与新代码均保持未提交状态；保留其余用户dirty文件。

### 13.11 2026-09-06：发现并修复 OMOMO 人体关节列顺序

13.10 的“最终 v3”结论已被取代。根因是 `prepare_sequence.py` 直接保存
`BodyModel.Jtr` 的 SMPL-H native 顺序，却从 7bcea97 起给它贴上 OmniRetarget
顺序的名字；`prepare_batch_retarget_inputs.py` 又原样打包。已知成功的
`demo_data/OMOMO_new/sub3_largebox_003.pt` 于 2026-01-12 已入库，早于 2026-09-02
新增的 renderer/adaptor 路径，并且关节拓扑正确。正确 permutation 与既有
`export_cari4d_retargeting_input.py` 一致：`[0,1,4,7,10,2,5,8,11,...]`。

修复详情和结果见
`exp/eval/sub9_clothesstand_058/20260906_joint_pose_fixed/JOINT_ORDER_FIX_AND_INTERACTION.md`。
新增共享 `smplh_joint_order.py`、显式 layout marker 和拓扑 guard；20 条 OMOMO bundle
已重建，dynamic plan 已按修正关节重新 resolve，新 v4 PT root 为
`exp/retargeting/omomo_joint_pose_fixed_20260906/input/`。resolver 也已改为调用
dynamic action/function executor，而非旧固定八角色 executor。46项相关测试通过。

衣架在 64-part compound、原 step=0.2 和全部原物理约束下，Original/B4 均成功
完成 126/126 帧（此前错误版 frame 7/8 infeasible），参考视频姿势正常。但 IsaacSim
prescribed-reference/free-object 测试仍失败：两者最终 object position error 约1.620m、
orientation error约168.6°，与 object-only 近似；训练 half-sphere hand 到物体表面
主要有约7–22cm间隙。当前剩余问题是 retarget 接触目标/手几何对应，不是人体关节列。
未启动RL，旧训练 motions 仍不可用。
