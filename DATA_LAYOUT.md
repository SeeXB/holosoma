# 数据与实验结果目录

从 2026-09-19 起，实验输入统一按以下规则存放：重定向输入放 `demo_data`；仅用于 RL 的输入放训练 `data`；两者共用的数据放 `demo_data`。`exp/` 存运行结果、日志、checkpoint、诊断与报告。

## 重定向及共享输入

根目录：`src/holosoma_retargeting/holosoma_retargeting/demo_data/`。

```text
demo_data/
├── omomo/
│   ├── official_inputs/             # 官方 InterMimic PT + scenes
│   └── bundles/<task>/
│       ├── input/                   # OMOMO GT NPZ、输入元数据
│       ├── semantic_keyframes/      # resolved semantic JSON + event_plan JSON
│       └── videos/                  # 用于 semantic planning 的 rerender 视频
├── retarget_inputs/<version>/       # 各版本的 input/、共享碰撞 assets/
├── models/                         # 重定向机器人/物体网格、URDF、MJCF、模板
├── semantic_keyframes/             # 既有通用计划
│   └── frozen/<task>/<experiment>/  # 当前实验冻结的共享计划
├── lafan/                          # LAFAN 原始 NPY
├── lafan_bvh/                      # LAFAN BVH
├── OMOMO_new/                      # 既有 demo 输入，区别于 official_inputs
├── body_models/cari4d_smpl/         # 已有 SMPL/SMPL-H 数据
├── pretrained/                     # 本地 CARI4D 上游模型缓存
└── external/omomo -> 外部数据目录    # 本机 /mnt/sdadrive/shixiongbo/omomo
```

`sub3_largebox_003` 的 bundle 名仍为 `sub03_largebox3`，其余任务通常与任务名一致。
仓库外的数据继续使用软链接，不复制进仓库；其他外部模型缓存链接也保留。新机器应重新链接自己的数据盘位置。

`retarget_inputs` 保留版本名，防止把旧错误输入和修复后输入混用。`omomo_batch`、仅修 pose 的 v2、v3 等历史输入不因搬家变成可训练数据；任务 readiness 和原 provenance 仍有效。

## 仅训练需要的输入

根目录：`src/holosoma/holosoma/data/`。

```text
data/
├── motions/
│   ├── tasks/<task>/<version>/      # 已转换的 RL Original/B4 参考 NPZ
│   ├── benchmarks/<benchmark>/      # 既有 benchmark 转换后的 RL 参考
│   ├── retargeted/<version>/        # 历史物理诊断使用的转换后参考
│   └── g1_29dof/                   # 仓库已有训练 demo
├── semantic/<task>/                # 仅用于 RL 的派生 sampler plan
├── experiments/<task>/<experiment>/# 使用新输入路径的配置模板
├── robots/                         # 训练机器人资产
└── scene_objects/                  # 训练专用场景资产
```

共享的 semantic plan、物体碰撞资产不再放一份到训练目录；训练配置直接引用 `demo_data`。

## 结果留在 exp

- `exp/retargeting/`：重定向输出 qpos、性能/穿透统计、诊断与 ALL39 历史汇总。
- `exp/omomo_cari4d/`：相机搜索、渲染帧、Blender 场景、VLM 原始回答与日志。供 semantic generator 消费的编码视频已归入 `demo_data/omomo/bundles/.../videos/`。
- `exp/training/`：训练日志、checkpoint、运行状态及冻结配置快照。
- `exp/eval/`：rollout、评测指标和 actual/reference 视频。
- `exp/data_layout/`：本次迁移记录与校验报告。

配置在 `data/experiments/` 中作为启动模板，在 `exp/` 中作为历史运行快照，后者不修改。新实验应选择独立的 tag / work_root / eval_root。

## 当前训练兼容性

旧输入路径保留了指向新目录的兼容软链接；因此 `exp/` 中仍能看到少量 `input`、`motions`、semantic JSON 等链接，但数据实体已移出。它们用于当前进程、旧 checkpoint 的绝对路径及历史 provenance，不作为新实验默认入口。**当前三组 largetable 训练和自动评测完成前，不要删除这些链接。**

当前正在运行的配置仍是 `exp/training/sub1_largetable_028/officialpt_compound1mm_20260917/production.json`；它是冻结快照，不要拿新路径模板覆盖它或重启同一实验。新路径模板位于 `data/experiments/sub1_largetable_028/officialpt_compound1mm_20260917/production.json`。

迁移没有改动轨迹、semantic 或当前共享碰撞文件内容；38 个当前训练依赖文件的 SHA256 保持不变。现存 URDF/XML 及元数据中的旧引用通过兼容链接解析；新生成的输入使用新路径。跨机器搬迁时仍需调整文件中的绝对路径和外部软链接。

## 后续生成与迁移

- `prepare_sequence.py` 默认输出 bundle 到 `demo_data/omomo/bundles`。
- `generate_omomo_semantic_plans.py` 的 plan 输出在同一数据根下，`--audit-root` 默认留在 `exp/omomo_cari4d`。
- `run_omomo_batch_retarget.py` 默认读取 `demo_data/omomo/official_inputs`，求解结果仍输出 `exp/retargeting`。
- 自建 PT adapter 与 LAFAN 预处理输出位于 `demo_data/retarget_inputs`。
- `build_omomo_compound_collision.py --output` 应传 `demo_data/retarget_inputs/<新版本>`；该目录的 `assets/` 和 `input/` 是共享输入。
- 转换器的 `--output-name` 应传 `data/motions/tasks/<task>/<version>/<name>.npz`。
- 原始 retarget qpos 作为阶段结果保留在 `exp/retargeting`；通过转换/验证后生成供 RL 使用的独立输入文件，不能把两种 NPZ 布局混用。

本次一次性迁移脚本为 `tools/reorganize_experiment_inputs.py`，默认只打印计划，`--apply` 执行移动。清单已保存到 `exp/data_layout/migration_20260919.json`。数据目录的新增大文件和外部链接已设置 Git 忽略，迁移结果需要单独传输，不会随 clone 自动获得。
