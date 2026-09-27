# Holosoma Semantic Retargeting / WBT 交接文档

## 2026-09-27 精简起止帧v2：删除rationale/criticality，新批次6/12定位通过

用户要求VLM可以内部思考但不输出理由，删除rationale及废弃critical等冗余，再看结果。已确认当前B4从runtime.load_semantic_plan_projection读取并忽略criticality，RL semantic_transition_sampler不使用此字段；历史precision仍计算加权指标，所以仅保留历史读取/重放，不破坏旧报告。修改adjacent_delta_boundaries.py：SCHEMA升为holosoma.adjacent_delta_boundary_plan.v2，prompt adjacent_delta_compact_v2；新动作严格仅action/body_parts/start/end，模型输出与最终JSON都没有rationale、criticality、criticality_level/source或占位权重。旧v1仍走兼容分支且18份候选与原记录完全一致。pipeline.validate_semantic_keyframe_json增加显式require_criticality=False，身体部位/帧校验仍在；默认旧格式严格校验不变。pipeline分派同时支持v1/v2。说明docs/adjacent-delta-keyframes.md已更新成占位符模板，不再使用left_hand示例。

重复的54身体信号描述合为18部位×height/speed/object_distance，再列12其他信号；66信号集合和相邻帧/全片幅度公式、独立起止匹配逻辑均不变。恢复逐部位内部可见证据、不能默认邻近/父部位的通用要求，但不要求输出说明。没有任务专用答案。每任务基础提示词14920→3948字符（-73.54%）；12项最后回复12150→6311字符（-48.06%，字符不是token且事件数量也变了）。本轮同时改输出理由、提示词长度和部位指令，不能视作仅删rationale的单因素试验。

新产物 `exp/semantic_plans/adjacent_delta_keyframes_compact_v2_qwen35_20260927/`，REVIEW.md入口，review_summary.json比较逐任务/部位/候选历史。12项全部运行结束，qwen3.5-27b、同Aliyun host、整段原MP4、temp0 seed0、37请求37回复，每项<=4；最后12份都格式有效，但6/12完整定位（9事件18边界）。通过whitechair118、trashcan024、woodchair004、suitcase053、plasticbox077、suitcase031；失败largebox003/089、smallbox023、plasticbox013、woodchair020、largetable028。没有追加额外数值反馈或人工阈值修正，旧版sub15的额外反馈结果保留。

失败均有阈值超范围：largebox003/089用object_displacement单步+3%，可观测最大2.4094/2.6042%；smallbox高度+5%vs2.8576%；woodchair020+5%vs3.6128%；largetable+5%vs4.0108%；plasticbox013下降8%vs4.9508%。largetable最后候选只一个lift table事件，body_parts=双hand/双elbow/双knee/waist；选回elbow不意味着语义正确。12份候选共21动作，19选双手、18选waist、4选双elbow、5选双knee等；未通过项也纳入候选部位对比，不与6份有效导出混淆。

全部12份最后合法候选在candidates/omomo/<task>/candidate.event_plan.json及resolution.json；仅6份完整通过在plans/omomo/<task>/semantic_plan{,.event_plan}.json；失败audit原文都保留。没有裁剪动作或短事件/语义表述拒绝。150项测试通过（144生成/旧规则+6 active criticality兼容），37份新候选全部重放一致，新输出通过active retarget projection。1841保护文件及生成源码SHA未变；session2022正常exit0，state=complete_reviewed。旧12/12批次、canonical/训练输入未覆盖，未干预RL，未commit/push。

## 2026-09-27 plasticbox013用户授权追加重试成功，总计12/12通过

用户明确要求sub15_plasticbox_013多重试几次，追加最多4次、成功即停；实际仅追加1次（该任务总第5次）即通过。仍qwen3.5-27b、同host、整段原MP4、temperature0/seed0、相同基础提示词与起止帧执行器。本次修复反馈增加失败信号的实际归一化delta范围[-0.04950806355018531,0.01756545101958051]及无匹配计数，没有提供替代阈值、目标帧或时间区间。模型自行把carrying_box结束规则object_height decreasing阈值0.05改为0.03，起点steady0.02保持；本地求得start1/end63/trigger1@30Hz，末边界实际ratio=-0.03225370309587365。未手工改计划、加容差或取最近帧。

目录仍 `exp/semantic_plans/adjacent_delta_keyframes_v1_qwen35_20260927/`：新原文/提示/plan/resolution为audit/sub15_plasticbox_013/attempt_4.*，完整输出在plans/omomo/sub15_plasticbox_013/。retry_round1_sub15_plasticbox_013/保存追加授权、实际错误反馈、此前11/12报告与原失败结果，retry_sub15.py为本次一次性重试入口，不要重复执行。新增GT数值可行性反馈与原先只回传“无匹配”的条件分开记录；不是完全同一提示词的独立重采样。

当前12/12通过格式与起止帧定位，12任务18请求18回复、20事件40边界；18份候选全部重放一致，58条候选边界各一个数值阈值，133测试通过的生成源码未改。1666原保护文件及本轮1808保护文件SHA均未变，其他11任务结果未动。session58568正常exit0；state=complete_reviewed，REVIEW.md/review_summary.json/verification.json/generation_summary.json/process_completion.json均更新。没有语义表述/短事件拒绝；未替换canonical/训练输入，未干预RL，未commit/push。

## 2026-09-27 起止帧delta方案已完成其余10项，总计11/12生成定位通过

用户要求不纠结“减速解释成停止”的语义表述，继续完成其余任务。沿用上一节当前起止帧代码、qwen3.5-27b、同Aliyun host、整段原MP4、temp0 seed0、每任务最多1+3次请求；没有调整提示词/阈值规则。产物仍为 `exp/semantic_plans/adjacent_delta_keyframes_v1_qwen35_20260927/`。原两项audit/plan/result共20文件SHA未变，旧两项报告归档pilot_review/，本轮范围/保护SHA在continuation.json。

新增10项中9项通过1项失败，总计12项全部尝试、11项格式与起止帧匹配通过；17请求17回复（本轮15、原2），生成session78093正常exit0。通过11项19事件38边界：largebox003=3、smallbox023=2、largebox089=3、whitechair118=1、trashcan024=1、woodchair004=2、suitcase053=1、woodchair020=2、largetable028=2、plasticbox077=1、suitcase031=1。单帧/相邻帧以及语义表述不作为失败条件，未手工裁剪或删事件。largebox089和woodchair004各2次请求，其余通过项首轮。

唯一未定位：sub15_plasticbox_013，4次回复结构化计划完全相同，carrying_box起点object_height steady0.02能匹配，但终点object_height decreasing0.05全片无候选；实际最负比例-0.04950806355018531（下降最大4.9508%，略低于5%阈值）。这是数值谓词无匹配，不是语义表述/持续时长拒绝。没有放宽阈值、近值回退、伪造完成或追加第5次请求；所有候选保留audit，不存在该任务可导出的完整semantic_plan.json。

根目录REVIEW.md、review_summary.json、verification.json已更新为12项，17份候选确定性重放与导出dispatch一致，56条候选规则每边界一个阈值，无VLM时间/帧号/区间/基准输出。1666保护文件、20原两任务文件及生成源码SHA未变；源码沿用已通过133测试的版本，本轮未改生成代码。state=complete_reviewed（所有生成尝试完成，不等于12/12通过）；process_completion.json记录进程退出。未覆盖canonical/训练输入、未干预RL、未commit/push。

## 2026-09-27 用户更正：delta分别定位起始帧/结束帧，不是连续满足区间

**当前规格覆盖下方相邻帧连续段方案。** 用户明确delta规则用于选起始帧和结束帧，不让VLM选择一个区间；抓取可能只是一对帧，不能因为事件短就判失败。此前把单个动作阈值的连续满足段直接当作整个动作，是助手理解错误。当前公式不变d[t]=(s[t]-s[t-1])/(max(s)-min(s))，但VLM输出独立start/end，每条规则各自signal/condition/threshold（一边界一个数、每动作两个数），无时间、帧号、搜索区间、基准或动作全程谓词。

新实现 `semantic_keyframes/adjacent_delta_boundaries.py`，schema `holosoma.adjacent_delta_boundary_plan.v1`，prompt `adjacent_delta_separate_keyframes_v1`；batch方法 `--method adjacent-delta-boundaries`，说明 `docs/adjacent-delta-keyframes.md`。旧adjacent_delta连续段模块/schema/CLI仍仅为归档复现保留，不要拿旧prompt再生成当前任务。pipeline增加新分派；batch快照同时保护共享adjacent_delta.py依赖。

本地按开始顺序求关键帧：start为前一动作start之后首个满足帧，end为当前start起首个满足帧。选当前帧t，[t-1,t]是证据，不把起点回退到t-1。中间帧无需满足任一边界条件；不提取连续段、不对delta再施加第三帧穿越条件，不设最短时长。start=end点事件、相邻帧事件均允许。允许事件重叠，终点不裁到下一个起点；某end失败不影响后续start顺序，某start失败才阻断后续顺序。没有部分计划导出、近值回退或事件删除。兼容criticality仍uniform level3/0.75且明确placeholder，未用于训练。

只试跑两视频（没有新跑全部12项）：`exp/semantic_plans/adjacent_delta_keyframes_v1_qwen35_20260927/`，REVIEW.md入口。qwen3.5-27b、同已授权Aliyun host、整段原MP4、temp0 seed0；两项均首轮格式有效且8边界都能定位，2请求2回复，无修复。largebox003：pick up start1/end34、carry start25/end43、put down start133/end170@30Hz，搬运未被删，但3D速度增减可来自抬升，尚非语义准确证据。suitcase031：pushing object start12/end24；模型将速度下降解释成停止，而end速度仍0.5775m/svs前帧0.6144m/s，理由错误。问题是信号解释/语义，不是事件太短。不要引用上一轮“抓取只有一对帧所以不可用”的说法。

133测试通过（含28项新边界测试），两份候选与官方dispatch确定性重放一致。1666保护文件及生成源码SHA未变，session89679正常exit0；state=complete_reviewed。实际提示词audit/<task>/base_prompt.txt、完整回复/plan/resolution、review_summary.json和verification.json已保存。未覆盖训练数据/旧计划、未干预RL、未commit/push。用户尚未要求将实验规则正式用于训练或重跑其余10项。

## 2026-09-27 固定相邻两帧差分、每动作一个阈值，qwen3.5 试跑完成

用户纠正比较基准：不让VLM选帧/时间/基准，delta固定当前帧减上一帧，每动作只输出一个量。新增 `semantic_keyframes/adjacent_delta.py`、`tests/test_adjacent_delta_semantics.py`、`docs/adjacent-delta-semantic-plans.md`；batch新增 `--method adjacent-delta`，pipeline分派schema `holosoma.adjacent_delta_plan.v1`。公式 d[t]=(s[t]-s[t-1])/(max(s)-min(s))，分母仍按用户选的全片原信号幅度，第一帧无差分、D=0未确认。VLM只选action/body_parts/signal/condition/threshold/rationale，唯一数字为threshold；拒绝reference/time/frame/start/end/criticality等额外字段。increasing为d>=tau，decreasing为d<=-tau，steady为abs(d)<=tau。保留GT原30Hz、不乘fps；没有平滑、噪声门槛、最短时长、滞回或近值回退。

程序按动作顺序取首个完整连续满足段，不切开已开始的段；差分步a..b导出姿态[a-1,b]、trigger=a。失败后其余动作blocked，不导出部分成功计划；所有候选保留。单步、多候选、全片成立均标记。兼容输出的criticality统一level3/0.75，有明确placeholder来源，不是VLM给的第二个量，未安装到训练。

产物 `exp/semantic_plans/regenerate_latest12_adjacent_delta_v1_qwen35_20260927/`，入口REVIEW.md；12项结束，32请求32回复，同已授权阿里云host、qwen3.5-27b、整段原MP4、temp0 seed0，每项最多4请求；没有客户端API抽帧，.env未改。7/12数值可执行，5未确认，最终12份均格式有效；七项22区段，中位16步、1项单步。不是语义准确率，不要重复发API或覆盖训练计划。

具体：sub3_largebox003接近[0,12]/抬起[33,53]/放下[132,151]，模型在修复时删除首轮carry，故语义不完整。smallbox023把lift/raise合为lift_box_overhead，最终两区段[25,43]/[60,99]。sub4_suitcase031高度幅度仅3.194mm，阈值5%相当于单帧0.160mm，grasp=[41,42]一对帧；数值匹配不证明抓取/放下。trashcan024抬/放仅4/5步。其余数值通过whitechair118、woodchair020、largetable028；失败largebox089、woodchair004、plasticbox013、suitcase053、plasticbox077，都是顺序约束下没有完整后续满足段。当前归一化仍会放大微小抖动、单阈值截取较快运动片段，未解决完整语义边界。

105测试通过；31份合法候选确定性重放一致，88动作均只有一个数字且无时间/基准，所有通过导出也通过官方dispatch重放。1449保护文件SHA、生成源码SHA未变，请求预算/模型检查通过。pilot session7529、其余11项session73404均exit0，复核脚本正常完成。旧计划、canonical、bundle、冻结训练输入未变，没有干预RL，没有commit/push。

## 2026-09-27 全片幅度差值比例 v1 + qwen3.5 试跑完成

用户认为局部静止基线/MAD方案超参数太多，要求信号改为差值比例、继续由VLM撰写逻辑；用户明确选择分母 max(s)-min(s)。新增 `semantic_keyframes/range_delta.py`、`tests/test_range_delta_semantics.py`、`docs/range-delta-semantic-plans.md`；既有 batch 入口新增 `--method range-delta`，pipeline 增加 schema `holosoma.range_delta_plan.v1` 分派。VLM选择粗时间范围、信号、reference（clip_start/range_start/action_start，仅end可action_start）、signed ratio和up/down。r=(s[t]-s[reference])/D，D全片幅度；正负与上/下穿独立。无平滑、噪声门槛、静止基线、最小速度、持续时间、滞回或近值回退；D=0未确认，微小抖动会被归一化放大，明确保留原始幅度。视频边界用简单字符串clip_start/end；格式校验一次返回全部独立错误。76测试通过。

产物 `exp/semantic_plans/regenerate_latest12_range_delta_v1_qwen35_20260927/`，入口REVIEW.md，review/summary.json与12张曲线图，audit全部提示词/回复/数值诊断。12项完成，39请求39回复，固定qwen3.5-27b，完整原MP4，同阿里云host，temperature0 seed0，每任务<=4；已结束，不要重跑该目录。只用进程内模型覆盖，未修改.env。

形式上4/12完整计划通过，但sub1_plasticbox077只有整段[0,162]的Lift and carry，两端均视频声明、没有任何GT比例判断，单列为纯视觉边界，不计定位成功。实际包含GT比例穿越的完整通过项仅3/12：sub3_largebox003三阶段[3,28]/[40,119]/[128,195]；sub10_largebox089[0,34]/[34,94]/[94,157]；sub11_trashcan024[7,53]/[84,123]，均30Hz。三项也未语义审核通过：sub3放下end延到片尾195而物体更早停止，trashcan名为lift的start由手距变化触发、物体当时尚未明显抬起。没有替换canonical或训练数据。

所有12个最后回复格式有效，最近候选60边界=23GT穿越+18视频截断+19未确认。19项均没有区间内匹配。smallbox1.2–1.6s选高度+5%而该段仍约0；plasticbox013结束速度要求-80%而可观测最大仅-7.4%；woodchair020选+95%但因为片尾高度低于片首，全片幅度分母下实际峰值相对片首只有约69%。sub4 suitcase高度幅度仅3.194mm，模型仍编lift/place，未通过。不能把归一化当作消除数值估计/语义错误的保证。旧基线方案1/12、旧分位数7/12（无回退/零长5），通过口径/提示及格式修复同时改变，不是严格单因素比较。

1204保护文件SHA未变、生成源码SHA与本轮快照相同；38份可解析候选确定性重放逐项一致，所有通过计划也通过官方dispatch重放。Miniconda hsretargeting，无GPU/训练干预、未commit/push。可继续离线检查已保存候选的实际比例范围，避免相同设置反复API调用；若要给VLM GT比例范围反馈，应明确记为新条件。

## 2026-09-27 VLM 粗区间/信号 + GT 定位 v1 已实现并试跑，效果尚差

用户授权实现 VLM 判断大致时间区间并选择信号、程序计算 GT 阈值，使用 qwen3.5 测试。新增 `semantic_keyframes/bounded_signals.py`、`tools/generate_bounded_semantic_plans.py`、`tests/test_bounded_semantic_signals.py`、`docs/bounded-semantic-plans.md`；pipeline 仅增加新 schema 分派，保留旧规则兼容。VLM 每边界输出 time_range_s/signal/transition，不输出物理阈值或 q；GT 平滑、有符号导数或速度、局部低活动基线/MAD 噪声、滞回/持续时间定位。配置为全任务统一初始工程值，未独立标注校准。禁止最近值回退、零时长、剪成单事件；视频首尾截断单独计数。52 项新/回归测试通过。

产物 `exp/semantic_plans/regenerate_latest12_bounded_v1_qwen35_20260927/`，入口 `REVIEW.md`，具体证据 `review/summary.json`、`review/<task>.png`、audit 每次完整提示/回复/校验/定位记录。12 项全部完成，47 请求/47 回复，固定 qwen3.5-27b，同已授权阿里云接口、整段原 MP4、temperature0 seed0，每项最多4次。视频完整解码验证了GT帧数/帧率；没有API抽帧，也没有修改.env。所有调用结束，勿重复跑该目录。

只有 sub15_woodchair_020 完整程序通过（第三请求）：举椅[0,37]、降低[61,122]@30Hz，3 GT边界+1视频起点截断；视频抽查提示漏举持阶段/降低起点可能过早，仍未批准用于训练。10项曾进入GT定位，9项最后回复格式有效；最近格式有效候选共64边界，13 GT确认、11视频截断、40未确认（25无可靠局部基线、15区间内无变化）。smallbox最后回复退回格式错误，统计使用其第2次可解析候选且明确标注；whitechair118、plasticbox013从未通过格式。主要协议错为object当身体部位、clip边界非null或非精确端点。

sub3最终7动作，抬起起点25帧和下降结束164帧成功，其余12边界未确认。方法自身对连续弯腰/行走要求安静基线过严，不能只怪模型；模型也有时间误差和错误动作/信号选择（sub4 suitcase仍套lift/place）。旧v3为7/12通过但允许回退/少量零时长，口径不同，不能据通过率直接声称语义准确率变化。全部候选/失败保留，未替换canonical或训练输入，未干预GPU0既有RL。943份保护文件SHA和实验源码SHA验证未变。下一步宜先用保存候选离线改善连续运动检测/边界协议，避免继续同设置API重试。源码尚未commit/push。

## 2026-09-27 threshold_only_v3 + qwen3.5-27b 生成完成

用户看到7B效果差要求不再追加、换qwen3.5-27b。最新目录exp/semantic_plans/regenerate_latest12_threshold_v3_qwen35_20260927/video_input/，REVIEW.md入口，MODEL_COMPARISON.md/json四条件对比。12项完成，7通过5失败，27请求全保存无超时；每项<=4。所有调用结束，state=complete_reviewed，不要重复启动。模型仅在本轮worker进程固定qwen3.5-27b，未改.env；同阿里云host、12完整MP4、GT、逐字相同v3基础提示词、66信号/4阈值规则、temp0 seed0及源码。765保护文件和源码SHA不变，进程退出见process_completion.json。

对比v3：2.5为2/12（无零时长/回退0），3.5为7/12（无两类问题5），3.6为10/12（无两类问题7）。3.5七项首轮通过，五项虽修改计划但4次后仍失败；3.6有4项经内容修复通过。3.5首次通过在共同有首回复11项中为7vs3.6的5，但最终率较低。失败为sub3_largebox003、sub14_woodchair004、sub15_suitcase053、sub15_woodchair020、sub1_plasticbox077，均晚/重合窗口。smallbox通过但crouch=[36,36]；sub15_plasticbox有1回退。sub4_suitcase通过只有drag一个事件，是原完整输出。

sub3诊断：lift选object_height上穿q=.15，GT阈值-0.0015298m，整段无上穿；执行器最近值回退166（实际为下穿），后续全部继承搜索下界166→三阶段均[166,166]。证据sub3_threshold_diagnostic.json。因此模型能力/规则选择有影响，分位数谓词+无匹配回退的设计也存在共同问题。单次小样本、实际服务端权重/取帧未独立核实，不宣称唯一因果或估算价格。GPU0接触奖励训练继续，canonical语义/bundles/冻结输入未覆盖。

## 2026-09-27 threshold_only_v3 + qwen2.5-omni-7b 对照完成，不再追加

目录exp/semantic_plans/regenerate_latest12_threshold_v3_qwen25_20260927/video_input/，REVIEW.md、MODEL_COMPARISON.md/json。42请求全部有回复，2/12通过（sub15_woodchair020、sub1_largetable028），两项都有最近值回退，woodchair还有place=[140,140]；通过且无零时长/回退为0。7B v2为5/12（干净3）、3.6 v3为10/12（干净7）。538保护文件及源码SHA未变，state=complete_reviewed；不重发本轮请求。

本轮42份回复无极值，但7项跨动作重复起点规则，78/86边界用q=.9/.1，smallbox对单调不减object_rotate找下穿，woodchair004用不存在object_horizontal_speed。10修复任务里8个所有归一化计划一致，6个原文一致；通常只有初始/修复两种提示，后续相同错误导致相同提示，不能当4次独立试验。3.6的6个多回复任务都修改了计划。sub3为lift[72,72]、carry[72,83]、place[83,168]，失败。结果支持7B配置的复杂规则执行/修复较弱，同时v3对7B较v2退步，不能把问题只归于极值。服务端模型权重/取帧/截断缓存未独立核实，audit记录请求model名。训练与canonical计划未改。

## 2026-09-27 smallbox 接触位置奖励 G3 第14000轮评估完成

用户要求现在eval。使用contact_surface_v1_20260926的model_14000.pt（SHA69263c4452d15346fdcc3cfbbefd628bb635951ae2e59cd81d1fd662814e5cb0），GPU0与训练并行，未停止/重启训练。沿用PaperDR、seed42、32env×各前10完整回合、frame0、初始噪声0、扰动开启、物体1m/45°、238帧4.76s协议。

结果294/320成功=91.875%，平均到达226.115625帧，中位/最远238；26次失败均bad_object_ori。物体位置RMSE0.0927606m、朝向9.35369°、身体点0.0789029m；旧G3 30k为0/320、平均135.390625、最远148、物体位置0.353754m、身体点0.0540716m。不同训练轮数/覆盖时段，不是同轮数因果消融，也非所有指标改善。300/320回合物体原点上升超过0.5m，抬升中位1.21789m；frame180高度中位1.278mvs参考1.377m。

产物exp/eval/sub7_smallbox_023/contact_surface_v1_20260926/interim_14000_20260927/，REPORT.md、comparison.json、evaluation_comparison.png；group3_14000/actual_vs_reference.mp4左参考右实际，env0首回合成功，0.5倍速238动作帧加1秒hold，263帧解码通过。MuJoCo离线回放IsaacSim状态，沿用可视化橡胶手，手部几何不等同训练半球手；不要从渲染表面推断接触力。OSMesa首次失败已留档，EGL重试成功，仅重渲染未重eval。输入哈希未变，state=complete，结果本地未上传。训练PID618508仍运行，最后核查14333/30000，不要重复启动。

## 2026-09-27 threshold_only_v3 新模型整段视频复跑完成

用户要求去掉极值再生成，并自行更换 OPENAI_MODEL、授权直接继续调用。新生成仅允许 threshold_crossing_up/down、sustained_above/below，保留66信号；pipeline新增generation专用校验拒绝4种极值，旧计划兼容解析未改。提示词版本threshold_only_v3，26项测试通过（含4种极值回复拒绝后修复测试）。源码及测试尚未commit/push。

最新目录 `exp/semantic_plans/regenerate_latest12_threshold_v3_20260927/video_input/`，入口REVIEW.md。仍同一12个完整GT MP4，`.env`每次请求重新读取OPENAI_MODEL；本轮实际均为qwen3.6-27b，仍同一阿里云host。每请求模型与状态在audit/<task>/request_attempt_N.json，不记录密钥。全部结束，不要重复发起本批请求，每任务总预算4次包含超时。

最终10/12程序通过、2失败；上一轮5/12，但规则和模型同时改变，不能当prompt-only ablation。29请求、25回复、4超时（woodchair004两次，suitcase053两次），均计入预算。woodchair第3请求通过；suitcase第4请求仍窗口挤在118–121/122帧、carry/place同为[121,121]；sub1_plasticbox077四次仍lift/carry同起118帧。两个失败完整保留rejected_candidates，没有剪成单事件。主batch PID668542和recovery PID671053均退出，process_completion.json已查证。

重点：sub3_largebox_003第4次通过，5阶段approach_and_grasp[14,24]、lift_box[24,59]、carry_box[59,87]、place_box[87,138]、release_and_stand[138,190]@30Hz；最后end的waist_height上穿q=.8未匹配，用最近值回退，不能当作已验证精确标注。smallbox为crouch/lift/hold_overhead三阶段，但crouch仅[2,4]。sub10_largebox placing=[127,127]、trashcan adjusting=[141,141]零时长，后者两端回退。10通过中8无零时长、2有阈值回退；程序ok不等于语义正确。所有25回复都没有极值规则。

compare.py、compare_prompts.py和verify_results.py均完成。COMPARISON.md对原训练文件，PROMPT_COMPARISON.md/json对v2并含逐阈值证据。final_verification.json确认342个protected文件及源码哈希不变、12项预算/输入/提示词/规则/元数据匹配。state=complete_reviewed。canonical语义/bundles/训练冻结输入没有覆盖，也没有新训练。此前v1/v2产物保护不变。旧执行器最近值回退与少量零时长可通过的问题尚未修改。

## 2026-09-27 提示词补全信号/规则定义后再次整段视频复跑完成

用户指出极值与q及各信号定义不清，要求修改再生成。已修改 `semantic_keyframes/pipeline.py`：新增完整66信号说明、8种primitive真实语义、极值不读q/threshold=null、分位数与时间/进度比例区别、纯数值示例、实际搜索/回退行为，修复提示同步澄清。版本 `signal_rule_contract_v2`，未加入任务答案或改变关节词汇引导。现有22测试通过，AST确认信号计算/归一化/执行/校验函数未改；数值例子实测吻合。源码尚未commit/push。

最新产物为 `exp/semantic_plans/regenerate_latest12_prompt_v2_20260927/video_input/`，看 `REVIEW.md` 和 `PROMPT_COMPARISON.md/json`（提示词前后），`COMPARISON.md`另为与原训练文件对比。12项同视频/同GT/同模型/同seed完整复跑结束，35请求全部保存，每项<=4。最后回复的极值附带阈值错误由9/12降到0/12，现有程序通过数4→5/12，通过且无零时长2→4，但尚不能直接用于训练。旧批次2次回复丢失是对照限制。

通过：sub7_smallbox_023(3事件)、sub10_largebox_089(3)、sub15_suitcase_053(2)、sub1_largetable_028(3)、sub4_suitcase_031(4)。sub10_largebox的place=[123,123]仍零时长；它与largetable各有2条阈值无匹配后的最近值回退。smallbox carry只有1帧；两个suitcase的语义/信号对应仍可疑。不能把程序通过数当语义准确率。whitechair、sub15_plasticbox从通过变失败，新增通过3项，非全体改善。

sub3_largebox_003仍失败：现在使用合法local_min/max，起点却落在170/196帧、carry/place同起177帧。其他失败主要重合/过晚窗口，sub1_plasticbox还生成不存在的object_horizontal_speed。未剪成单事件、未手工改事件/区间、未更换解析器。199保护文件(54原数据+145旧批次产物)全部SHA不变，canonical语义/bundles/训练冻结输入未覆盖，未启动新训练；新batch已结束，不要重复发送。本轮实际发送的仍是整段MP4，服务端采样未知。

## 2026-09-27 12 项 OMOMO 整段视频 semantic plan 重生成完成，未替换训练输入

用户质疑 sub3_largebox_003 只有一个事件，要求用当前代码重新生成 latest12 并对比，随后明确授权将现有完整 GT MP4 发送到已配置的阿里云 qwen2.5-omni-7b 接口。批次产物 `exp/semantic_plans/regenerate_latest12_20260927/video_input/`，入口报告 `REVIEW.md`、逐任务 `COMPARISON.md` 和 `comparison.json`。12 项生成尝试已结束：4 项程序校验通过、8 项无可用计划；不能声称新计划均可训练。

实际使用 video_url/base64 发送整段 MP4，无本地 API 抽帧；服务端取帧数未知。现有 dynamic pipeline 未改，HEAD282f4fb。官方 PT 导出的 GT 信号单独放在实验 inputs 中。36 个请求预算名额、34 份保存的回复；批处理 exit143 造成 sub10_largebox_089 最后一次和 sub11_trashcan_024 第三次回复缺失，均保守计入每任务最多4次预算。前者不补发第5次，后者完成剩余第4次仍校验失败；中断具体原因未知。不要再次运行任务超出预算。

关键根因：旧版9/12是 `tools/prune_dynamic_plan_failures.py` 在完整计划失败后截取最长有效单事件的产物，metadata为 `unchanged_valid_single_event_from_oversegmented_vlm_response`；不是视频只有一个事件。本轮11/12的最后完整候选有多个事件，未做singleton截取。sub3_largebox_003四次均识别 lift/carry/place_box（双手），但每个事件起止均为 global_minimum(object_height)，quantile无效、归一化后相同，故全部失败；完整候选保留rejected_candidates。

通过项：sub7_smallbox_023四事件、sub10_whitechair_118一事件、sub15_plasticbox_013三事件、sub1_largetable_028三事件。其中smallbox lift_box=[43,43]、largetable place_table=[188,188]为零时长；现有校验允许少量零时长。smallbox末尾place_box与预览仍举持箱子不符，whitechair单push可能漏阶段，plasticbox区间到57/110帧仍需核对。程序ok不代表语义正确，本批4份通过计划也尚未批准替换训练输入。模型/提示词/谓词生成和校验仍需后续修正；本轮未修生产代码或手工改动作。

54份protected文件及代码SHA复核全部未变，canonical语义、bundle、smallbox接触奖励G3冻结输入均保持原样，没有启动新训练。所有生成任务已结束，state=complete_reviewed；根目录早期抽帧state已标superseded，勿当成仍待授权。本地抽帧仅作视觉复核，未发送；.env密钥未记录进报告。

## 2026-09-27 用户更正为 sub3_largebox_003，两引擎视频已重做

用户明确此前要的是 `sub3_largebox_003`，不是 smallbox。最终交付应使用 `exp/eval/sub3_largebox_003/b4_cap10_right_side_20260927/` 中两个 `sub3_largebox_003_semantic_b4_cap10_{mujoco,isaacsim}_right_side.mp4`。325帧/50fps/1080p/6.5秒，完整解码和五关键帧预览通过。B4 cap10 reference SHA22d06ad853c89112878ce736e26294ccec94667ff0ed1dde56bf67173bbad54f，与本任务PROVENANCE匹配。使用本任务 largebox 模型；同一轨迹和相机，两原生引擎逐帧回放，无physics integration/policy。相机跟随pelvis保持机器人右侧水平正视，distance3.0m/target_z0.70m/fovy30°。详见目录README、metadata、validation。保留各自原模型的手部差异；未修改源数据/训练配置，未中断训练或上传。此前smallbox视频是任务误解的产物，不要再次当作本次交付。

## 2026-09-27 smallbox B4 cap10 两引擎右侧视频完成

用户要求 MuJoCo 和 IsaacSim 回放，机器人右侧正视。产物在 `exp/eval/sub7_smallbox_023/b4_cap10_right_side_20260927/`，两个 `sub7_smallbox_023_semantic_b4_cap10_{mujoco,isaacsim}_right_side.mp4`，均 1080p/50fps/239帧/4.78s。使用同一 B4 cap10 转换后参考 SHA a8f68171…，无 physics integration/policy；原 solver NPZ 缺失未重造。相机跟随 pelvis yaw、保持机器人右侧水平观察，eye/target 两端一致。

保留 MJCF 橡胶手 / IsaacSim 训练 URDF 半球手的既有几何，只有渲染副本路径、材质灯光地面调整。PhysX 关节和物体位置读回零误差；两既有模型共同连杆 FK 最大差 10.08mm、手腕 5.17mm，独立 URDF FK 复算吻合，见 model_fk_comparison.json。视频完整解码/预览通过，见 README.md、validation.json 与两份 metadata。仅使用 GPU0；未停止、重启或改动本机训练；未上传或 git 提交视频。

## 2026-09-27 重定向数据已同步至 aozhou，软链接及校验完成

用户要求将重定向数据补传到 aozhou@100.64.131.165。SSH BatchMode 认证已可用；目标 `/ssd-empty/shixiongbo/holosoma`，工作仓库 `/home/aozhou/shixiongbo/holosoma`。增量 rsync 正常退出，本次新增 1148 普通文件 / 8,492,527,454 bytes；全部清单 2527 普通文件 SHA256、6 相对链接均匹配，没有缺失/差异。24 个数据入口均指向 SSD（20 新建、4 沿用）；15 旧入口有 `.before-ssd-retarget_to_aozhou_20260927_021355` 备份；远端独有 7 个 LAFAN 文件已保留合并。未启动远端任务或停止/重启本机训练。

记录在 `exp/data_transfer/retarget_to_aozhou_20260927_021355/`（两端均有），详见 REPORT.md、manifest.json、verification.json。最新指针 `exp/data_transfer/LATEST_RETARGET_TO_AOZHOU`。包含官方 OMOMO PT、23 GT bundle、LAFAN、人体模型、对象/机器人资产、现有 SAM3/Sapiens 权重、本机 retargeting/历史 benchmark。不要重复传输。

用户追加询问原始 OMOMO 视频：本机 bundle 有 23 个 `_rerender.mp4`，已经同步，属于 GT 重渲染。metadata 的 23 个原始 source_video 路径均不存在，旧原始根 `/mnt/sdadrive/shixiongbo/omomo` 不在本机。Hunyuan3D-2 / UniDepth 缓存及 smallbox 最新 B4 原始 retarget NPZ 也缺失，未传；已转换的 RL NPZ 另有保存，勿混淆。历史 XML/URDF 的绝对路径仍需运行时核对。本次只有数据 SHA 和链接验证，没有运行所有重定向任务。

## 2026-09-26 接触奖励代码已成功推送（覆盖下方待认证状态）

用户配置SSH密钥并要求继续push。origin现为git@github.com:SeeXB/holosoma.git。GitHub首次连接所需的ED25519主机公钥已与官方文档指纹核对并加入known_hosts，正常推送成功（未force）。远端main经git ls-remote复核为282f4fbfc4f4f48afe72722ab6f532a2b279f13b，与本地HEAD一致。提交包含6个接触奖励相关代码文件；其他本地修改和实验产物仍未提交。本次操作未重启训练。

## 2026-09-26 接触奖励代码已本地提交，推送待 GitHub 认证

用户要求git提交到远端。本地main提交282f4fb（feat: add reference contact reward for semantic adaptive training），包含6个文件：接触点提取脚本、训练监督器、reward注册和preset、新reward模块、测试。其他已有环境/恢复/文档改动与实验产物未加入该提交。最近8次提交统一作者Runze Zhang；本机无git identity，故本次通过git -c临时沿用该身份，没有修改全局配置。

当前真实远端origin=https://github.com/SeeXB/holosoma.git，branch main（本节优先于历史seexb/origin说明）。推送前ls-remote确认远端main=ae11bf356bbe4f6093ba42348ba3215039333c33。git push origin HEAD:main失败：could not read Username。无HTTPS credential helper/标准凭据文件/GH_TOKEN/GITHUB_TOKEN/gh登录，无SSH私钥和agent；SSH严格主机检查也未通过。尚未推送成功，需要用户在本机配置GitHub认证后继续普通push，不要force push。本次git操作未重启或停止训练。

## 2026-09-26 G3 新增接触位置奖励，重新训练 smallbox

用户要求仅为 semantic adaptive 组加入重定向接触位置奖励并重训，仍GPU0。新实验 `exp/training/sub7_smallbox_023/contact_surface_v1_20260926/`，配置production.json，groups=[3]，从头训练，4096env/seed42/30000iterations/checkpoint500。原G1/G2与旧G3结果保留。supervisor PID618493；训练PID与实时阶段见production/group3/state.json；W&B ID `523fda01`，entity/project yumou0319-/WholeBodyTracking。当前仅启动，不能写成训练完成；不要重复启动。

只新增reference_contact_position（weight2,sigma0.05m），原11奖励/物理/观测/终止/采样不变。接触artifact来源为同一B4参考橡胶手与箱子凸网格最近点（距离≤0.03m激活），不是人工human-contact标签；左右均frame68–238@50Hz。保留目标在object局部坐标，并映射到实际Isaac半球手凸包表面点，训练中用实际wrist和实际object位姿计算两点距离奖励。网格本身未统一或更改；实际半球手在原参考姿态下到目标平均6.02cm、最大10.94cm，所以奖励会鼓励手臂补偿几何差异。不是力奖励，不保证已学会抓取。详细公式、来源和限制见EXPERIMENT.md及inputs/reference_contacts.json。原semantic仍right_ankle事件，手接触掩码独立从轨迹几何提取。

接触artifact SHA fc5845c942a24faaf4e0b3f86a647555c651be98a94db6d9bf5b031fa8c8e0b7，B4/semantic SHA与旧实验相同。训练数据源在src/holosoma/holosoma/data/motions/tasks/sub7_smallbox_023/contact_position_v1_20260926/，实验inputs为冻结副本。新增模块managers/reward/terms/reference_contact.py，离线脚本scripts/build_reference_contact_targets.py，可选reward preset注册，监督器支持groups与contact_file；默认旧三组不变。source_snapshot与PREPARATION.json记录代码指纹。

25项CPU测试通过；32env×2轮短训有非零奖励、320回合评估和视频通过；首次eval出现历史tyro解析偶发错误并卡在报错输出，终止单独失败eval后原样重试成功，未修改依赖/评估协议，失败记录smoke/group3/first_eval_failure.json。正式训练结束后自动执行同协议320回合、视频和W&B指标上传。新正奖励预算接触时段7→9，属于reward ablation，不能当作仅采样对比或直接比较总reward。

启动核验：训练PID618508，日志已到iteration 22/30000，首个checkpoint确认contact reward weight2/sigma0.05、4096env/seed42/30000且未加载旧checkpoint。W&B 523fda01为running，服务端已有contact奖励和位置误差history；进程CUDA_VISIBLE_DEVICES=0，GPU1空闲。证据launch_verification.json。此为启动快照，后续查询live状态。

## 2026-09-26 smallbox来源与接触诊断（用户要求先核对来源再查碰撞）

当前smallbox.obj由tools/generate_omomo_urdf_assets.py从OMOMO captured_objects/smallbox_cleaned_simplified.obj保留拓扑、顶点均值居中、按OMOMO obj_scale中位0.31268005073070526缩放导出，不是InterMimic后处理网格，也不是从零建模。资产清单demo_data/models/omomo_asset_manifest.json记录源路径（旧机器/mnt/sdadrive/shixiongbo/omomo，本机不存在，未做源文件逐顶点验证）；当前35286顶点70712面、尺寸0.38385447×0.43467248×0.23465387m吻合清单。URDF的0.1kg、近似惯量和摩擦为脚本设置，非OMOMO实测值。

GPU0局部诊断（G3最终模型，32env×2回合，64次）：PhysX读回物体/手static,dynamic friction均1.0、restitution0；物体与半球手碰撞均开启、convexHull。基线0/64平均134.6875帧；仅箱子摩擦2.0为0/64平均120.375；仅箱子mass0.1/COM0/标称惯量为0/64平均146.40625，全部最远148。没有修改原训练配置或模型，也未上传。

独立手-箱过滤接触视图复现基线。env0首回合法向接触>1N：左20帧，右1帧，同时1帧；frame100以后到失败双手手箱接触力均0，实际box原点z约0.063m而参考最终1.024m。全部64回合frame>=100共2736个非终止采样点，无双手同时>1N的采样点。记录为每控制步末物理子步的法向力，非连续全部子步或总摩擦力。

重要建模问题：训练main_mesh_collision_halfspherehand.urdf用固定半球手；retarget及MuJoCo视频scene用left/right_rubber_hand.obj。左腕局部x方向网格包围盒，实际半球最远0.09100m，retarget/render橡胶手最远0.17333m，相差8.23cm。视觉接触不能代表实际碰撞体的夹持；这是需要优先统一的几何不一致，但尚未通过统一模型A/B证明是唯一根因。用户目前只要求追溯/检查，尚未改资产或重训。

完整报告与证据：exp/eval/sub7_smallbox_023/anatomy_fulltimeline_ae11bf3_20260922/contact_diagnostic_20260926/REPORT.md、summary.json、contact_and_height.png、各variant的physics_audit.json/rollout/hand_net_forces.npz，pair_baseline下contact_summary.json。诊断脚本均留在同目录，运行已完成。

## 2026-09-26 sub7_smallbox_023 全部训练、最终评估和上传完成

三组均已完成30000轮（最后model_29999.pt）。G1/G2于9月24日17:11–17:14完成训练/评估/视频/上传；G3训练于19:06正常完成，但最初最终评估在tyro配置解析时报AttributeError并exit -11。9月26日用户询问30k改善，已核对代码/输入SHA不变，仅从G3最终checkpoint重试评估，成功完成320回合、视频、上传，没有重新训练或改协议。原失败状态保留production/group3/failed_eval_state_20260924.json，重试记录finish_group3_eval*。三组state现均complete，training/evaluation/video/upload_complete全部True。

30k对比21k：三组完整成功率仍均0/320。G1/G2平均到达19.8625帧、最远21，与21k相同；G3平均130.609375→135.390625（+4.78125帧，约3.7%，+0.095625s），中位147和最远148均不变，目标终点238步。G3 30k有289/320次bad_object_pos、31次bad_object_ori；21k为278/42。持续跟踪有小幅改善，完整任务成功率没有改善，不应宣称已学会任务。

最终产物在exp/eval/sub7_smallbox_023/anatomy_fulltimeline_ae11bf3_20260922/：final_results.json、FINAL_EVAL_REPORT.md、comparison_21000_vs_30000.json和各groupN_29999/actual_vs_reference.mp4。最终W&B G1=4ne1yh0b，G2=vqysa2a5，G3=t8ndd1a7，三组汇总=1c874svs（yumou0319-/WholeBodyTracking）。上传的是原先授权的最终实验评估；额外21k中期报告仍保持本地，未绕过之前自动审批拒绝。

## 2026-09-24 sub7_smallbox_023 第21000轮中期评估已完成

用户询问实际效果，已在GPU0不停止训练的情况下顺序执行三组同为21000轮checkpoint的标准评估（32env×前10完成回合，frame0，噪声0，PaperDR pushes开启，阈值1m/45°，horizon4.76s）。三组均0/320完整成功；G1/G2平均19.8625帧、最远21帧，全部bad_object_ori；G3平均130.609375帧、中位147、最远148，278次bad_object_pos、42次bad_object_ori。G3明显延长跟踪但尚不能完成238步整段。结果不能与训练区间累计成功率混淆，也不是30000轮最终评估。

产物：`exp/eval/sub7_smallbox_023/anatomy_fulltimeline_ae11bf3_20260922/interim_21000_20260924/`，results.json / REPORT.md / groupN_21000的rollout和视频。此次未修改算法、轨迹、终止条件或暂停训练。中期汇总W&B上传被自动审批拒绝（认为缺少该具体中期数据上传授权），未执行上传，详见upload_status.json；本地三组视频已完成。离线视频使用本目录tools/ffmpeg（指向hssim已有imageio-ffmpeg二进制），普通PATH找不到ffmpeg；未来最终视频若失败可使用此路径，不需要重新训练/评估。

## 2026-09-22 09:38 新版 semantic adaptive：sub7_smallbox_023 三组重新启动

用户在更新代码后明确要求再次启动三组，继续仅GPU0，Miniconda hssim，W&B yumou0319-/WholeBodyTracking。本次三组从头训练，旧实验保持暂停。新目录 `exp/training/sub7_smallbox_023/anatomy_fulltimeline_ae11bf3_20260922/`，配置production.json，supervisor PID361645，启动launch_production.sh；不要重复启动。三组4096env、seed42、30000iterations、checkpoint每500轮、原reward/PaperDR，完成后沿用既定320回合评估/视频/指标上传流程。

重要更新：不只是采样代码，B4和配套semantic也已更新。按TRAINING_DATA_MANIFEST_latest12.json验证并复制到新目录inputs冻结：Original SHA596eab80…（未变），B4 SHAa8f68171…，semantic SHA4c4e72ed…。新版semantic为initial_contact，right_ankle，window61–107@30Hz，与新B4 provenance一致，不使用旧pick_up_and_raise_smallbox计划。新采样代码ae11bf3未作本地修改；分区[0,102)、[102,178)、[178,238)@50Hz，ordinary/critical/ordinary，critical weight2，10%global uniform，失败EMA改变三个区间概率。旧“仅一个区间概率恒1”的描述不适用于本次。

15项采样/恢复测试通过；三组32env×2iterations短跑均完成训练→320回合评估→视频。G2首次IsaacSim调试标记材质初始化失败，单独重试通过；保留失败记录。新监督器增加可选worker_start_interval_s，本次25秒。09:40 正式核验通过：G1/G2/G3 PID361666/362108/362485，W&B e8c3d60a/9361c581/24c3f816，实际迭代19/10/4；W&B均running且服务端已有history。三个进程均CUDA_VISIBLE_DEVICES=0，GPU1空闲，系统约60GiB可用内存，swap未使用。 核验见launch_verification.json和wandb_launch_verification.json；进展以production/group*/state.json为准。旧实验B4源文件已被传入新版覆盖，若以后需要续训旧实验，须找回旧SHA匹配数据。

## 2026-09-22T09:21:48.200655+08:00 用户要求停止：sub7_smallbox_023 三组已停止（优先于下方启动状态）

用户明确“停止训练”。GPU 0 小箱子 G1/G2/G3 训练进程 346464/344800/344801 已通过 SIGINT 停止；两个监督器 344786/346453 已停止，未使用 SIGKILL。三组 state.json 已标记 paused_by_user，不得自动恢复或启动评估。保留所有 checkpoint、日志及 W&B 历史；停止快照和最新 checkpoint 路径见 `exp/training/sub7_smallbox_023/activepair_cap10_tol10mm_20260922/user_stop.json`。只有用户重新要求继续时再恢复。

## 2026-09-22 06:30 sub7_smallbox_023 三组已启动（覆盖下方历史待确认状态）

用户明确纠正：“我没说跳过啊，你直接启动”。当前任务为 sub7_smallbox_023 三组、仅 GPU 0；不要再询问是否跳过单事件任务。Original 已传入，和新版 B4 都是 239 samples / 50Hz，object 轨迹完全相同。使用 Miniconda hssim，原 reward / Paper-DR，seed42，4096 env，30000 iterations，每 500 轮保存 checkpoint。

正式根目录 `exp/training/sub7_smallbox_023/activepair_cap10_tol10mm_20260922/`，配置 production.json，启动 launch_production.sh。supervisor PID 344786；G1/G2/G3 当前 PID 346464/344800/344801，W&B IDs `31b2d4b6` / `03e5a002` / `44a336d4`，entity/project `yumou0319-/WholeBodyTracking`。G1 首次 run 06d5026e 在 IsaacLab 资产加载事件循环异常中退出（未完成任何 PPO iteration），错开初始化后重试；retry supervisor 346453，入口 launch_group1_startup_retry.sh，保留失败记录。三组最终汇总由 G1 retry supervisor 负责，原 supervisor 保留最初失败 Future。独立 nohup 会话；不要重复启动。实时状态以 production/group*/state.json 和 train.log 为准。

G1=Original + original_adaptive；G2=新版 B4 + original_adaptive；G3=同一 B4 + semantic_adaptive 单窗口变体。G3 例外已明确记录：原语义只有 pick_up_and_raise_smallbox，start/trigger61,end104,fps30；独立 RL plan 显式 single_event_sampling=window，映射 [102,173)@50Hz，保留10%全局均匀混合，区间概率恒为1，没有跨区间自适应。原 semantic_plan.json 未改，没有伪造第二个事件。W&B tags/notes 和 protocol_exception.json 同步标记。B4 active-pair cap10 的 10mm 是 refinement 接受容差，不是 solver foot/penetration 容差。

20 项采样/恢复测试通过；三组 32env × 2iterations 短跑均已完成训练、320回合评测和离线视频，state=complete。正式训练正常结束后，监督器将分别执行既定评测、离线视频和 W&B 指标上传。详见 EXPERIMENT.md、input_sha256.json 与 launch_verification.json（生成后）。06:33 启动核验：G1/G2/G3 已到 3/25/24 轮；三组 W&B 服务端均有训练 history。GPU0 约19.3GiB，GPU1空闲，系统约60GiB可用内存，swap未用。本节为启动快照，不能理解为3万轮已完成。

## 2026-09-22 06:24 OmniRetarget 已补传，待明确任务/分组

sub7_smallbox_023 Original 现已在 motions/tasks/sub7_smallbox_023/omniretarget_original_default1mm/，与新版 B4 都为 239 samples/50Hz，hash/有限值/四元数检查通过，object position/quaternion 完全一致，共用 scene hash。缺 Original 的问题已解决。production.pending.json 已准备，尚未开训。用户前一条说“这两个确实只有1个事件哈哈，先不管了”，助手回复先跳过两个单事件任务；最新又说“开始训练吧那，omniretarget也传了”。已提出一个合并澄清：小箱子先跑G1/G2、小箱子三组且G3单区间标记，还是换其他任务。小箱子原 semantic 仍只有trigger61/end104，不能直接跑现有G3；等待用户明确，勿擅自伪造第二个事件或挑选另一任务。仅GPU0约束继续有效，两卡当前无训练进程。


## 2026-09-22 sub7_smallbox_023 新版数据训练准备（最新请求）

用户从另一服务器更新了重定向代码/数据，要求仅使用 GPU 0 开始 sub7_smallbox_023 的原三组训练。新版 B4 在 `src/holosoma/holosoma/data/motions/tasks/sub7_smallbox_023/semantic_b4_active_pair_cap10_tol10mm/`，239 samples / 50 Hz。B4、官方 PT、场景、semantic/event-plan 均与来源清单哈希匹配；本机化 scene 通过 37 个 mesh 路径检查和 MuJoCo 加载。

尚未启动训练，待用户回答两项：没有本任务 Original 轨迹，已问本机生成还是提供另一服务器已有文件；新版 semantic_plan 仅一个事件（trigger/start=61, end=104, fps30），现有 semantic_adaptive 至少需要两个 trigger，已问提供多事件计划还是使用明确标记的单区间变体（没有跨区间自适应分配）。不要直接复制 B4 冒充 Original 或伪造语义事件。详情及冻结输入见 `exp/training/sub7_smallbox_023/activepair_cap10_tol10mm_20260922/PREPARATION.md` 与 PREPARATION.json。此前两任务进程当前均不在运行，GPU0/1检查为空闲；本次不要在GPU1启动新任务。


## 2026-09-20 01:38 内存升级后六组已重新启动（优先于下方历史暂停/驱动阻塞状态）

用户已解决 Secure Boot 驱动问题，并再次明确要求继续训练。本机 93 GiB RAM；两张 RTX 5090 均通过 hssim 的 CUDA 实际矩阵运算检查。六组已通过 checkpoint/输入 SHA 校验并以独立 nohup 会话启动，Miniconda 环境不变。恢复计划、进程/W&B 对照及验证在 `exp/training/memory_upgrade_resume_20260920/`，详见 RESUME.md；不要再运行旧 launch_production.sh 或重复启动。

桌子 GPU 0：G1/G2/G3 从 8001/4001/8001 轮恢复，W&B `55d38425` / `cc392dfd` / `4a430daf`，训练 PID 6069/6068/6066，supervisor 6002/6003/6004。显示器 GPU 1：三组从 4001 恢复，W&B `1701fd3c` / `58d4a977` / `ae040251`，训练 PID 6071/6067/6070，supervisor 6005/6007/6010。显示器仍标记 foot5mm、penetration1mm。均保持4096 env、seed42、原 reward/Paper-DR，最终model_29999.pt；checkpoint每500轮保存。恢复模型/优化器/归一化；旧checkpoint没有仿真/RNG/采样器现场，因此这部分重新初始化。训练正常完成后各组独立评测并上传，每个任务仅G1 supervisor负责最后三组汇总。

以上是启动快照，实际进展以 state.json、最新日志和验证记录为准。暂不需要之前提出的额外swap扩容。


## 2026-09-20 内存升级后恢复已授权，等待驱动证书注册（最高优先级）

用户已装好内存并明确要求继续六组训练。本机识别 93 GiB RAM、约 89 GiB 可用；旧训练/监督进程均已退出。但新启动内核 6.8.0-138-generic 下 NVIDIA 驱动未加载；两张 5090 可由 lspci 识别，DKMS 590.48.01 已安装。Secure Boot enabled，用户执行 `sudo modprobe nvidia` 明确报 `Key was rejected by service`。

已核对 `/var/lib/shim-signed/mok/MOK.der`：CN=shihang-MS-7E06 Secure Boot Module Signature key，serial=204E2EB13B6844001AB384072EADB88BB7C83FA6，与 `modinfo -F sig_key nvidia` 一致，但 mokutil 显示未 enrolled。需要用户本机执行 `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` 设置临时注册密码，再重启进入 MOK Manager 完成 Enroll MOK。无需更换 CUDA/conda 环境。暂未启动任何训练，不能声称恢复成功。GPU恢复后已有用户授权可继续，不必重复确认启动许可。

六组恢复配置已生成：`exp/training/memory_upgrade_resume_20260920/plan.json` 及每组 `.sh`。桌子 G1/G2/G3 从 8001/4001/8001 继续；显示器均从 4001 继续，仍标 foot5mm/penetration1mm。目标最后一轮 29999，每 500 轮保存，恢复模型/优化器/归一化，仿真/RNG/采样器重新初始化。`resume_three_group_worker.py` 已增加显式 `--allow-paused`、恢复原因和 `--skip-finalization`（各任务仅 G1 汇总，避免三组争抢最终评估）；保留 W&B 原有协议备注。3 项 CPU 单元测试通过，六个 checkpoint 内容/哈希/轨迹匹配均已验证，见 checkpoint_validation.json。一次 monitor G2 预检进程出现 SIGSEGV，重试预检和独立六 checkpoint 校验通过；原失败日志保留，不据此声称硬件已做稳定性测试。仍需驱动恢复后验证 CUDA、实际启动进展与 W&B 上传。


## 2026-09-19T23:38:50.943195+08:00 用户要求全部暂停，加装内存（最高优先级）

用户明确要求先停掉所有本次训练，准备加装内存。两个任务的全部训练及监督进程均已停止；不应自动重启，不需要继续执行先前待确认的 swap 扩容。5 个尚存训练进程通过 SIGINT 退出，没有强制 SIGKILL；桌子 Group 2 的第二次恢复在暂停前已再次失败（详情见暂停快照）。六组 state 均标为 `paused_by_user`，原状态保存在 user_pause 和 `exp/training/two_task_launch_20260919/pause_for_ram_upgrade.json`。现有 checkpoint/log 全部保留，最近日志进度不等于 checkpoint 进度。加内存并得到用户恢复指令后，先核对实际内存和 CUDA，再依据快照恢复；目前的 resume_three_group_worker.py 只接受 failed 状态，恢复前需显式适配 paused_by_user，不要直接重复旧启动脚本。

| Task | Group | Last logged iteration | Latest saved checkpoint |
|---|---:|---:|---|
| sub16_largetable_013 | 1 | 11312 | model_08000.pt |
| sub16_largetable_013 | 2 | 4302 | model_04000.pt |
| sub16_largetable_013 | 3 | 10592 | model_08000.pt |
| sub9_monitor_025 | 1 | 5260 | model_04000.pt |
| sub9_monitor_025 | 2 | 5227 | model_04000.pt |
| sub9_monitor_025 | 3 | 4878 | model_04000.pt |


## 2026-09-19 22:06 桌子 Group 2 崩溃恢复（优先于下方启动快照）

`sub16_largetable_013` Group 2 / 原 W&B `50bdbeb3` 于 14:00:12 被 SIGKILL，最后日志 5448；会话 cgroup `oom_kill=1`，支持主机 OOM 原因。用户要求修复并重启。已从 `model_04000.pt` 恢复，后续从 4001 到 29999；模型/优化器/归一化状态恢复，仿真/RNG/采样器现场原 checkpoint 未保存，因此重新初始化。当前新 W&B `4457898a`，保留旧 run 历史，避免步数回退导致 W&B 丢弃新指标。只恢复这组，其他五组未重启。

恢复入口 `scripts/resume_three_group_worker.py`，启动脚本 `exp/training/sub16_largetable_013/officialpt_default1mm_20260919/launch_recovery_group2.sh`。恢复 supervisor PID 2562080，进程详情读 `production/group2/state.json`。分配器设 `MALLOC_ARENA_MAX=2` / `MALLOC_TRIM_THRESHOLD_=131072`，checkpoint 每 500 轮保存；训练配置/1 mm 条件不变。PPO.load 修复为 checkpoint iter+1，3 项 CPU 回归测试通过。自动评估仍使用最终 model_29999.pt；恢复 supervisor 会在其他组结束后补齐汇总。详细证据及限制见 `exp/training/sub16_largetable_013/officialpt_default1mm_20260919/GROUP2_RECOVERY.md`。这些内存设置是缓解措施，不能保证其他用户作业占用增加时仍不发生 OOM。

恢复补充：第一次恢复 `420a19fb` 在 4006 再次 SIGKILL。当前第二次恢复启用 `HOLOSOMA_CPU_MALLOC_TRIM_INTERVAL=25` 主动归还空闲 CPU 页，supervisor 改为子进程预检，常驻 RSS 从约 400 MiB 降到 24 MiB。六组同时运行仍仅约 1.3 GiB 可用内存、2 GiB swap 全满；已准备 `scripts/add_training_swap.sh` 增加独立 16 GiB swap，当前账号 sudo 需密码，已请用户在终端执行。该操作在 swapon 验证前仍为待完成，不能声称内存短缺已根治。诊断只读日志最后 32 KiB，避免全量读入大日志占用内存。

## 2026-09-19 本机两任务启动（优先于历史机器状态）

用户确认当前 `TASK_OBJECTS` 的 20 条 OMOMO 名单正确，暂不做 LAFAN。要求 GPU 0/1 各跑一个任务的三组训练，W&B entity 为 `yumou0319-`。本机仓库 `/home/shixiongbo/holosoma`，Miniconda 环境为 `~/miniconda3/envs/hssim` 和 `hsretargeting`。

- **GPU 0：`sub16_largetable_013` 已正式启动。** Original/B4 使用官方 PT、原始单凸包模型、foot/penetration 1 mm、step 0.2，完整完成 188 帧；RL 为 312 samples / 50 Hz，horizon 6.22 s。三组均 seed 42、4096 env、30000 PPO iterations、原始 reward / Paper-DR。短跑已完成三组训练→各 320 回合评测→离线视频。正式目录 `exp/training/sub16_largetable_013/officialpt_default1mm_20260919/`，supervisor PID 852675，训练 PID 853097/853100/853098，W&B IDs `8debfc4b` / `50bdbeb3` / `33dbce7d`。服务端已验证有训练 history。每组训练正常完成后自动评测并上传指标。
- **GPU 1：`sub9_monitor_025` 已获用户批准使用 5 mm，三组正式训练已启动。** 1 mm 原始模型下 Original 完成、B4 在零基 frame 64 不可行；32 块显示器碰撞下两方法分别在 frame 50/52 不可行。2 mm 脚固定容差下 B4 完成、Original 仍失败；5 mm 下两种方法均完成 194 帧，防穿透仍为 1 mm，其余约束/预算不变。5 mm 候选参考、配置在 `exp/training/sub9_monitor_025/officialpt_foot5mm_20260919/`，三组短跑均已完成训练→各 320 回合评测→视频；用户已明确批准“改成5mm，mark这个任务，然后开始训练”。这是已标记的协议例外：Original/B4 同时使用 foot 5 mm，penetration 仍为 1 mm，不能标成 default1mm。标记保存在 `protocol_exception.json`、EXPERIMENT.md 和 W&B tags/notes。Supervisor PID 1146024；训练 PID 1146468/1146469/1146470；W&B IDs `44d05b83` / `f9c4da2e` / `9eed1869`。此处是启动快照，运行状态以 live 文件为准。
- 大桌子本任务试过上一任务的 32 块资产，但两种方法都在 frame 36 不可行，因此正式使用本任务成功的原始模型。失败尝试完整保留，未以不完整轨迹开训。
- 两个任务的 semantic schema、动态重新执行结果、时间窗和 RL trigger 映射均通过。原资产/场景含旧机器路径，新实验另存本机路径；为 largetable/monitor 缺失的兼容 OBJ 创建了指向 `demo_data/models` 的链接。
- 启动不依赖终端存活：`nohup` + 独立会话运行；无重启自启动。监控 `production/groupN/state.json` 与 `train.log`，不要重复启动。所有当前状态以 live 文件和进程为准。
- 本机没有 OSMesa，离线视频使用 `MUJOCO_GL=egl`；训练不启用实时相机。W&B 同时设置 `WANDB_USERNAME` 与实际 logger 使用的 `WANDB_ENTITY` 为 `yumou0319-`。
- 汇总及准备脚本：`exp/training/two_task_launch_20260919/`。本节是启动快照，不意味着已经完成 30000 轮训练。

## 2026-09-19 数据目录已整理（路径信息优先于下方历史记录）

用户要求：重定向数据放 `src/holosoma_retargeting/holosoma_retargeting/demo_data/`，训练专用数据放 `src/holosoma/holosoma/data/`，共享输入统一放前者，`exp/` 仅作为结果目录。已移动 168 项输入（1138 个实体文件、约 7.29 GiB），没有复制外部数据。完整目录图和使用说明见 [DATA_LAYOUT.md](DATA_LAYOUT.md)。

- 官方 InterMimic PT：`demo_data/omomo/official_inputs/`（这里的 demo_data 指上面的完整根目录）。
- OMOMO bundle/semantic/video：`demo_data/omomo/bundles/<task>/`；旧输入版本及共享碰撞资产：`demo_data/retarget_inputs/<version>/`。
- 重定向模型资产：`demo_data/models/`；原 `models/` 路径保留兼容链接。
- RL motions：`src/holosoma/holosoma/data/motions/tasks/<task>/`，新路径配置模板在 `data/experiments/`。
- 仓库外 OMOMO 通过 `demo_data/external/omomo` 软链接访问，已有外部模型缓存软链接继续保留。
- 原 exp 输入路径是兼容软链接；报告、日志、checkpoint、旧运行配置快照不移动。当前三组 largetable 仍用冻结配置和旧路径别名继续训练，38 个输入 SHA256 未变。不要删除兼容链接或用新模板覆盖正在运行的配置。
- 脚本默认输入/生成路径已更新；semantic 原始回答及渲染中间结果仍写 exp。迁移清单：`exp/data_layout/migration_20260919.json`。
- 这次整理不改变旧数据的有效性；历史错误 OMOMO 输入/结果仍不可用作正式比较。
- 验证：61 项重定向/semantic 回归测试、3 项迁移测试通过；43 个 Python/Shell 文件语法检查通过；12 个 MuJoCo scene 成功加载，URDF mesh 引用和 RL NPZ 可读，迁移目录无失效链接。结果在 `exp/data_layout/validation_20260919.json`。


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
