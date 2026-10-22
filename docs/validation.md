# 验证记录与可复现核查

## 模型权重补充发布验证

本次在原82项基础上加入权重完整性、可信加载、特征顺序、许可副本、模型文件直接推理及官方下载校验等检查，累计96项软件测试通过，完整本地运行37.785秒。新增检查不增加真实研究的独立评估样本。

冻结HistGB按857例训练样本重新拟合后导出；206例原评估预测与历史私有预测最大差为1.1102230246251565×10⁻¹⁶，保存加载往返预测差为0。新权重为236,606字节，SHA-256及环境见[模型清单](../models/frozen_histgb_seed42.manifest.json)，完整核对见[模型一致性记录](../models/model_parity.json)。同时用该模型文件在不读取原始或预处理数据的条件下启动本机服务，并完成一组未缓存参数的合成路口API调用；这只验证模型接入。

本地22个原始分片的大小和SHA-256与[官方下载清单](../configs/womd_download_manifest.json)全部一致，共8,349,457,941字节。原始数据没有上传；公开模型随附[协议副本](../licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt)及[署名和用途声明](../models/NOTICE.txt)。下文82项及更早记录描述其对应历史阶段。

<!-- request-fallback:start -->
## 独立行为求助补充

已加入不依赖模型评分的停滞、互堵和多次规划失败求助，并将协助处理改为显式指令模块。原模型高分通道保留；分数无效不再自动阻断行为请求。固定八场景、开/关遮挡共16个配对设置中，完成任务由70/96变为74/96，请求总数由0变为38，矩形接触机器人帧由76变为80。这些是固定合成条件下两项改动的组合机制比较，不能全部归因于求助触发；新增通道不是机器学习训练结果；未完成、接触和规划失败全部保留。

完整触发假设、逐场景对照、原因计数和残留限制见[人行道报告](sidewalk_demo_zh.md)与[配对比较](../results/sidewalk/fallback_comparison.json)。原真实研究数据与模型选择不变；用户已于2026-10-23授权公开本项目，发布记录见[publication.json](../publication.json)。
<!-- request-fallback:end -->

当前人行道机器人主版本累计**82项软件测试通过**，完整运行40.341秒，覆盖原实验、模型推理、观测隔离、运动/通行域，以及新增行为求助、协助指令、可见静止机器人规划占地与执行反馈等软件行为。测试不等于82次独立实证实验，也没有验证机器人域预测精度或真实安全。早期人行道阶段累计68项通过是历史记录。[当前人行道说明](sidewalk_demo_zh.md) · [动画](../demo/sidewalk_scenarios.gif) · [分区总览](../demo/sidewalk_atlas.png)

当前主数据源为 `demo/sidewalk_gallery.js`。机器人和行人仅在人行道/斑马线，所有汽车含停放车仅在机动车道；独立审计按旋转矩形完整占地与导出的互不重叠矩形域检查，而非只检查中心点。行人运动由相邻真实保存帧中同ID对象的位移核对，不能用图标摆腿替代位移证据。具体全场审计与合成结果以 `results/sidewalk/` 和人行道报告为准；失败、接触与越界字段不因表现差而删除。

界面检查确认正常设置下自动播放，减少动态偏好下显示明确开始按钮；位置与航向可在实际帧间平滑插值，分数保持原帧输出。缺少新的人行道数据会报错，不回退 `example_run.js` 或旧城市道路画廊。

历史追加验证阶段累计45项测试；城市道路阶段累计57项。这些旧场景作为历史记录保留，不能替代当前人行道版。原模型206个评估分数重现的最大差1.11×10⁻¹⁶，见[一致性审计](../results/simulation/model_parity.json)。追加分析与排队模拟有独立聚合核对和输入/脚本指纹，它们都不增加独立真实评估样本。

## 本次实际验证

- 原真实研究阶段37项软件测试全部通过，覆盖几何标量/批量一致性、TFRecord双CRC与未知protobuf字段、未来信息隔离、删失规则、预算同分处理、数据来源/缓存哈希保护和小型人工样例的完整流程。人工样例只用于测试软件。
- 对22个真实原始文件重新读取SHA-256与文件大小，核对来源、缓存、协议、提取代码指纹与划分，见[来源审计快照](../results/real/source_audit.json)和[去除本机路径的来源清单](../results/real/source_manifest.json)。该较早快照记录案例文档尚不存在；后续数值审计快照已确认文档存在。
- 使用独立AP/AUC/预算/阈值公式复算84条锁定评估记录，并重算开发集选模与300次配对分片自举，0项不一致。详情见[数值审计](../results/real/numeric_audit.json)。
- 按固定规则抽取首个训练分片的前20个完整场景进行标量几何重算，共7,068次矩形比较；距离、标签、资格及首次事件时间均一致。该分片496条记录均通过双CRC。详情见[标签审计](../results/real/label_audit.json)。
- 三个脚本确定的正确、误报、漏报案例完成助手图形及数值核查，见[案例记录](case_review.md)。它不等于人类专家对危险或接管需求的标注。
- Google CLI已远程注销，本地认证目录已移除，见[清理摘要](../results/real/auth_cleanup.json)。

## 复现顺序

先按README自行取得受许可约束的官方数据并提取，使用新的结果目录重跑；仓库不会分发原始数据或私有预测。建模环境中可执行：

```text
python scripts/train_evaluate.py --input artifacts/prepared/samples.csv --output results/reproduction
python scripts/audit_results.py --prepared artifacts/prepared --results results/reproduction --raw-root data/womd --output artifacts/audit_results.json
python -m unittest discover -s tests -v
```

独立核查需要原始提取目录和本地逐场景预测。预测默认保存在与结果同级的 `private/<结果目录名>/`，不能只用公开聚合表重算AP。标量标签核查在预处理环境执行：

```text
python scripts/audit_labels.py --source data/womd/training/training.tfrecord-00000-of-01000 --cache artifacts/prepared/shards/training.tfrecord-00000-of-01000.csv --output artifacts/audit_labels.json
```

本机预处理、建模分别使用Python 3.12.14及两组依赖，版本固定在两份requirements文件和原始运行清单中。实际运行使用CPU；没有使用GPU、外部训练服务或更改Windows应用控制策略。

## 当前人行道回放的复现与核查

保留原训练特征文件的建模环境可重新生成一套独立结果，再核对整个机体的通行域与行人位移；不要覆盖保留的原运行：

```text
python scripts/build_sidewalk_gallery.py --samples artifacts/prepared/samples.csv --output results/sidewalk_reproduction --gallery artifacts/sidewalk_gallery.js
python scripts/audit_sidewalk_gallery.py --gallery artifacts/sidewalk_gallery.js --output results/sidewalk_reproduction/geometry_motion_audit.json
python scripts/audit_assistance_gallery.py --gallery artifacts/sidewalk_gallery.js --output results/sidewalk_reproduction/assistance_audit.json
python scripts/render_sidewalk_animation.py --input artifacts/sidewalk_gallery.js --output artifacts/sidewalk_scenarios.gif
```

画廊生成时调用原已选出的模型，不重新选模。动画渲染只读取保存的坐标、航向、资格、分数与接触字段，不调用模型、不填补分数；输出清单记录画廊SHA-256、共同帧时间及同ID行人的位移统计。几何与运动审计只验证这些合成数据的内部一致性，不证明真实配送行为。

新增求助机制的验证还须检查：模型分数为无效时行为请求仍可进入队列；瞬时等待不会立即滥发请求；请求受次数、冷却与队列约束；请求原因与席位状态一致。显式协助模块须核对指令下发、允许区域内执行、保持后恢复检查以及现场升级仍保留未完成。求助与处理器同时改变的主对照属于组合机制，不能只归因于模型或一个触发器；仅启用行为求助的消融应单独报告。

[协助生命周期审计](../results/sidewalk/assistance_audit.json)采用不导入控制器的独立脚本，核对保存的请求、指令、执行状态与实际运动。当前界面默认从人群场景首帧播放，不自动跳到协助或到达片段，避免只展示成功时刻。

## 记录与审批含义

`report.md`、`summary.json`和`run_manifest.json`保留生成时的自动草稿与待审状态，未为后续写作修改运行证据。后续助手核查状态单独记录在[review_status.json](../results/real/review_status.json)，最终叙述以[中文报告](final_report_zh.md)为准。

数值核对与助手案例查看均已完成；人类专家标签有效性研究未进行。用户已于2026-10-23授权公开本项目，授权与实际发布状态见[publication.json](../publication.json)。原审计和运行产物中的待审字段保留生成时状态。数值一致性不证明代理事件等于危险，也不修复删失选择偏差或未知驾驶日志相关性。
