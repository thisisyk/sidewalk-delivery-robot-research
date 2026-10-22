# 数据获取、溯源与发布范围

当前状态：**已从官方存储桶下载并核查22个 WOMD v1.3.1 Scenario分片，共8,349,457,941字节，完成真实数据实验。** 用户完成许可注册并明确批准CLI认证；下载后已远程注销并移除本地凭据，没有创建云项目或启用计费。原始数据仅本地保留，公开包不含账号与认证记录。

## 官方入口

| 用途 | 官方来源 |
| --- | --- |
| 数据格式、历史及未来边界 | [WOMD Motion 说明](https://waymo.com/intl/en-ca/open/data/motion/) |
| 数据版本与下载入口 | [Waymo Open Dataset Download](https://waymo.com/intl/en-ca/open/download/) |
| 注册、使用与分发条件 | [Waymo Open Dataset Terms](https://waymo.com/open/terms/) |
| Scenario 字段定义 | [官方 scenario.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto) |
| 数据集引用和工具 | [官方代码仓库](https://github.com/waymo-research/waymo-open-dataset) |
| 与 Uber Eats 的合作关系 | [Uber 官方公告，2024-04-03](https://www.uber.com/us/en/newsroom/uber-and-waymo-autonomous-deliveries-phoenix/) |

本项目选择不含原始相机和激光雷达的 Scenario 轨迹数据，以降低 CPU 实验的下载与处理负担。使用有未来真值的 training 和 validation；官方 test 不能用于本项目自定义标签评价。下载页面所列新版本不代表本项目已经使用该版本，实际版本以清单为准。

本次固定使用 `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario` 下的训练分片00000—00011，以及验证分片00000—00009；下载后重新核对全部文件大小与SHA-256。固定划分、保留量与删失比例见[最终报告](final_report_zh.md)，核查摘要见[验证记录](validation.md)。

## 获取步骤

1. 用户在官方入口登录，阅读并接受适用的数据条款，确认账户获得访问权。
2. 从官方 Cloud Storage 页面选择同一版本的 Scenario training 与 validation 数据；先获取少量 shard 检查解析和有效标签数量。
3. 将原始 TFRecord 放入本地数据目录。目录必须被 Git 忽略，云访问凭据不写入任何配置或日志。
4. 记录下载来源、对象路径、版本、文件名、大小、下载时间与 SHA-256。区分完整 shard 下载与手动提供的本地文件。
5. 预处理后核对场景数量、ID 重复、时间戳、对象有效性、标签正例率和删失计数。真实训练仅在这些审计完成后运行。

无需一次下载整个数据集。样本上限与停止下载的依据必须预先配置，不能因锁定结果不好而选择不同 shard。小样本不足时可以扩展样本，但需登记为后续实验并重新保持独立评估。

## 许可与 GitHub 内容

Waymo 数据协议与官方代码仓库的 Apache 许可证是不同文件。数据和训练产物不能因为项目源代码有一个开源许可，就自动获得相同分发权限。[官方代码仓库](https://github.com/waymo-research/waymo-open-dataset)

截至本次核实，官方数据页面提供的是 2025 年 3 月非商业使用协议；其中另列数据再分发、训练模型发布、署名和用途限制。使用与发布前以实际适用的[官方协议](https://waymo.com/open/terms/)为准。本文件不是完整协议的替代文本。

本项目拟发布内容限于独立编写的源代码、配置、方法文档、聚合评估表及合规的说明图。默认排除以下文件：

- 原始 TFRecord、复制的数据集、地图和轨迹批量导出。
- 逐场景特征、逐场景标签、可重建原数据的缓存或打包子集。
- Google Cloud 凭据、账户标识、访问令牌、用户下载记录。
- 训练权重与模型文件；如后续确需分享，先按官方条款处理适用协议、完整协议副本与署名。

研究插图也应核实其范围和适用条款；不默认向公共仓库上传大量真实场景图。聚合结果只应包括研究所需统计，不携带可还原源轨迹的记录。

使用数据后的最终报告和发布包应包含官方要求的署名及数据集引用。数据访问失败时保留明确的状态文件；不将合成软件测试记为真实数据分析。

## 用户最终批准

拟上传内容清单、真实结果、结论和限制需先展示给用户。数据获取授权与最终 GitHub 发布批准是不同步骤；取得前者不能代替后者。本项目用户已于2026-10-23明确授权公开发布，授权范围与实际发布状态见[publication.json](../publication.json)；原始数据、缓存、逐场景标签与预测、账号及权重仍不公开。
