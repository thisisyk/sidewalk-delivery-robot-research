# 数据获取、溯源与发布范围

当前状态：**已从官方存储桶下载并核查22个 WOMD v1.3.1 Scenario分片，共8,349,457,941字节，完成真实数据实验。** 用户完成许可注册并明确批准CLI认证；下载后已远程注销并移除本地凭据，没有创建云项目或启用计费。原始数据仅本地保留，公开包不含账号与认证记录。

用户随后要求补充原始数据和模型权重。**本次公开冻结配置模型、完整适用协议、署名和官方数据下载校验工具；不向无身份限制的公共仓库复制原始分片。** 这由Waymo协议对数据与训练模型的不同分发条件决定，详见下文。

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

### 本次22分片的复现下载

[下载清单](../configs/womd_download_manifest.json)记录本研究的22个官方 `gs://` 对象、字节数与SHA-256，不含原始数据、签名下载链接或账号凭据。每位复现者须自行在Waymo官网注册并同意协议，再通过Google官方工具完成自己的认证。脚本不替用户接受条款或登录，也不会共享本项目作者的访问权。

在已有合法访问权和Google官方CLI的环境中运行：

```text
python scripts/download_womd.py --destination data/womd --download --registered-with-waymo
```

仅核对已有的本地文件，无需下载：

```text
python scripts/download_womd.py --destination data/womd --verify-only
```

脚本按清单核对文件大小和SHA-256；不能通过检查的文件不能作为同一研究输入。未注册、权限不足或官方对象不可访问时，应按官方渠道处理，不从本仓库寻找数据镜像。

## 许可与 GitHub 内容

Waymo 数据协议与官方代码仓库的 Apache 许可证是不同文件。数据和训练产物不能因为项目源代码有一个开源许可，就自动获得相同分发权限。[官方代码仓库](https://github.com/waymo-research/waymo-open-dataset)

截至本次核实，官方数据页面提供的是 2025 年 3 月非商业使用协议；其中另列数据再分发、训练模型发布、署名和用途限制。使用与发布前以实际适用的[官方协议](https://waymo.com/open/terms/)为准。本文件不是完整协议的替代文本。

官方协议第2(b)节将数据及其复制、修改的分发限定于已在Waymo注册并同意协议的接收者。公共GitHub仓库不能保证每个接收者满足这一条件，因此不上传本地8,349,457,941字节的原始分片，也不以Git LFS或Release附件绕过这一接收者限制。[Waymo官方协议](https://waymo.com/open/terms/)

第2(c)节另允许带条件公开训练模型，包括权重与偏置。本次新增[模型文件与清单](../models/README.md)，随附[完整协议](../licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt)和突出用途限制、下游条款与署名。**模型整体及下游使用、修改仅限协议允许的非商业用途；第4(a)节禁止车辆操作及辅助操作、生产系统和商业用途。** 项目使用合成环境演示，不提供真实车辆部署。[Waymo官方协议](https://waymo.com/open/terms/)

公开内容现包括源代码、配置、方法文档、聚合评估表、合成说明图、合规附带声明的模型和官方数据下载校验清单。继续排除以下文件：

- 原始 TFRecord、复制的数据集、地图和轨迹批量导出。
- 逐场景特征、逐场景标签、可重建原数据的缓存或打包子集。
- Google Cloud 凭据、账户标识、访问令牌、用户下载记录。
- 除明确列入本次模型发布清单外的私有缓存和训练中间产物。

研究插图也应核实其范围和适用条款；不默认向公共仓库上传大量真实场景图。聚合结果只应包括研究所需统计，不携带可还原源轨迹的记录。

使用数据后的最终报告和发布包应包含官方要求的署名及数据集引用。数据访问失败时保留明确的状态文件；不将合成软件测试记为真实数据分析。

模型按原857例训练集、194列特征及HistGB seed42冻结配置重拟合后导出，没有重新选择模型或引入评估样本训练。文件哈希、运行环境与保存加载核验以[模型清单](../models/frozen_histgb_seed42.manifest.json)为准。原运行记录中的 `no_weights_saved` 保留当时状态，本次导出不改写历史实验结果。直接运行合成Demo无需取得原始数据，见[操作指南](demo_guide_zh.md#本地服务)。

## 用户最终批准

拟上传内容清单、真实结果、结论和限制需先展示给用户。数据获取授权与最终 GitHub 发布批准是不同步骤；取得前者不能代替后者。本项目用户已于2026-10-23明确授权公开发布，并随后明确要求补充原始数据和模型权重。授权范围与实际发布状态见[publication.json](../publication.json)。按官方分发条件，本次上传模型及配套声明、官方数据清单与下载工具；原始分片、特征缓存、逐场景真实标签与预测、账号和凭据仍不公开。
