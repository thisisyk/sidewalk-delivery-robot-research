# 冻结HistGB研究模型 / Frozen research model

**NON-COMMERCIAL USE ONLY. This model as a whole and any downstream use or modification are subject to the Waymo Dataset License Agreement for Non-Commercial Use, including Section 4. Do not use it for vehicle operation or assistance, production systems, or commercial purposes.**

**仅限非商业研究用途。模型整体及下游使用、修改受Waymo非商业数据协议约束；不得用于车辆操作或辅助操作、生产系统、商业用途。** 请阅读随模型提供的[完整协议](../licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt)及[署名声明](NOTICE.txt)；[官方协议](https://waymo.com/open/terms/)第2(c)节规定训练模型的分发条件。本说明不能替代完整协议。

## 文件与来源

- `frozen_histgb_seed42.joblib`：原冻结全特征HistGradientBoosting配置在857例训练样本上的重拟合导出，194列特征，seed42。没有重新选模，没有用开发或评估样本训练。
- [frozen_histgb_seed42.manifest.json](frozen_histgb_seed42.manifest.json)：文件大小及SHA-256、环境版本、特征顺序、拟合配置、来源和一致性核验。
- [NOTICE.txt](NOTICE.txt)：适用协议、下游用途限制和Waymo署名。

原实验没有保存权重文件；历史运行记录中的 `no_weights_saved` 保持原样。本次文件由同一冻结配置重新生成，单独记录导出，不把新文件冒充原运行时保存的二进制。原206例锁定评估的AP为0.7302762063131997；新增导出的实际预测差及保存加载结果应查看本次清单，不覆盖原结果。

模型文件保存拟合参数和评分所需元数据，不附原始TFRecord、逐场景特征、标签或预测表。原始22分片须由每位用户自行取得Waymo注册及许可，并从官方存储桶下载；详见[数据获取](../docs/data_access.md)。

## 直接启动合成Demo

在仓库根目录安装 `requirements.txt` 和本项目后，核对模型来源及清单，再运行：

```text
python scripts/serve_demo.py --model-artifact models/frozen_histgb_seed42.joblib --model-manifest models/frozen_histgb_seed42.manifest.json --trust-model-artifact --port 8766
```

打开 `http://127.0.0.1:8766/`。无需下载原始数据；依赖版本按项目要求，其中scikit-learn为1.7.2。

**仅加载可信来源的文件。** `joblib`反序列化可能执行代码；SHA-256核对只能确定文件与已知清单一致，不能证明未知文件安全。`--trust-model-artifact`显式确认加载这一可信文件，不解除许可限制。

原训练域为道路汽车，人行道机器人属于未经验证的域外迁移；模型分数不是事故概率或真实接管概率。避障、步道约束、求助与模拟协助由独立规则处理。此模型只供非商业离线研究与本机合成演示，不是车辆控制或真实配送部署模型。完整研究限制见[报告](../docs/final_report_zh.md)与[Demo指南](../docs/demo_guide_zh.md)。
