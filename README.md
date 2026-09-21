# 基于改进 U-Net 的视网膜血管分割与血管形态特征分析（DRIVE）

课程实践仓库 · 选题：基于眼底图像的视网膜血管分割与血管形态特征分析

- **基线模型**：U-Net（编码器-解码器 + 跳跃连接）
- **改进模型**：Attention U-Net + Tversky 损失（加大漏检 FN 惩罚，提升细血管召回）
- **附加分析**：血管形态定量分析（血管密度、直径分布、总长度、迂曲度、分形维数）

## 目录结构

```text
retinal-vessel-segmentation/
├── data/
│   └── README.md          # 数据集下载与放置说明（数据本身不入库）
├── src/
│   ├── preprocess.py      # 预处理：绿色通道 + CLAHE，统一导出 PNG
│   ├── dataset.py         # PyTorch Dataset：归一化 / 翻转旋转增强 / padding
│   ├── models.py          # U-Net 基线 与 Attention U-Net 改进版
│   ├── losses.py          # BCE / Dice / BCE+Dice / Tversky
│   ├── train.py           # 训练（留 2 张训练图做验证，按 val Dice 存最优）
│   ├── evaluate.py        # 测试集指标 + 分割可视化（TP/FN/FP 着色）
│   └── morphology.py      # 血管形态特征分析（报告加分项）
├── outputs/               # 训练/评估产物（已 gitignore）
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖（国内建议清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 下载数据集并按 data/README.md 放置，然后预处理
python src/preprocess.py

# 3. 训练基线（AutoDL/Kaggle GPU 均可；本机无 GPU 会慢但能跑）
python src/train.py --model unet --epochs 100

# 4. 测试集评估
python src/evaluate.py --weights outputs/checkpoints/best.pth

# 5. 血管形态分析（对 GT 或预测掩膜均可）
python src/morphology.py --mask outputs/predictions/test/01_pred.png --fov data/DRIVE/processed/test/fov/01.png
```

主要可调参数见各脚本 `--help`：`--model unet|attn_unet`、`--loss bce|dice|bce_dice|tversky`、`--width`（首层通道数，显存不足就调小）。

## 参考达标线（DRIVE 上的常见水平）

| 指标 | 参考值 |
|---|---|
| Dice | 0.79 ~ 0.82 |
| Accuracy | 0.95 ~ 0.96 |
| Sensitivity | 0.76 ~ 0.80 |
| Specificity | 0.98 |
| AUC | 0.97 ~ 0.98 |

明显低于这组值时优先检查：预处理是否做了 CLAHE、数据增强是否生效、轮数/学习率是否足够。

## 与实验报告模板的对应

| 报告章节 | 对应内容 |
|---|---|
| 题目 | 本 README 标题（可再润色） |
| 数据集说明与预处理方法 | `data/README.md` + `src/preprocess.py` |
| 实验方案（模型原理和改进） | `src/models.py`（UNet / AttentionUNet）、`src/losses.py` |
| 评价指标 | `src/evaluate.py`（Dice/IoU/SE/SP/Acc/AUC 的实现） |
| 实验结果与分析 | `outputs/train_log.csv`（训练曲线）、`outputs/eval_report.csv`、预测可视化图 |
| 改进方向 | 细血管召回（clDice/拓扑连续性）、跨数据集泛化（STARE/CHASE） |

## 三周计划

- **第 2 周**：跑通数据 → 基线 U-Net，出第一版指标
- **第 3 周**：Attention U-Net / Tversky 消融对比；形态分析出图
- **第 4 周**：整理图表、写报告、准备验收

## 数据与算力

- 数据集下载见 `data/README.md`（官网或国内镜像）
- 没有本地 GPU：用 Kaggle 免费 GPU（国内可直连）或 AutoDL 租卡（约 1 元/小时）
- 不建议为了挂 Colab/Kaggle 在校园网长期挂代理
