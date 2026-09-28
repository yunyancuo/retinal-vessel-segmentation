# 基于眼底图像的视网膜血管分割与血管形态特征分析（FIVES）

课程实践仓库 · 选题：基于眼底图像的视网膜血管分割与血管形态特征分析

- **数据集**：FIVES，800 张 2048×2048 眼底图（600 训练 / 200 测试），专家逐像素血管标注
- **基线模型**：U-Net（编码器-解码器 + 跳跃连接）
- **改进模型**：Attention U-Net + Tversky 损失（加大漏检惩罚，提升细血管召回）
- **附加分析**：血管形态定量（密度、直径分布、总长度、迂曲度、分形维数）

## 目录结构

```text
retinal-vessel-segmentation/
├── src/
│   ├── prepare_fives.py  # FIVES 数据准备：parquet → 缩放1024 → 512裁块 → 绿通道CLAHE → 自动FOV
│   ├── dataset.py        # PyTorch Dataset：归一化 / 翻转旋转增强 / padding
│   ├── models.py         # U-Net 基线 与 Attention U-Net 改进版
│   ├── losses.py         # BCE / Dice / BCE+Dice / Tversky
│   ├── train.py          # 训练（按验证 Dice 存最优）
│   ├── evaluate.py       # 指标（FOV 内）+ 可视化（TP/FN/FP 着色）
│   ├── morphology*.py    # 血管形态特征分析（单图 / 批量）
│   └── preprocess.py     # （旧）DRIVE 数据预处理，保留备用
├── results/              # 测试结果包（逐块指标 CSV + 摘要）
├── figures/              # 报告用图
└── 实验报告.md
```

## 快速开始（数据与环境在服务器上）

```bash
# 1. 数据准备：FIVES parquet 放 data_src/ 后
python src/prepare_fives.py --src-dir data_src --out-dir data/FIVES/processed

# 2. 训练基线与改进
python src/train.py --data-root data/FIVES/processed --model unet --loss bce_dice \
    --epochs 15 --batch-size 4 --val-count 120 --out-dir outputs/unet_bce_dice
python src/train.py --data-root data/FIVES/processed --model attn_unet --loss tversky \
    --epochs 15 --batch-size 4 --val-count 120 --out-dir outputs/attn_unet_tversky

# 3. 测试集评估
python src/evaluate.py --data-root data/FIVES/processed --weights outputs/unet_bce_dice/checkpoints/best.pth

# 4. 形态分析（金标准 vs 预测）
python src/morphology_batch.py --sources 金标准=data/FIVES/processed/test/masks \
    基线=outputs/unet_bce_dice/predictions/test --fov-dir data/FIVES/processed/test/fov
```

## 测试结果（800 块）

| 指标 | 基线 U-Net | 注意力 U-Net（Tversky） |
|---|---|---|
| Dice | **0.8713** | 0.8575 |
| 敏感性 | 0.8602 | **0.8957** |
| 特异性 | **0.9912** | 0.9833 |
| AUC | **0.9900** | 0.9708 |

详见 `results/测试结果.md` 与 `实验报告.md`。
