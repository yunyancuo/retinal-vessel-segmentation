# 数据集放置说明

本目录存放 DRIVE 数据集。**数据不入库**（已在 .gitignore 中忽略），只保留本说明。

## 下载

- 官网：<https://drive.grand-challenge.org/>（需注册）
- 国内镜像：百度 AI Studio 数据集频道搜索「DRIVE retina」

## 期望目录结构

解压后按如下结构放到 `data/DRIVE/`：

```text
data/DRIVE/
├── training/
│   ├── images/          # 21_training.tif ~ 40_training.tif
│   ├── 1st_manual/      # 21_manual1.gif  ~ 40_manual1.gif（金标准血管标注）
│   └── mask/            # 21_training_mask.gif（视野 FOV 掩膜）
└── test/
    ├── images/          # 01_test.tif ~ 20_test.tif
    ├── 1st_manual/      # 01_manual1.gif
    └── mask/            # 01_test_mask.gif
```

## 预处理

```bash
python src/preprocess.py --drive-dir data/DRIVE --out-dir data/DRIVE/processed
```

输出 `processed/{training,test}/{images,masks,fov}/*.png`，之后训练/评估只读 `processed` 目录：

- `images/`：绿色通道 + CLAHE 增强后的灰度图
- `masks/`：血管金标准（0/255）
- `fov/`：视野掩膜（评估时只在视野内算指标，去掉图像黑边干扰）
