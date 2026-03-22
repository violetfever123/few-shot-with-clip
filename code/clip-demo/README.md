# CLIP 基础实验 / CLIP Basic Experiment

---

## 实验目的 / Purpose

演示CLIP模型的基本使用方法，包括：
- 模型加载和配置
- 图像和文本的预处理
- 计算图像-文本相似度
- 零样本图像分类

---

## 数据集 / Dataset

本实验使用：
- **内置图像**: skimage库中的示例图像
- **示例文本**: 手动定义的图像描述

---

## 运行方法 / How to Run

### 安装依赖

```bash
pip install torch torchvision clip ftfy regex tqdm pillow matplotlib scikit-image
```

### 运行代码

```bash
cd few-shot-with-clip/code/clip-demo
python demo.py
```

或运行Jupyter Notebook：

```bash
jupyter notebook demo.ipynb
```

---

## 实验结果 / Results

### 模型信息

```
可用模型 (Available Models):
- RN50
- RN101
- RN50x4
- RN50x16
- ViT-B/32
- ViT-B/16
```

**ViT-B/32模型规格**:
- 参数量: 151,277,313
- 输入分辨率: 224×224
- 上下文长度: 77 tokens
- 词表大小: 49,408

### 图像-文本匹配

CLIP能够准确匹配图像和对应的文本描述，例如：
- 宇航员照片 ↔ "a portrait of an astronaut with American flag"
- 摩托车照片 ↔ "a red motorcycle standing in a garage"
- 咖啡杯照片 ↔ "a cup of coffee on a saucer"

### 零样本分类示例

在CIFAR-100数据集上：
- CLIP可以为图像提供合理的类别预测
- 使用多个文本模板可以提高分类准确性
- Top-5准确率通常高于Top-1

---

## 核心代码说明 / Key Code Explanation

### 1. 模型加载

```python
import clip
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
```

### 2. 图像预处理

```python
from PIL import Image
image = Image.open("image.jpg")
image_input = preprocess(image).unsqueeze(0).cuda()
```

预处理包括：
- 调整大小到224×224
- 中心裁剪
- 归一化（使用ImageNet均值和标准差）

### 3. 文本标记化

```python
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).cuda()
```

默认填充到77个token，不区分大小写。

### 4. 特征提取

```python
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)
```

### 5. 相似度计算

```python
# 归一化
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 余弦相似度
similarity = (100.0 * image_features @ text_features.T)
probs = similarity.softmax(dim=-1)
```

---

## 提示工程建议 / Prompt Engineering Tips

1. **使用多种模板**: 不要只使用 "a photo of a {}"
2. **添加上下文**: 考虑图像的场景和语境
3. **具体化描述**: 包含颜色、大小、位置等细节
4. **测试不同格式**: "a {}", "a type of {}", "a photo showing {}"

### 推荐模板

```python
templates = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a drawing of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of the {}.",
    "a {} in a video game.",
]
```

---

## 注意事项 / Notes

- CLIP模型较大，首次下载需要时间
- 建议使用GPU运行（CUDA）
- 图像分辨率固定为224×224（ViT-B/32）
- 文本最大长度为77个token
- 特征维度为512维

---

## 扩展方向 / Extensions

1. **Few-shot分类**: 结合少量标注数据提升性能
2. **领域适应**: 针对特定领域（医学、遥感等）优化
3. **检索应用**: 图像检索和文本检索系统
4. **多语言支持**: 扩展到非英语语言
