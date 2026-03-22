# CLIP 提示工程指南 / CLIP Prompt Engineering Guide

---

## 概述 / Overview

提示工程（Prompt Engineering）是提升CLIP模型性能的关键技术。通过精心设计的文本模板和描述，可以显著提高零样本分类和检索的准确性。

---

## 基本原理 / Basic Principles

### 1. 提示模板 / Prompt Templates

CLIP使用文本提示来生成类别嵌入，不同的模板会产生不同的性能表现。

**基本格式**：
```python
template = "a photo of a {}."  # {} 将被类别名称替换
```

**示例**：
```python
templates = [
    "a photo of a {}.",
    "a drawing of a {}.",
    "a {} in a photo.",
]
```

### 2. 多提示集成 / Multiple Prompt Ensemble

使用多个模板并对结果进行平均，可以提高分类稳定性。

**代码示例**：
```python
import torch
import clip

model, preprocess = clip.load("ViT-B/32")

# 类别列表
classes = ["cat", "dog", "bird"]

# 多个模板
templates = [
    "a photo of a {}.",
    "a drawing of a {}.",
    "a photo of the {}.",
    "a type of {}.",
]

# 生成所有提示
all_prompts = []
for template in templates:
    for cls in classes:
        all_prompts.append(template.format(cls))

# 编码所有提示
text_tokens = clip.tokenize(all_prompts).cuda()
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 按模板分组
num_templates = len(templates)
features_per_class = text_features.view(num_templates, -1, text_features.shape[-1])

# 对每个类别的多个提示求平均
class_embeddings = features_per_class.mean(dim=0)
```

---

## 提示模板分类 / Prompt Template Categories

### 1. 基础模板 / Basic Templates

```python
basic_templates = [
    "a photo of a {}.",
    "a photo of {}.",
    "a {}.",
]
```

### 2. 场景模板 / Contextual Templates

```python
contextual_templates = [
    "a photo of a {} in the wild.",
    "a close-up photo of a {}.",
    "a photo showing a {}.",
    "a {} in a natural setting.",
]
```

### 3. 风格模板 / Style Templates

```python
style_templates = [
    "a photo of a {}.",
    "a drawing of a {}.",
    "a painting of a {}.",
    "a sketch of a {}.",
    "a cartoon of a {}.",
    "a 3D render of a {}.",
]
```

### 4. 质量模板 / Quality Templates

```python
quality_templates = [
    "a photo of a {}.",
    "a good photo of a {}.",
    "a bad photo of a {}.",
    "a high quality photo of a {}.",
    "a low resolution photo of a {}.",
]
```

### 5. 属性模板 / Attribute Templates

```python
attribute_templates = [
    "a large {}.",
    "a small {}.",
    "a big {}.",
    "a tiny {}.",
    "a clean {}.",
    "a dirty {}.",
]
```

---

## 针对特定数据集的模板 / Dataset-Specific Templates

### ImageNet

```python
imagenet_templates = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of my {}.",
    "a drawing of a {}.",
    "a centered photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a photo showing the {}.",
]
```

### CIFAR-10/100

```python
cifar_templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
]
```

### Food101

```python
food_templates = [
    "a photo of {}, a type of food.",
    "a photo of a {}.",
    "a picture of {}.",
]
```

### Birds (鸟类)

```python
bird_templates = [
    "a photo of a {}, a type of bird.",
    "a bird called a {}.",
    "a photograph of a {}.",
]
```

### Flowers (花卉)

```python
flower_templates = [
    "a photo of a {}, a type of flower.",
    "a flower called a {}.",
    "a picture of a {}.",
]
```

---

## 高级技巧 / Advanced Techniques

### 1. 动态模板生成 / Dynamic Template Generation

根据类别名称自动生成更合适的提示：

```python
def generate_contextual_prompts(class_name):
    """根据类别名称生成上下文提示"""
    # 复数处理
    if class_name.endswith('s'):
        singular = class_name[:-1]
        plural = class_name
    else:
        singular = class_name
        plural = class_name + 's'

    return [
        f"a photo of {singular}.",
        f"a photo of {plural}.",
        f"a photo of many {plural}.",
        f"a photo showing {singular}.",
    ]
```

### 2. 类别名称优化 / Class Name Optimization

优化类别名称以提高匹配度：

```python
# 不好的类别名称
bad_names = [
    "crane",  # 可能被理解为机器或鸟类
    "bank",    # 可能被理解为银行或河岸
]

# 优化的类别名称
good_names = [
    "construction crane",  # 明确为机器
    "river bank",         # 明确为河岸
    "financial bank",      # 明确为银行
]
```

### 3. 多语言提示 / Multi-language Prompts

如果CLIP训练数据包含多种语言，可以尝试：

```python
multi_lang_templates = [
    "a photo of a {}.",           # English
    "一张{}的照片。",              # Chinese
    "Une photo de {}.",             # French
]
```

### 4. 域特定提示 / Domain-Specific Prompts

针对特定领域定制提示：

```python
# 医学影像
medical_templates = [
    "a medical image showing a {}.",
    "an X-ray of a {}.",
    "an MRI scan of a {}.",
]

# 遥感图像
satellite_templates = [
    "a satellite image of {}.",
    "an aerial photo showing {}.",
    "remote sensing imagery of {}.",
]
```

---

## 实用代码示例 / Practical Code Examples

### 完整的零样本分类函数

```python
import torch
import clip
from typing import List, Tuple

def zeroshot_classifier(
    model: clip.CLIP,
    classes: List[str],
    templates: List[str],
    device: str = "cuda"
) -> Tuple[torch.Tensor, List[str]]:
    """
    创建零样本分类器

    Args:
        model: CLIP模型
        classes: 类别列表
        templates: 提示模板列表
        device: 设备

    Returns:
        (class_embeddings, prompt_list): 类别嵌入和提示列表
    """
    # 生成所有提示
    prompts = []
    for template in templates:
        for class_name in classes:
            prompts.append(template.format(class_name))

    # 编码所有提示
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 按模板分组并平均
    num_templates = len(templates)
    num_classes = len(classes)

    # 重塑为 (num_templates, num_classes, feature_dim)
    features_reshaped = text_features.view(num_templates, num_classes, -1)

    # 对每个类别的多个提示求平均
    class_embeddings = features_reshaped.mean(dim=0)

    return class_embeddings, prompts


# 使用示例
model, preprocess = clip.load("ViT-B/32")

classes = ["cat", "dog", "bird", "car", "house"]
templates = [
    "a photo of a {}.",
    "a drawing of a {}.",
    "a photo showing the {}.",
]

class_embeddings, all_prompts = zeroshot_classifier(model, classes, templates)

print(f"生成了 {len(all_prompts)} 个提示")
print(f"类别嵌入形状: {class_embeddings.shape}")
```

### 提示效果评估

```python
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_prompts(
    model,
    image_features: torch.Tensor,
    classes: List[str],
    templates: List[str],
    ground_truth: int,
    device: str = "cuda"
) -> dict:
    """
    评估不同提示组合的性能

    Returns:
        包含top-1和top-5准确率的字典
    """
    results = {}

    for num_prompts in range(1, len(templates) + 1):
        # 使用前num_prompts个模板
        selected_templates = templates[:num_prompts]

        # 生成分类器
        class_embeddings, _ = zeroshot_classifier(
            model, classes, selected_templates, device
        )

        # 计算logits
        logits = 100.0 * (image_features @ class_embeddings.T)
        probs = logits.softmax(dim=-1)

        # 获取预测
        top1_pred = probs.argmax(dim=-1).cpu().numpy()
        top5_preds = probs.topk(5, dim=-1).indices.cpu().numpy()

        # 计算准确率
        top1_acc = accuracy_score([ground_truth], [top1_pred]) * 100
        top5_acc = (ground_truth in top5_preds[0]) * 100

        results[num_prompts] = {
            "top1": top1_acc,
            "top5": top5_acc,
        }

    return results
```

---

## 常见问题与解决方案 / Common Issues and Solutions

### 1. 类别相似导致混淆

**问题**: 相似的类别（如不同品种的狗）容易被混淆

**解决方案**:
- 使用更具体的描述
- 添加品种或特征信息
- 使用属性模板

```python
specific_templates = [
    "a photo of a {}, a breed of dog.",
    "a {} type of dog.",
    "a dog that is a {}.",
]
```

### 2. 域不匹配

**问题**: CLIP训练数据主要来自网络图片，对特定领域（医学、遥感等）性能差

**解决方案**:
- 使用领域相关的模板
- 添加领域特定的前缀
- 考虑领域自适应方法

```python
domain_templates = [
    f"a medical scan showing a {{}}.",
    f"an X-ray of a {{}}.",
]
```

### 3. 计算效率

**问题**: 使用大量提示导致计算开销大

**解决方案**:
- 提前计算并缓存文本嵌入
- 选择性使用模板
- 使用提示选择策略

```python
# 预计算文本嵌入
def precompute_text_embeddings(model, templates, classes, device):
    """预计算所有文本嵌入"""
    embeddings = []
    for template in templates:
        for cls in classes:
            text = template.format(cls)
            tokens = clip.tokenize(text).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb)
    return torch.stack(embeddings)

# 运行时直接使用预计算的嵌入
precomputed = precompute_text_embeddings(model, templates, classes, device)
```

---

## 最佳实践 / Best Practices

1. **从简单开始**: 先使用基本模板测试，再逐步优化
2. **多样化模板**: 使用不同类型、风格、质量的模板组合
3. **领域适配**: 根据应用领域定制提示
4. **性能评估**: 在验证集上评估不同提示组合的效果
5. **缓存计算**: 预计算和缓存文本嵌入以提高效率
6. **迭代优化**: 基于评估结果持续改进提示设计

---

## 参考资料 / References

- [CLIP Paper - Section on Prompt Engineering](https://arxiv.org/abs/2103.00020)
- [CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
- [OpenAI CLIP GitHub - prompts.md](https://github.com/openai/CLIP/blob/main/data/prompts.md)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
