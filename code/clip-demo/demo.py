"""
CLIP 基础演示 / CLIP Basic Demo
演示CLIP模型的基本使用方法
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def print_model_info(model):
    """打印模型基本信息"""
    print("\n" + "="*60)
    print("模型信息 / Model Information")
    print("="*60)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量 / Parameters: {param_count:,}")
    print(f"输入分辨率 / Input Resolution: {model.visual.input_resolution}×{model.visual.input_resolution}")
    print(f"上下文长度 / Context Length: {model.context_length}")
    print(f"词表大小 / Vocabulary Size: {model.vocab_size}")
    print(f"特征维度 / Feature Dimension: {model.visual.output_dim}")


def demo_image_text_similarity(model, preprocess, device):
    """演示图像-文本相似度计算"""
    print("\n" + "="*60)
    print("演示：图像-文本相似度 / Demo: Image-Text Similarity")
    print("="*60)

    # 创建示例图像和文本
    image = Image.new("RGB", (224, 224), color=(128, 128, 255))
    image_input = preprocess(image).unsqueeze(0).to(device)

    texts = [
        "a blue square",
        "a red circle",
        "a green triangle",
        "a purple rectangle"
    ]
    text_input = clip.tokenize(texts).to(device)

    # 提取特征
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("\n相似度结果 / Similarity Scores:")
    print("-"*60)
    for i, text in enumerate(texts):
        score = logits_per_image[0][i].item() * 100
        print(f"{text:30s}: {score:6.2f}%")


def demo_zeroshot_classification(model, preprocess, device):
    """演示零样本分类"""
    print("\n" + "="*60)
    print("演示：零样本分类 / Demo: Zero-Shot Classification")
    print("="*60)

    # 创建一个简单的测试图像（绿色）
    test_image = Image.new("RGB", (224, 224), color=(0, 128, 0))
    image_input = preprocess(test_image).unsqueeze(0).to(device)

    # 定义类别和提示模板
    classes = ["cat", "dog", "bird", "car", "house"]
    templates = [
        "a photo of a {}.",
        "a drawing of a {}.",
        "a photo of a large {}.",
        "a photo of the {}.",
    ]

    # 生成文本提示
    text_prompts = []
    for template in templates:
        for cls in classes:
            text_prompts.append(template.format(cls))

    text_input = clip.tokenize(text_prompts).to(device)

    # 提取图像特征
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 编码所有文本提示
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits = (100.0 * image_features @ text_features.T)

        # 对每个类别的多个提示进行平均
        num_templates = len(templates)
        class_logits = logits.view(len(text_prompts), -1).split(num_templates)
        class_scores = torch.stack([logits.mean() for logits in class_logits])

        # 获取Top预测
        probs = class_scores.softmax(dim=-1)
        values, indices = probs[0].topk(len(classes))

    print(f"\n测试图像: 绿色纯色图像 / Test Image: Green solid color")
    print("\n零样本分类结果 / Zero-Shot Classification Results:")
    print("-"*60)
    for i, (value, idx) in enumerate(zip(values, indices)):
        cls_name = classes[idx]
        prob = value.item() * 100
        print(f"{i+1}. {cls_name:15s}: {prob:6.2f}%")


def demo_prompt_engineering(model, preprocess, device):
    """演示提示工程的效果"""
    print("\n" + "="*60)
    print("演示：提示工程 / Demo: Prompt Engineering")
    print("="*60)

    # 测试图像：一个红色圆形
    test_image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    image_input = preprocess(test_image).unsqueeze(0).to(device)

    # 不同的提示模板
    simple_prompts = ["a photo of a red circle"]

    detailed_prompts = [
        "a photo of a red circle",
        "a red circular shape",
        "a photograph showing a red circle",
        "a drawing of a red circle",
        "a red circle in the center",
    ]

    # 编码图像
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 测试简单提示
    simple_text = clip.tokenize(simple_prompts).to(device)
    with torch.no_grad():
        simple_features = model.encode_text(simple_text)
        simple_features = simple_features / simple_features.norm(dim=-1, keepdim=True)
        simple_sim = (100.0 * image_features @ simple_features.T)[0][0].item()

    # 测试详细提示
    detailed_text = clip.tokenize(detailed_prompts).to(device)
    with torch.no_grad():
        detailed_features = model.encode_text(detailed_text)
        detailed_features = detailed_features / detailed_features.norm(dim=-1, keepdim=True)
        detailed_sims = (100.0 * image_features @ detailed_features.T)[0]
        detailed_sim = detailed_sims.mean().item()

    print(f"\n测试图像: 红色圆形 / Test Image: Red Circle")
    print("\n提示工程对比 / Prompt Engineering Comparison:")
    print("-"*60)
    print(f"单一提示 (Single Prompt): {simple_sim:.2f}")
    print(f"多提示平均 (Multiple Prompts Average): {detailed_sim:.2f}")
    print(f"\n改进幅度 / Improvement: {detailed_sim - simple_sim:+.2f}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("CLIP 演示程序 / CLIP Demo Program")
    print("="*60)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备 / Device: {device}")

    # 显示可用模型
    print("\n可用模型 / Available Models:")
    for model_name in clip.available_models():
        print(f"  - {model_name}")

    # 加载模型
    print("\n正在加载模型 / Loading model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 打印模型信息
    print_model_info(model)

    # 运行各种演示
    demo_image_text_similarity(model, preprocess, device)
    demo_zeroshot_classification(model, preprocess, device)
    demo_prompt_engineering(model, preprocess, device)

    print("\n" + "="*60)
    print("演示完成 / Demo Complete")
    print("="*60)


if __name__ == "__main__":
    main()
