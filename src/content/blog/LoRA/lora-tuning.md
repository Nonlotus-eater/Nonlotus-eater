---
title: "LoRA微调原理详解：高效的大语言模型适配技术"
description: "LoRA(Low-Rank Adaptation)微调技术的数学原理、实现细节和应用实践"
publishDate: 2025-11-17
tags: ["machine-learning", "deep-learning", "fine-tuning", "lora", "large-language-models", "parameter-efficient"]
comment: true
---

# LoRA Fine-tuning

## 数学原理

首先设置$W_0 \in R^{d \times k}$为神经网络当前层的权重矩阵，其中$d$是上一层的输出维度，$k$是下一层的输入维度。

### 低秩分解(Low-Rank Decomposition)

一个矩阵 $M \in \mathbb{R}^{m \times n}$ 的秩 $r$ 定义为： 
$$ \text{rank}(M) = \dim(\text{Im}(M)) = \dim(\text{span}{\text{columns of } M}) $$
矩阵的秩等于其列向量组的最大线性无关向量个数，也等于其行向量组的最大线性无关向量个数，即$\text{rank}(M) \leq \min(m, n)$

那么一个矩阵的低秩分解可以被定义如下：

如果矩阵 $M$ 的秩为 $r$（即 $\text{rank}(M) = r \leq \min(m, n)$），则 $M$ 可以分解为两个低秩矩阵的乘积： 
$$M = L \cdot R$$ 
其中：
$L \in \mathbb{R}^{m \times r}$ 是左侧低秩矩阵，$R \in \mathbb{R}^{r \times n}$ 是右侧低秩矩阵，$r \ll \min(m, n)$ 表示秩相对较小

### LoRA微调

在LoRA中，权重矩阵的更新被限制为低秩形式： 
$$W' = W + \Delta W = W + BA$$ 
其中：
$W \in \mathbb{R}^{d \times k}$ 是原始权重矩阵，$B \in \mathbb{R}^{d \times r}$ 是低秩分解的左侧矩阵，$A \in \mathbb{R}^{r \times k}$ 是低秩分解的右侧矩阵,$r \ll \min(d, k)$ 是秩的维度(模拟所谓本征秩Intrinsic rank)。

所以在低秩分解的前向传播过程应该理解为：
$$
h = W'x = W_0 x + BA x
$$
在微调的过程中，我们冻结t_0部分的权重，将矩阵A随机高斯初始化，将矩阵B零初始化，随后使用Adam优化器进行训练，最后将$W_0$与权重$BA$相加，得到优化后的参数。

![LoRA Mechanism](/images/LoRA/LoRA_mechanism.png)

### 应用于Transformer的情况

![LoRA in Transformer](/images/LoRA/LoRA_in_Transformer.png)

LoRA在Transformer中主要应用于$W_Q,W_K,W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$与输出投影层$W_O \in \mathbb{R}^{d_v \times d_{\text{model}}}$四个矩阵，并冻结MLP矩阵中的权重。

通过消融实验，文章发现当只调整$W_Q,W_V$时才取得最佳效果。

