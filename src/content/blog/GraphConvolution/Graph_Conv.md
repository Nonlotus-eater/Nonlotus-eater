---
title: "GConv-S图卷积原理及模型应用"
description: "Graph Convolution & its application"
publishDate: 2025-11-17
tags: ["machine-learning", "deep-learning", "fine-tuning", "lora", "large-language-models", "parameter-efficient"]
comment: true
---

## 图卷积发展简介

图卷积网络（Graph Convolutional Networks, GCNs）的起源源于深度学习在规则欧氏数据（如图像）上取得的巨大成功与图数据这一非欧氏结构处理需求之间的矛盾。传统卷积神经网络（CNN）在图像等具有规则网格结构的数据上表现出色，其核心在于利用局部连接和权值共享机制有效提取空间特征。然而，当面对社交网络、分子结构、知识图谱等不规则图数据时，节点连接关系的任意性和拓扑结构的不规则性使得直接应用CNN的卷积操作变得困难。

在图卷积的早期探索中，研究者提出了两种主要的技术路线：

### 空间方法

试图将图数据映射到规则的空间坐标系中，然后借鉴图像处理的方式进行分块卷积操作。

具体而言，通过为图节点赋予空间坐标，将图结构转换为类似于图像的规则网格表示，再应用传统的CNN架构。

缺点：
   
   1. 无法有效处理边权
   2. 导致图拓扑结构失真（强制将非欧氏空间的图数据嵌入到欧氏空间） 
   3. 无法保证卷积操作的平移不变性

### 频域方法

在频域中定义图卷积操作。

核心思想：利用图拉普拉斯矩阵的特征分解进行图傅里叶变换，将图信号从空间域转换到频域，然后在频域中应用可学习的滤波器进行卷积，最后通过逆变换回到空间域。

## 数学原理

$$\theta_{*g} x = U \theta(\Lambda)U^T x$$

其中$\theta_{*g}$表示图卷积算子，$\theta$表示滤波器，$*g$表示卷积核，$U$是输入数据的Laplace矩阵的特征向量矩阵，$\Lambda$是特征值对角阵。

故$\theta(\Lambda)$可以理解为图数据的频域滤波操作，整个公式可以理解为对图数据进行频域滤波后再进行图傅里叶变换。

其中对称归一化的Laplace矩阵满足以下性质：

$$ L = I_n - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$$

其中W为输入图的邻接矩阵，D为输入图的度矩阵。

然而，$U \theta(\Lambda)U^T$的时间复杂度达到了$O(n^3)$，运算代价太大，最好的简化方法是将$\theta(\Lambda)$当成多项式进行计算，此时需要有一种能够近似频域滤波图的多项式，这里图卷积采用了Chebyshev多项式进行近似。

<Aside type="note" title="Chebyshev多项式">
图卷积使用到的性质如下：

$$
对\forall x \in [-1,1], \exists f(x), 满足 f(x) \approx \Sigma_{k=0}^\infty \theta_k T_k(x)
$$
其中
$$
T_0(x)=1, T_1(x)=x, T_2(x)=2xT_1(x)-T_0(x),...,T_{n+1}(x) = 2xT_n(x)-T_{n-1}(x)
$$

由Chebyshev多项式的性质，若要将$\theta(\Lambda)$当成多项式进行计算，需将$\Lambda$映射到$[-1,1]$，故对$\Lambda$进行如下操作：

$$
\tilde{\Lambda} = 2 \cdot \frac{\Lambda}{\lambda_{max}} - I_n
$$
</Aside>
111
215154
## 代码