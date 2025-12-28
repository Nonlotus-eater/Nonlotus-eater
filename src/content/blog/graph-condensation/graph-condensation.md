---
title: "GCond-S Graph Condensation: A Survey"
description: "图压缩的学习笔记"
publishDate: "2025-09-13"
tags: ["Astro", "Blog"]
draft: false
comment: true
---

## An overview of Graph condensation

![An overview of Graph Condensation](/images/graph_condensation/Overview.png)

> GC focuses on synthesizing a compact yet highly representative graph, enabling GNNs trained on it to achieve performance comparable to those trained on the original large graph.

图压缩旨在应对large scale的图训练任务，通过压缩图的节点个数、labels和节点特征达到减少训练量的作用。问题建模如下：

### Problem Modeling

1. 数据描述
   
   图数据集：A large-scale dataset $\mathcal{T}=(\mathcal{V}, \mathcal{E})$,$|\mathcal{V}|=N$,$|\mathcal{E}|=M$，包括节点特征矩阵$X\in \mathcal{R}^{N \times d}$,邻接矩阵$A\in \mathcal{R}^{N \times N}$和labels $Y$.
2. 任务要求
   
   图压缩需要找到一个小型压缩图$\mathcal{S}=(\mathcal{V'}, \mathcal{E'})$, 其中$|\mathcal{V'}|=N', N'<<N$，当然也包括压缩后的包括节点特征矩阵$X'\in \mathcal{R}^{N' \times d}$,邻接矩阵$A\in \mathcal{R}^{N' \times N'}$和labels $Y$.
3. 超参数
   1. 压缩率 $\tau = \frac{N'}{N}$

4. 目标函数
    
    图压缩的目的是希望通过一个参数化的中继图模型(relay graph model)$f_{\theta}(\cdot)$，找到一个小但保留信息的图 S，使其在中继模型中的表示接近原图，从而将图压缩的任务转化成一个优化问题：

    $$
    S = \arg\min_S \mathcal{L}_{cond}(f_\theta(T), f_\theta(S))
    $$
    其中$\mathcal{L}_{cond}$是图压缩的优化函数。

Graph Condensation主要分为两部分：Optimization strategies 和 Condensed graph generation。本文的后续部分将从这两部分进行介绍。

## Optimization Strategies

$\mathcal{T}$ 和 $\mathcal{S}$ 的Loss:

$$
\mathcal{L}^T(\theta) = \ell(f_\theta(T), \mathbf{Y}), \\
\mathcal{L}^S(\theta) = \ell(f_\theta(S), \mathbf{Y}'),
$$
其中$\ell$是任务特定的目标，如交叉熵，故图压缩的目标函数可以被塑造成接下来的bi-level问题：
$$
\min_S \mathcal{L}^T(\theta^S) \quad \text{s.t.} \quad \theta^S = \arg\min_\theta \mathcal{L}^S(\theta)
$$
结合公式，Bi-level Optimization的思路应该划分为内外两层：

1. 内循环：只使用压缩图 S 和其标签 Y' 来计算损失 $L^S$，并据此更新中继模型 $f_θ$ 的参数。
2. 外循环：使用原始图 T 和其标签 Y 来计算损失 L^T，并据此更新压缩图 S 的结构（节点特征 X' 和邻接矩阵 A'）。

![Bi-level Optimization](/images/graph_condensation/Algorithm1.png)

在具体的实现上，目前已有的方法包括Gradient Matching, Trajectory Matching, Kernel Ridge Regression(KRR), Distribution Matching四种。

| 优化策略 (Optimization Strategy) | 损失函数 (Loss Function) 与 数学符号解释 (Mathematical Symbols Explained) | 优缺点简述 (Brief Pros & Cons)                                       |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **梯度匹配 (Gradient Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta_0 \sim \Theta} \left[ \sum_{t=1}^T D(\nabla_\theta L_T(\theta_t), \nabla_\theta L_S(\theta_t)) \right]$<br>**解释:**<br>- $\mathbb{E}_{\theta_0 \sim \Theta}$: 对中继模型 $f_\theta$ 的初始参数 $\theta_0$ 从分布 $\Theta$ 中采样并求期望，以提高鲁棒性。<br>- $D(\cdot, \cdot)$: 距离度量函数（如余弦相似度或L2距离），用于衡量两个梯度向量的差异。<br>- $\nabla_\theta L_T(\theta_t)$: 在第 $t$ 步、模型参数为 $\theta_t$ 时，在**原始图 $T$** 上计算的任务损失 $L_T$ 对参数 $\theta$ 的梯度。<br>- $\nabla_\theta L_S(\theta_t)$: 在第 $t$ 步、模型参数为 $\theta_t$ 时，在**浓缩图 $S$** 上计算的任务损失 $L_S$ 对参数 $\theta$ 的梯度。<br>- $\theta_{t+1} = \text{opt}(L_S(\theta_t))$: 约束条件，表示模型参数 $\theta$ 仅在浓缩图 $S$ 上通过优化器 $\text{opt}(\cdot)$ 进行更新。 | **优:** 主流方法，效果好。<br>**缺:** 计算开销大，是双层优化。             |
| **轨迹匹配 (Trajectory Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta'_t \sim \Theta'} \left[ D(\theta_{t+T}^T, \theta_{t+L}^S) \right]$<br>**解释:**<br>- $\mathbb{E}_{\theta'_t \sim \Theta'}$: 对从原始图训练轨迹中采样的中间参数（检查点）$\theta'_t$ 从集合 $\Theta'$ 中采样并求期望。<br>- $\theta_{t+T}^T$: 从起点 $\theta'_t$ 开始，在**原始图 $T$** 上再训练 $T$ 步后得到的模型参数。<br>- $\theta_{t+L}^S$: 从**同一个起点 $\theta'_t$** 开始，在**浓缩图 $S$** 上再训练 $L$ 步后得到的模型参数。$T$ 和 $L$ 是控制更新步数的超参数。<br>-$\theta_{t+1}^T = \text{opt}(L_T(\theta_t^T))$ <br>-$\theta_{t+1}^S = \text{opt}(L_S(\theta_t^S))$ | **优:** 匹配更全局信息，性能通常更好。<br>**缺:** 计算开销极大，是三层优化。 |
| **核岭回归 (Kernel Ridge Regression)** | **公式:** $L_{\text{cond}} = \frac{1}{2} \| Y - K_{TS} (K_{SS} + \lambda I)^{-1} Y' \|^2$<br>**解释:**<br>- $K_{TS}$: 核矩阵，其元素 $K_{TS}[i,j]$ 表示原始图中第 $i$ 个样本与浓缩图中第 $j$ 个样本在核空间中的相似度。<br>- $K_{SS}$: 核矩阵，其元素 $K_{SS}[i,j]$ 表示浓缩图中第 $i$ 个样本与第 $j$ 个样本在核空间中的相似度。<br>- $\lambda$: 正则化系数| **优:** 计算高效，有闭式解。<br>**缺:** 核矩阵内存消耗大。     |
| **分布匹配 (Distribution Matching)** | **公式:** $L_{\text{cond}} = \mathbb{E}_{\theta_0 \sim \Theta} \left[ D(f_\theta(T), f_\theta(S)) \right]$<br>**解释:**<br>- $f_\theta(\cdot)$: 中继模型 $f_\theta$ 对图进行编码后得到的特征表示集合。 | **优:** 计算最高效，无梯度计算。<br>**缺:** 通常需类别标签。 |

## Condensed Graph Generation

压缩图$\mathcal{S}=(\mathcal{X'}, \mathcal{A'})$由特征生成(Feature Generation)和图结构生成(Structure Generation)两个核心部分组成：

### Feature Generation

特征矩阵$X'$很大程度上由反向传播和梯度下降来迭代更新，导致了特征矩阵的最终效果对初始化方法非常敏感，不同的初始化会导致收敛速度和最终性能的差异。所以不同的初始化策略相当重要，如随机采样、核心集选择(coreset method)和聚类的方法；此外还有非梯度的初始化方法，求出初始化的闭式解情况。

### Structure Generation

压缩图的邻接矩阵$A'$ 定义了压缩节点之间的关系，旨在保留原始图的拓扑信息。其构建方式（如同质性、谱特性、稀疏性）会显著影响GNN的性能。

目前的研究包括生成式模型(Generative Model)、Parameterization、Pre-defined Structure和Condensed Graph Sparsification等四种方法。

## Evaluation Metric

评估指标可以从Effectiveness/Generalization/Efficiency/Fairness/Robustness五个角度出发，评估压缩图的性能：

### Effectiveness
评估指标为**准确率（Accuracy）**。通过比较在不同压缩率（condensation ratio）下，使用压缩图训练的GNN模型与使用原始图训练的GNN模型在特定下游任务（如节点分类）上的准确率来衡量。准确率越接近原始图的性能，说明该图压缩方法越有效。

### Generalization
评估指标为**跨架构和跨任务的平均性能**。具体而言，是在压缩图上训练多种不同的GNN架构（如GCN, SAGE, GAT等）并执行多种下游任务（如节点分类、链接预测、异常检测等），然后计算其平均准确率。公式表示为：对于特定任务 *t*，其泛化性能 *pt* 为 *n* 个GNN架构准确率的平均值，即 
$$pt = (1/n) ∑ acc_i$$
该值越高，表明压缩图的泛化能力越强。

### Efficiency
评估指标为**压缩过程总耗时（Total Time）**。直接测量从开始到生成最终压缩图所需的全部时间。耗时越短，表明该方法的效率越高，越具有实际应用的可行性。

### Fairness
评估指标为**模型预测偏差（Bias）**，主要使用两个公平性度量标准：
*   **人口统计均等性差异 (ΔDP)**: 衡量模型预测结果在不同敏感群体间的差异。
  $$ΔDP = |P(ŷ=1|s=0) - P(ŷ=1|s=1)|$$
   
*   **机会均等性差异 (ΔEO)**: 衡量在真实正例中，不同敏感群体获得正向预测的几率差异。
  $$ΔEO = |P(ŷ=1|y=1, s=0) - P(ŷ=1|y=1, s=1)|$$
GC方法若能有效降低GNN模型在压缩图上的ΔDP和ΔEO，使其低于或等于在原始图上的偏差，则被认为是更公平的。

### Robustness
评估指标为**在噪声数据上的准确率（Accuracy under Noise）**。通过在原始图中引入不同水平的噪声（如随机添加或删除边），然后基于噪声图生成压缩图，并评估在此压缩图上训练的GNN模型的准确率。准确率越高，表明该GC方法在压缩过程中能更好地保留核心信息，对原始图中的噪声更具鲁棒性。

## 阅读感受

1. 图压缩通常用于分类问题，因为不可避免的要在损失函数和优化策略中使用到分类标签和压缩的分类标签$Y 和 Y'$,所以若要使用于其他任务，需要开发任务无关(Task-Agnostic)/自监督(Self-supervised)的优化策略
   1. 可以参考的论文：
      1. CTGC:使用self-supervised learning，输入数据完全避免标签。[Contrastive Graph Condensation: Advancing Data Versatility through Self-Supervised Learning](https://dl.acm.org/doi/abs/10.1145/3711896.3736892)