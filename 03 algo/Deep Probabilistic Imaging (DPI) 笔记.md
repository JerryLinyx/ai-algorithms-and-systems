
本文档整理自论文《Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging》

## 一、研究背景与动机

### 欠定性（Underdetermined）

在许多成像问题中，观测数据不足以唯一确定一个图像解，即使没有噪声也可能存在无穷多个图像满足同一个观测。这类问题称为“欠定问题”，需要引入先验信息来缩小解空间。

### 非凸性（Non-convexity）

当成像系统的 forward model 是非线性时（例如闭合相位观测或压缩感知MRI），反问题往往是非凸的。这意味着：
- 存在多个局部最优解；
- 传统优化方法容易陷入局部；
- 贝叶斯后验具有复杂结构。

### 核心需求

为了解决这些问题，我们希望：
- 不只是给出一个最优图像解，而是估计完整的后验分布；
- 理解图像重建的不确定性；
- 建模后验分布中的多个模式（multi-modality）。

---

## 二、传统方法及其局限性

### (1) 正则化最大似然（RML）

RML 是一个经典的确定性优化方法：

$$
\hat{x} = \arg \min_x \left\{ \mathcal{L}(f_y(x), y) + \lambda R(x) \right\}
\tag{1}
$$

- $begin:math:text$ \\mathcal{L}(f_y(x), y) $end:math:text$：观测模型下的拟合损失；
- $begin:math:text$ R(x) $end:math:text$：图像的正则项（如 TV、L1、MEM）；
- $begin:math:text$ \\lambda $end:math:text$：权重系数。

此方法只输出一个单一解，无法反映图像空间的结构或不确定性。

---

## 三、贝叶斯视角：后验建模

### (2) Maximum a Posteriori (MAP)

从贝叶斯视角来看，RML 等价于最大后验估计：

$$
\hat{x} = \arg \max_x \left\{ \log p(y \mid x) + \log p(x) \right\}
\tag{2}
$$

- $begin:math:text$ \\log p(y \\mid x) $end:math:text$：观测数据的似然；
- $begin:math:text$ \\log p(x) $end:math:text$：图像的先验概率。

这等价于 (1)，但无法提供后验分布。

---

## 四、深度先验方法：Deep Image Prior (DIP)

### (3) DIP 框架

DIP 使用网络结构作为隐式先验：

$$
x = g_w(z), \quad w^* = \arg \min_w \mathcal{L}(f_y(g_w(z)), y)
\tag{3}
$$

- $begin:math:text$ g_w(z) $end:math:text$：神经网络生成图像；
- $begin:math:text$ z $end:math:text$：固定随机向量；
- $begin:math:text$ w $end:math:text$：网络权重。

此方法的输出仍是单一图像，不具备概率建模能力。

---

## 五、DPI 核心思想：后验分布建模

使用 flow-based 生成模型进行后验建模。

### Flow-based 模型概述 

#### (4) 可逆变换关系

$$
x = G_\theta(z), \quad z = G_\theta^{-1}(x), \quad z \sim \mathcal{N}(0, I)
\tag{4}
$$

- $G_{\theta}$：可逆神经网络；
- $z$：标准高斯 latent 变量；
- 这种变换允许计算密度和采样。

---

#### (5) Change-of-variables 定理

给定变换 $begin:math:text$ x = G(z) $end:math:text$，则其密度为：

$$
\log q_\theta(x) = \log p_Z(z) - \log \left| \det \left( \frac{\partial G_\theta(z)}{\partial z} \right) \right|
\tag{5}
$$

- 后项为 Jacobian determinant；
- 控制样本空间的体积变换，直接影响分布熵。

---

## 六、DPI 的训练方法

### (6) 定义生成分布

$$
x \sim q_\theta(x), \quad \text{其中 } x = G_\theta(z), \ z \sim \mathcal{N}(0, 1)
\tag{6}
$$

---

### (7) 目标函数：最小化 KL 散度

$$
\theta^* = \arg \min_\theta D_{\text{KL}}(q_\theta(x) \, \| \, p(x \mid y))
\tag{7}
$$

---

### (8) 展开 KL，代入后验表达式：

$$
\theta^* = \arg \min_\theta \mathbb{E}_{x \sim q_\theta} \left[
- \log p(y \mid x) - \log p(x) + \log q_\theta(x)
\right]
\tag{8}
$$

---

### (9) 替换 q(x) 的 log 概率表达式

将 $begin:math:text$ \\log q_\\theta(x) $end:math:text$ 替换为公式 (5) 所得表达，得到：

$$
\theta^* = \arg \min_\theta \mathbb{E}_{z \sim \mathcal{N}(0,1)} \left[
L(y, f(G_\theta(z))) + \lambda R(G_\theta(z)) - \beta \log \left| \det \left( \frac{\partial G_\theta}{\partial z} \right) \right|
\right]
\tag{9}
$$

- $begin:math:text$ \\beta $end:math:text$：调节熵项的强度；
- 当 $begin:math:text$\\beta$end:math:text$ 增大，鼓励更大的分布熵，生成多样化图像；
- 当 $begin:math:text$\\beta$end:math:text$ 减小，鼓励更精确地拟合观测，但可能导致 collapse。

---

### (10) Monte Carlo 近似损失

使用 N 个采样点近似期望：

$$
\theta^* = \arg \min_\theta \frac{1}{N} \sum_{k=1}^{N} \left[
L(y, f(G_\theta(z_k))) + \lambda R(G_\theta(z_k)) - \beta \log \left| \det \left( \frac{\partial G_\theta}{\partial z_k} \right) \right|
\right]
\tag{10}
$$

---

## 七、Toy Example 中的变体简化（势函数）

在 toy setting 中，构造势函数：

$$
p(x \mid y) \propto \exp(-\Phi(x))
$$

于是 KL 可写为：

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim q_\theta(x)}[\Phi(x)] - \mathcal{H}[q_\theta(x)]
\tag{11}
$$

- $begin:math:text$\\Phi(x)$end:math:text$：势函数，定义后验的结构；
- $begin:math:text$\\mathcal{H}[q]$end:math:text$：q 分布的熵；
- 熵通过 Jacobian determinant 间接参与优化。

---

## 小结

DPI 利用 flow 模型建模后验分布，通过优化包含数据项、正则项和熵项的目标函数，在欠定和非凸的计算成像问题中实现了：
- 后验建模；
- 多模态表示；
- 不确定性量化。