[视频：Flow-based Generative Model](https://www.youtube.com/watch?v=uXY18nzdSsM)
[细水长flow之NICE：流模型的基本概念与实现](https://kexue.fm/archives/5776)

  

## Introduction

主流的生成模型可分为三类，component-by-component（autoregressive）、AutoEncoder 和 GAN，但是这些模型都有着或多或少的问题：

* component-by-component 指 逐个元素地生成，对于图片来说，生成顺序不一定是最佳的，而且逐元素的生成是很慢的; Slow generation；

* VAE要优化的目标ELBO和实际需要优化的目标 对数似然 之间存在距离; Optimizing a lower bound

* GAN一方面训练不稳定，另一方面因为不是component-by-component，所以在语音生成上不如 autoregressive

  
  

Flow-based model可以解决上述问题 (效果上不如GAN)，对于从Normalized Distribution $\pi(z)$ 生成复杂的目标分布 $p_G(x)$ 而言，需要优化的目标可以表示为 最大化 $p_G$ 采样的对数似然：

  
  

$$

\begin{align}

G^* & = \arg\max_{G} \sum_{i = 1}^{m} p_G(x^i) \nonumber \\

& \approx \arg\min_{G} \sum_{i}^{m} D_{KL}(p_{\text{data}} \| p_G) \nonumber

\end{align}

$$

  
  
  

其中， $\{x^1, x^2, ... , x^i\}$ from $p_{data}(x)$, $p_G(x^i)$ 是真实图片在生成分布 $p_G$ 上的似然 likelihood (KL Divergence)，第二行即VAE优化的目标ELBO，而Flow-based model可以直接优化第一行。

  

使得$P_G$ 和 $P_{data}$ 越像越好

>Flow based model directly optimizes the object function
![[assets/Flow-based Model/file-20260302003402129.png]]
---

## Jacobian Matrix

假设 $\mathbf{z}$ 和 $\mathbf{x}$ 直接满足以下关系：

$$
\mathbf{x} = f(\mathbf{z}), \quad \mathbf{z} = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

那么 $\mathbf{z}$ 与 $\mathbf{x}$ 之间的导数可以写作一个偏微分矩阵，如下所示：

$$
J_f =
\overset{\text{input } \mathbf{z}}{
\overbrace{
\left[
\begin{array}{cc}
\frac{\partial x_1}{\partial z_1} & \frac{\partial x_1}{\partial z_2} \\
\frac{\partial x_2}{\partial z_1} & \frac{\partial x_2}{\partial z_2}
\end{array}
\right]
}
}
\left|
\text{output } \mathbf{x}
\right.
$$

该矩阵被称为 Jacobian Matrix（**雅各比矩阵**）。

性质：假如存在反函数 $\mathbf{z} = f^{-1}(\mathbf{x})$，则 Jacobian Matrix 可以表示为：

$$
J_{f^{-1}} = 
\begin{bmatrix}
\frac{\partial z_1}{\partial x_1} & \frac{\partial z_1}{\partial x_2} \\
\frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2}
\end{bmatrix}
$$

且 **两个逆过程的 Jacobian Matrix 相乘后等于单位矩阵**：

$$
J_{f^{-1}} \otimes J_f = \mathbf{I}
$$

这说明，互为反函数的两个函数，对应的 Jacobian Matrix 互为逆矩阵。

---
## Determinant（秩）

从几何意义上讲，秩代表了由矩阵的行向量所组成的图形的面积/体积/...，因此是非负数。

而且对于任意一个矩阵 $A$，如果存在逆矩阵 $A^{-1}$，则有：

$$
\det(A) = \frac{1}{\det(A^{-1})}
$$

应用到反函数的 Jacobian Matrix，可以得到：

$$
\det(J_{f^{-1}}) = \frac{1}{\det(J_f)}
$$
---
## Change of Variable Theorem

该定理描述了当 $\mathbf{x} = f(\mathbf{z})$ 时，$p(\mathbf{x})$ 和 $p(\mathbf{z})$ 满足以下关系：

$$
p(\mathbf{x}) \cdot |\det(J_f)| = \pi(\mathbf{z})
$$

$$
p(\mathbf{x}) = \pi(\mathbf{z}) \cdot |\det(J_{f^{-1}})|
$$

要加绝对值，$\mathbf{x}$ 和 $\mathbf{z}$ 变化方向不确定
为了方便理解，下面以二维的情形进行证明： 

![[assets/Flow-based Model/file-20260302003402131.png]]
### 证明

如上图所示，在 $\mathbf{z}$ 平面上随机选择一点 $\mathbf{z}'$ 及其周边极小的正方形区域，经过 $f(\cdot)$ 之后投影到 $\mathbf{x}$ 平面上一点 $\mathbf{x}'$ 及其周边的菱形区域，两个区域的概率应该相等，即：

$$
\pi(\mathbf{z}') = \frac{p(\mathbf{x}')}{\Delta z_1 \Delta z_2} \cdot 
\left| \det \begin{bmatrix} 
\Delta x_{11} & \Delta x_{12} \\
\Delta x_{21} & \Delta x_{22}
\end{bmatrix} \right|
$$

矩阵转置后秩不变，因此可以写为：

$$
\pi(\mathbf{z}') = \frac{p(\mathbf{x}')}{\Delta z_1 \Delta z_2} \cdot 
\left| \det \begin{bmatrix} 
\Delta x_{11} & \Delta x_{21} \\
\Delta x_{12} & \Delta x_{22}
\end{bmatrix} \right|
$$

进一步简化，表示为：

$$
\pi(\mathbf{z}') = p(\mathbf{x}') \cdot 
\left| \det \begin{bmatrix} 
\frac{\Delta x_{11}}{\Delta z_1} & \frac{\Delta x_{21}}{\Delta z_1} \\
\frac{\Delta x_{12}}{\Delta z_2} & \frac{\Delta x_{22}}{\Delta z_2}
\end{bmatrix} \right|
$$

$$
\pi(\mathbf{z}') = p(\mathbf{x}') \cdot 
\left| \det \begin{bmatrix} 
\frac{\partial x_{11}}{\partial z_1} & \frac{\partial x_{21}}{\partial z_1} \\
\frac{\partial x_{12}}{\partial z_2} & \frac{\partial x_{22}}{\partial z_2}
\end{bmatrix} \right|
$$

因此：

$$
\pi(\mathbf{z}') = p(\mathbf{x}') \cdot |\det(J_f)|
$$

---

### 结论

从图中不难看出，$\Delta x_{12}$ 表示因为 $\Delta z_2$ 导致的 $\Delta x_1$ 的变化，所以：

$$
\frac{\Delta x_{12}}{\Delta z_2} = \frac{\partial x_1}{\partial z_2}
$$

又因为 $\det(J_{f^{-1}}) = 1 / \det(J_f)$，所以：

$$
p(\mathbf{x}) = \pi(\mathbf{z}) \cdot |\det(J_{f^{-1}})|
$$
---
## Coupling Layer

根据 **Change of Variable Theorem** 可以得到：

$$
p_G(x^i) = \pi(z^i) \cdot |\det(J_{G^{-1}})|, \quad z^i = G^{-1}(x^i)
$$

代入最开始设定的优化目标，如式所示：

$$
G^* = \arg\max_G \sum_{i=1}^m \log p_G(x^i)
$$

等价于：

$$
G^* = \arg\max_G \sum_{i=1}^m \left[\log \pi(G^{-1}(x^i)) + \log |\det(J_{G^{-1}})|\right]
$$

---

### 对于展开式，存在两个问题：

1. **J的维度等于 $\mathbf{z}$ 和 $\mathbf{x}$ 的乘积，大维度下计算determinant比较困难**=> 设计好Generator
2. 因为训练时需要计算 $G^{-1}$，($G^{-1}$而非$G$是训练网络) ；确保$G$ 要 invertible，需要限制 $\mathbf{z}$ 和 $\mathbf{x}$ 的维度相同。在 GAN 中 $\mathbf{z}$ 的维度远小于 $\mathbf{x}$。

$G$ has limitations, 需要更多$G$



![[assets/Flow-based Model/file-20260302003402148.png]]

![[assets/Flow-based Model/file-20260302003402139.png]]
基于此，设计的 **Coupling Layer** 如下图所示

![[assets/Flow-based Model/file-20260302003402139 1.png]]

其中，网络 $F$ 和 $H$ 可以任意复杂，不影响 $G$ 的可逆性。这种新颖的 **网络结构** 解决了上述两个问题：

---

1. **针对求 $G^{-1}$，可以直接使用以下公式求解**：

$$
\mathbf{z}_{i \leq d} = \mathbf{x}_i, \quad
\mathbf{z}_{i > d} = \frac{\mathbf{x}_i - \gamma_i}{\beta_i}
$$

2. **针对计算 $\det(J_{G^{-1}})$，将 Jacobian matrix 列出**：

| $J_G$     | $z_1$          | $z_2$ | $\dots$ | $z_d$ | $z_{d+1}$  | $\dots$ | $z_D$ |
| --------- | -------------- | ----- | ------- | ----- | ---------- | ------- | ----- |
| $x_1$     | $I$ (Identity) | -     | -       | -     | $O$ (Zero) | -       | -     |
| $x_2$     | -              | -     | -       | -     | -          | -       | -     |
| $\vdots$  | -              | -     | -       | -     | $\vdots$   | -       | -     |
| $x_{d+1}$ | Don't care     | -     | -       | -     | Diagonal   | -       | -     |
| $\vdots$  | -              | -     | -       | -     | $\vdots$   | -       | -     |
| $x_D$     | -              | -     | -       | -     | -          | -       | -     |

---

### 解析

可见，这种将 $1 \sim d$ 维的向量直接拷贝的做法，使得雅各比矩阵的秩等于右下角的对角子矩阵，而该部分的秩等于 $\beta$ 的乘积：

$$
\det(J_{G^{-1}}) = 
\frac{\partial x_{d+1}}{\partial z_{d+1}} 
\cdot 
\frac{\partial x_{d+2}}{\partial z_{d+2}} 
\cdots 
\frac{\partial x_D}{\partial z_D} = \prod_{i=d+1}^D \beta_i
$$

为增加网络的复杂性，需要堆叠多个 $G^{-1}$，但是会造成一个问题：$1 \sim d$ 维的向量始终是不变的。为此，可以设置网络结构交替取反，如下图所示：
![[assets/Flow-based Model/file-20260302003402130.png]]

## 1×1 convolution

Google 提出了 Glow ([arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039))，将 Flow-based model 做到与 GAN 相媲美，主要的创新点包括 **Coupling Layer** 和 **1×1 convolution**，后者在目前的网络设计中十分常见，主要用来 shuffle channels，但它也可以被视作一种 Flow-based model。

---

### 对于 RGB 图像

设置卷积核 $W$ 大小为 $3 \times 3$，为了尽量使 $W$ 是可逆的，作者使用了一个可逆的 $3 \times 3$ 矩阵进行初始化，同时方便计算$W^{-1}$ 和 $det(W)$。针对 Flow-based model 损失函数中的 $G^{-1}$，可以套用公式计算 $W$ 的逆矩阵；针对 $\det(J_{G^{-1}})$，如下表所示：

| $J_{G^{-1}}$    | $z_{1,1}^{(1)}$ | $z_{1,1}^{(2)}$ | $z_{1,1}^{(3)}$ | $\dots$  | $z_{n,m}^{(1)}$ | $z_{n,m}^{(2)}$ | $z_{n,m}^{(3)}$ |
| --------------- | --------------- | --------------- | --------------- | -------- | --------------- | --------------- | --------------- |
| $x_{1,1}^{(1)}$ |                 |                 |                 | $\dots$  | Zeros           | Zeros           | Zeros           |
| $x_{1,1}^{(2)}$ |                 | $W$             |                 | $\dots$  | Zeros           | Zeros           | Zeros           |
| $x_{1,1}^{(3)}$ |                 |                 |                 | $\dots$  | Zeros           | Zeros           | Zeros           |
| $\vdots$        | $\vdots$        | $\vdots$        | $\vdots$        | $\ddots$ | $\vdots$        | $\vdots$        | $\vdots$        |
| $x_{n,m}^{(1)}$ | Zeros           | Zeros           | Zeros           | $\dots$  |                 |                 |                 |
| $x_{n,m}^{(2)}$ | Zeros           | Zeros           | Zeros           | $\dots$  |                 | $W$             |                 |
| $x_{n,m}^{(3)}$ | Zeros           | Zeros           | Zeros           | $\dots$  |                 |                 |                 |
|                 |                 |                 |                 |          |                 |                 |                 |
|                 |                 |                 |                 |          |                 |                 |                 |
$x = f(z) = Wz$
=> $J_f = W$
---

容易发现：

$$
\det(J_{G^{-1}}) = \det(W)^{n \times m}
$$


---
## 更多

Parallel WaveNet
WaveGlow
