
## Bias 和 Variance 辨析
- Bias (偏差): 偏差衡量的是模型预测的期望值与真实值之间的差距。它反映了模型本身的拟合能力或者说对问题的假设是否准确。
- Variance (方差): 方差衡量的是**模型在不同训练集上给出的预测值的波动程度**。它反映了模型对训练数据波动的敏感性。
<div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
  <img src="assets/00%20Deep%20Learning/file-20260301171229953.png" width="300" />
  <img src="assets/00%20Deep%20Learning/file-20260301171714039.png" width="300" />
</div>
Bias 和 Variance 是误差分解中的两个核心来源，通常存在此消彼长的关系：
- 高 Bias、低 Variance: 往往对应欠拟合 (underfitting)。
- 低 Bias、高 Variance: 往往对应过拟合 (overfitting)。

> Overfitting: model performs well on training data but does not
generalize well to unseen data (test data)
> Underfitting: model is not complex enough to capture pattern in the
training data well and therefore suﬀers from low performance on
unseen data

实践中的平衡思路：
- 当模型欠拟合 (高 Bias) 时：提高模型容量、增加有效特征、减弱正则化。
- 当模型过拟合 (高 Variance) 时：增加数据量、加强正则化、使用数据增强或早停。


## Regularization

Regularization（正则化）本质上就是一种**牺牲一点点 Bias（故意让模型拟合得不那么完美）来大幅度降低 Variance（让模型更稳定）** 的策略。它的核心目标是提升模型在未知数据上的泛化能力（Generalization）。

最经典的是针对权重参数 $w$ 的 L1 和 L2 正则化：
- **L1 正则化 (Lasso Regression)** 
	- 在损失函数后加上所有权重绝对值的和，即 $\lambda \sum |w_i|$。
    - **特点：** L1 会促使模型中的很多权重变成绝对的 $0$。因此，它自带**特征选择（Feature Selection）** 的功能，能够产出一个“稀疏（Sparse）”的模型，非常适合高维但只有少数特征起作用的场景。
- **L2 正则化 (Ridge Regression / Weight Decay)** 
	- 在损失函数后加上所有权重的平方和，即 $\lambda \sum w_i^2$。
    - **特点：** L2 会把模型的所有权重都往 $0$ 的方向拉，但很少会让它们真正变成 $0$。它倾向于让所有特征都发挥一点点作用，避免某个单一特征的权重过大（也就是让权重分布更平滑），这在训练深度神经网络时极为常用（通常被称为 Weight Decay）。

除了直接惩罚权重，在构建复杂的神经网络（如LLM）时，还会使用一些架构或训练流程上的正则化技巧：
- **Dropout：** 在训练过程中，每次随机“丢弃”（将激活值设为零）一部分神经元。这迫使网络不能过度依赖某几个特定的神经元，从而增强鲁棒性。
- **Early Stopping（早停）：** 监控验证集（Validation Set）的误差。当发现训练集误差还在下降，但验证集误差开始上升（说明开始过拟合了）时，立刻停止训练。
- **Data Augmentation（数据增强）：** 通过对训练数据进行微小的改变（如图像翻转、文本加噪）来人为扩大训练集，数据越多，模型越不容易过拟合。


## Performance Metrics
• Algorithmic performance: accuracy, precision, recall, F1-score, ROC,
• System performance: training bme, inference bme, training cost, memory requirement, training eﬃciency


### Confusion Matrix (混淆矩阵)

二分类任务里常用混淆矩阵来理解模型在“正/负”上的错误类型：

| 实际 \\ 预测        |     预测为正 (Positive) |     预测为负 (Negative) |
| --------------- | ------------------: | ------------------: |
| 实际为正 (Positive) |  TP (True Positive) | FN (False Negative) |
| 实际为负 (Negative) | FP (False Positive) |  TN (True Negative) |

常见指标：
### Accuracy 准确率
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
在正负样本极度不平衡的数据集中（比如罕见病检测），这个指标具有欺骗性

### Precision 精确率
$$Precision = \frac{TP}{TP + FP}$$
Precision 精确率 = 标出来的有多少是真的 → 注重“查准”。
- 假发现率 (False discovery rate) = $1 - Precision$
### Recall 召回率
$$Recall = \frac{TP}{TP + FN} $$

Recall 召回率 = 真实有多少被查到了 → 注重“查全”
- 召回率在医学和统计学中也常被称为 **灵敏度 (Sensitivity)** 或 **真阳性率 (True positive rate, TPR)**

两者往往相互影响：
> 想召回更多，就会多标一些，但可能会带来更多误报 → 精确率下降
> 想保持精确率，就严格筛选，但可能漏掉一些 → 召回率下降

所以常见用 **F1-Score** 衡量两者平衡：
> Harmonic mean of precision and recall; measure of classifier accuracy
### F1 Score

**F1 score:** Harmonic mean of precision and recall; measure of classifier accuracy
$$F_1 = \frac{2}{\text{recall}^{-1} + \text{precision}^{-1}} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \frac{tp}{tp + \frac{1}{2}(fp + fn)}$$
**$F_\beta$ score:** $\beta = 1$ is F1 score; recall is considered $\beta$ times as important as precision
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$
### Specificity 特异度

$$True negative rate = \frac{TN}{TN + FP}$$

- **含义：** 在所有真实的“负”样本中，模型成功认出了多少。也可以称为 **真阴性率 (True negative rate, TNR)**。
	- 假阳性率 (False positive rate, FPR) = $1 - Specificity$。

### Balanced accuracy 平衡准确率
$$Balanced accuracy = \frac{Sensitivity + Specificity}{2}$$
它综合考虑了混淆矩阵中的所有条目（同时兼顾了正类和负类的预测表现），值介于 $0$（最差的分类器）和 $1$（最好的分类器）之间。

## 训练流程
### Forward Phase
- compute the activations of the hidden units based on the current value of weights
- calculate output
- calculate loss function


### Backward Phase
- compute partial derivative of loss function w.r.t. all the weights;
- use _backpropagation algorithm_ to calculate the partial derivatives recursively
- backpropagation changes the weights (and biases) in a network to decrease the loss

**Q：为什么每一次 backward 之前都要把梯度置零？**

> 在 PyTorch 中，梯度默认是**累加**而不是自动覆盖的。
> 每次调用 loss.backward()，新计算得到的梯度都会加到参数的 .grad 上。如果不在每次反向传播前清零，梯度就会跨 iteration 叠加，导致参数更新不再只基于当前 batch，从而引入错误的优化方向。
> 因此，在标准训练流程中，我们会在每次 backward 前调用 optimizer.zero_grad()，确保当前参数更新只使用本轮 forward 对应的梯度。
> 这种设计也是为了支持像**梯度累积**这样的场景，例如在显存受限时，用多个小 batch 累加梯度来等效一个大 batch。

**Q：with torch.no_grad() 的作用是什么？**

> 在 PyTorch 中，默认情况下张量运算会被 autograd 跟踪，用于反向传播。
> 使用 torch.no_grad() 可以显式告诉框架当前不需要梯度计算，这不仅避免了无用的计算图构建，还能显著降低中间激活的显存开销，因此在 inference、validation、test 阶段是标准做法。


### Gradient Descent
- Update the weights using gradient descent

### Stochastic Gradient Descent (SGD)

$$\overline{W} \Leftarrow \overline{W} - \alpha \frac{\partial L_i}{\partial \overline{W}}$$

- Loss is calculated using one training data at each weight update.
- Stochastic gradient descent is only a randomized approximation of the true loss function.

## Hyperparameters
### Network architecture
number of hidden layers, number of hidden units per layer 
### Activation functions
### Weight initializer
### Learning rate
### Batch size

Batch size 指的是**一次前向传播和反向传播中同时参与计算的样本数量**。在训练过程中，模型会先对一个 batch 的数据进行 forward，计算该 batch 的平均（或总）loss，然后基于这个 loss 做一次 backward，得到梯度并更新参数。

大 Batch Size 可以让梯度估计更准确，允许使用更大的学习率来加快收敛。但缺点是容易让模型陷入“尖锐的局部最优解（Sharp minima）”，导致模型在未知数据上表现糟糕（泛化能力差）。
而小批量数据带来的梯度计算带有一定的随机噪声。这种噪声不但不是坏事，反而起到了隐式正则化（Implicit regularization）的作用，能把模型“震出”那些尖锐的局部最优坑，帮助它找到更平缓、泛化能力更强的“平坦最优解（Flat minima）”

苏剑林. (Nov. 14, 2024). 《当Batch Size增大时，学习率该如何随之变化？ 》[Blog post]. Retrieved from [https://kexue.fm/archives/10542](https://kexue.fm/archives/10542)
苏剑林. (Sep. 01, 2025). 《重新思考学习率与Batch Size（一）：现状 》[Blog post]. Retrieved from [https://kexue.fm/archives/11260](https://kexue.fm/archives/11260)

### Momentum
### Optimizer

## Scaling
[Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-SLI, NVSwitch and GPUDirect](https://arxiv.org/pdf/1903.04611)

### Verbcal scaling-up in a single node 垂直扩展 / 向上
- 在单台物理机（Node）内增加 GPU 的数量。
- 单机内多卡的通信瓶颈通常在于主板总线。
- NVLink 技术提供了比传统 PCIe 高得多的带宽，使得单机内的几块 GPU 之间能够极快地共享数据和同步梯度。
- **X-Bus** 是 IBM 的总线技术，专门用来在**同一个主板上**连接这两个 CPU, 类似 Intel 处理器之间的 **QPI / UPI** 总线，或者是 AMD 处理器之间的 Infinity Fabric
![](assets/00%20Deep%20Learning/file-20260301231138041.png)
![](assets/00%20Deep%20Learning/file-20260301231300692.png)
### Horizontal scaling-out across multiple nodes 水平扩展 / 向外
- 当单台机器（即使装了 8 张或 16 张卡）的算力或显存依然不够时，就需要通过高速网络（如 InfiniBand）把成百上千台机器连接成一个集群。
- e.g. GPU accelerated supercomputers like Summit and Sierra from US Department of Energy
### High Performance Networking


SMP: symmetric multiprocessing
- 主板上的所有 CPU 核心地位平等。它们共享同一片物理内存（RAM）、同一个操作系统以及所有的 I/O 总线。
- QPI: Quick-path interconnect (早期 25.6 GB/s) 
	- CPU 内部或 CPU 之间的高速点对点总线

PCIe: Peripheral Component Interconnect express
- 连接 CPU 与外部硬件设备（显卡 GPU、高速固态硬盘 NVMe SSD、万兆网卡甚至 InfiniBand 网卡）的高速串行扩展总线标准


Large Scale Parallel and Deep Learning applications needs:
- High Bandwidth (高带宽)
- Low Latency (低延迟)


| **Network technology** | **Bandwidth [Gb/s]** | **Latency [us]** |
| ---------------------- | -------------------- | ---------------- |
| **10GigE**             | 10                   | 4                |
| **40GigE**             | 40                   | 4                |
| **IB EDR**             | 100                  | 1                |
| **NVLink**             | > 400                | 0.1-0.2          |

Ethernet is not enough 
- 虽然万兆（10GigE）或四万兆（40GigE）以太网在日常服务器中已经很快了，但它的**延迟（Latency）**通常在 4 微秒左右，而且依赖 CPU 处理庞大的 TCP/IP 协议栈，这会消耗极大的计算资源并带来不可预测的延迟抖动

Infiniband (IB) is widely adopted
- 相比以太网，IB 网络（如表中的 EDR 版本）不仅带宽更高，更关键的是它的延迟极低（约 1 微秒），并且支持 **RDMA（远程直接内存访问）** 技术。这意味着一台机器的网卡可以直接读写另一台机器的内存（或显存），完全绕过 CPU 和操作系统的内核，极大地提升了跨节点（Scale-out）通信的效率

Custom Networks are the best
- NVLink 的带宽远超普通的 IB 网络，且延迟极低（0.1-0.2 微秒）。这就是为什么在单台物理机内部（Scale-up），NVIDIA 坚持要用自家的 NVLink 将 GPU 互联，而不是走传统的 PCIe 或普通网络

|**代数 / 世代**|**PCIe (x16 双向总带宽)**|**对应的 NVIDIA 架构**|**NVLink (单卡双向总带宽)**|**性能差距倍数**|
|---|---|---|---|---|
|**第 1 代** (2016)|PCIe 3.0: 32 GB/s|**Pascal (P100)**|160 GB/s|5 倍|
|**第 2 代** (2017)|PCIe 3.0: 32 GB/s|**Volta (V100)**|300 GB/s|9.3 倍|
|**第 3 代** (2020)|PCIe 4.0: 64 GB/s|**Ampere (A100)**|600 GB/s|9.3 倍|
|**第 4 代** (2022)|PCIe 5.0: 128 GB/s|**Hopper (H100)**|900 GB/s|7 倍|
|**第 5 代** (2024)|PCIe 6.0: 256 GB/s|**Blackwell (B200)**|1.8 TB/s (1800 GB/s)|7 倍|
|**第 6 代** (2026)*|PCIe 7.0: 512 GB/s|**Rubin**|3.6 TB/s (3600 GB/s)|7 倍|