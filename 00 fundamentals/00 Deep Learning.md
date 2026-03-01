


### Bias 和 Variance 辨析
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
### Regularization

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


### Performance Metrics
• Algorithmic performance: accuracy, precision, recall, F1-score, ROC,
• System performance: training bme, inference bme, training cost, memory requirement, training eﬃciency


#### Confusion Matrix (混淆矩阵)

二分类任务里常用混淆矩阵来理解模型在“正/负”上的错误类型：

| 实际 \\ 预测 | 预测为正 (Positive) | 预测为负 (Negative) |
|---|---:|---:|
| 实际为正 (Positive) | TP (True Positive) | FN (False Negative) |
| 实际为负 (Negative) | FP (False Positive) | TN (True Negative) |

常见指标：

$$Precision = \frac{TP}{TP + FP} \quad,\quad Recall = \frac{TP}{TP + FN} \quad,\quad F1 = \frac{2PR}{P + R}$$


- TP (True Positive)：真正例 —— 正样本识别成正
- FN (False Negative)：假负例 —— 正样本识别成负
模型**敏感度（Sensitivity）**
- **Precision 精确率** = 标出来的有多少是真的 → 注重“查准”
- **Recall 召回率** = 真实有多少被查到了 → 注重“查全”
两者往往相互影响：
> 想召回更多，就会多标一些，但可能会带来更多误报 → 精确率下降
> 想保持精确率，就严格筛选，但可能漏掉一些 → 召回率下降

所以常见用 **F1-Score** 衡量两者平衡：
$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

**Q：batch size 指的是什么？**

> Batch size 指的是**一次前向传播和反向传播中同时参与计算的样本数量**。
> 在训练过程中，模型会先对一个 batch 的数据进行 forward，计算该 batch 的平均（或总）loss，然后基于这个 loss 做一次 backward，得到梯度并更新参数。

**Q：为什么每一次 backward 之前都要把梯度置零？**

> 在 PyTorch 中，梯度默认是**累加**而不是自动覆盖的。
> 每次调用 loss.backward()，新计算得到的梯度都会加到参数的 .grad 上。如果不在每次反向传播前清零，梯度就会跨 iteration 叠加，导致参数更新不再只基于当前 batch，从而引入错误的优化方向。
> 因此，在标准训练流程中，我们会在每次 backward 前调用 optimizer.zero_grad()，确保当前参数更新只使用本轮 forward 对应的梯度。
> 这种设计也是为了支持像**梯度累积**这样的场景，例如在显存受限时，用多个小 batch 累加梯度来等效一个大 batch。

**Q：with torch.no_grad() 的作用是什么？**

> 在 PyTorch 中，默认情况下张量运算会被 autograd 跟踪，用于反向传播。
> 使用 torch.no_grad() 可以显式告诉框架当前不需要梯度计算，这不仅避免了无用的计算图构建，还能显著降低中间激活的显存开销，因此在 inference、validation、test 阶段是标准做法。
