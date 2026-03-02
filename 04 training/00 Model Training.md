pre-training 预训练使用大量无标注数据训练，得到基模

Encoder only BERT
Decoder only GPT
Encoder-Decoder T5

Autoencoding models 
采用 Masked Language Modeling (MLM)
Objective: Reconstruction（denoising）
Good use cases：
• Sentiment Analysis
• Named entity recognition
• Word classification
Example models:
• BERT
• ROBERTA

Autoregressive Models
采用 Causal Language Modeling (CLM)
Objective: Predict Next Token
Good use cases:
• Text generation
• Other emergent behavior
	• Depends on model size
Example models:
• GPT
• BLOOM

Sequence-to-sequence models
Span Corruption
Objective: Reconstruct span
Good use cases:
• Translation
• Text Summarization
• Question answering
Example models
• T5 （Text-to-Text Transfer Transformer
• BART

为什么选择 Autoregressive Models？
scalable




## Quantization 量化

| Format   | Bits | Exponent | Fraction | Memory needed to store one value |
|----------|------|----------|----------|----------------------------------|
| FP32     | 32   | 8        | 23       | 4 bytes                          |
| FP16     | 16   | 5        | 10       | 2 bytes                          |
| BFLOAT16 | 16   | 8        | 7        | 2 bytes                          |
| INT8     | 8    | —        | 7        | 1 byte                           |

- Reduce required memory to store and train models.  
- Statistically projects 32-bit floating point numbers into lower precision spaces.  
- Quantization-aware training (QAT) learns the quantization scaling factors during training.  
- BFLOAT16 is a popular choice.

## AMP（Automatic Mixed Precision）自动混合精度

**AMP 是一种训练加速技术**，通过在训练过程中**自动混合使用 FP16 / BF16 和 FP32**，在**几乎不损失模型精度的前提下**，显著提升训练速度并降低显存占用。

Q: 为什么 AMP 能提升性能？

>1. FP16 / BF16 计算更快
>现代 GPU（如 V100 / A100 / H100）对低精度运算有 Tensor Core 硬件加速，矩阵乘和卷积的吞吐量明显高于 FP32。
>
>2. 显存占用更低
>FP16 仅占 FP32 一半显存，使得同显存可使用更大的 batch size 或在相同 batch 下训练更大的模型
>
>3. 计算与访存效率更高
>更小的数据类型减少显存带宽压力，在大模型或大 batch 训练中尤为明显。

Q: AMP 为什么不会明显降低精度？

>1. 选择性的混合精度：
>- 大部分算子（如卷积、GEMM）使用 FP16 / BF16
>- 数值敏感操作（如 loss 计算、LayerNorm 归一化、Softmax、梯度累积）仍然使用 FP32
>
>1. 通过 Loss Scaling 解决 FP16 梯度下溢问题
>- 在反向传播前，先将 Loss 乘以一个很大的系数（比如 $2^{16}$）避免梯度变成 0。根据链式法则，所有的梯度都会被等比例放大，从而跳进 `float16` 能够表达的数值区间。更新参数前再缩放回来。
	
	
AMP 的 forward 阶段本身是安全的，真正的数值风险发生在 backward 时的 FP16 梯度。虽然权重梯度最终是在 FP32 中计算的，但如果 activation 梯度在 FP16 中已经 underflow 成 0，那么再精确的 FP32 计算也无济于事。
GradScaler 的作用正是通过放大 loss，防止这一关键梯度在反向传播早期消失。

### EMA 

模型使用正常的权重（受 AMP 缩放影响的梯度更新）进行反向传播
**影子权重（Shadow Weights）：** 另外维护一套权重的“影子副本”，它不是通过梯度直接更新，而是通过 EMA 公式缓慢跟随：

$$\theta_{EMA} = \beta \cdot \theta_{EMA} + (1 - \beta) \cdot \theta_{current}$$

其中 $\beta$ 通常设为 $0.999$ 或 $0.9999$。

提升泛化能力与稳定性
AMP 虽然快，但由于舍入误差，单步更新可能存在较大的随机性。EMA 通过累积历史权重的平均值，相当于在参数空间做了一种“平滑”，能够过滤掉 AMP 带来的部分数值噪音，使模型最终收敛在更平坦、更稳定的局部极小值点。

解决测试集表现波动
在混合精度训练中，验证集（Validation）的指标有时会剧烈抖动。使用 EMA 权重进行预测（Inference）通常能比直接使用最新训练权重获得更高的精度和更稳健的表现。

协同解决数值溢出
AMP 使用 GradScaler 来防止梯度消失/溢出。配合 EMA 使用时，EMA 权重作为一种“慢更新”机制，能有效防止某些由于梯度骤变（Spike）导致的模型崩溃。

显存开销更大 
EMA 需要为模型参数维护一份额外的 float32 副本。在 AMP 节省显存的同时，EMA 会消耗一部分显存（通常是模型大小的一倍）。




## 梯度检查

## 梯度累积

## 编译
### 动态图 vs. 静态图
动态图 (Dynamic Graph / Imperative)：
如 PyTorch (默认模式)。
“所见即所得”。代码运行到哪，计算图就建到哪。可以使用 Python 的 `if`、`for` 循环，方便调试（像写普通 Python 一样打断点）。
但是框架无法提前知道整个计算流程，难以进行全局优化，运行开销（Overhead）较大。

静态图 (Static Graph / Declarative)：
如 TensorFlow 1.x、TensorRT。
先定义完整的计算图（像画电路图），然后送入执行引擎运行。框架可以看到全局，能做算子融合（把多个操作合并成一个）、显存预分配，速度极快。
但是调试极其痛苦，不支持原生 Python 逻辑。

### JIT vs. AOT 
这两个术语描述的是什么时候把代码变成机器码：

JIT (Just-In-Time，即时编译)：
在运行程序时进行编译。
逻辑： 第一次运行某段代码时，编译器把它翻译成优化后的指令并缓存。PyTorch 的 torch.jit.script 就是这种。 兼顾了开发灵活性和运行速度。

AOT (Ahead-Of-Time，提前编译)：
在运行程序前（通常是部署阶段）完成编译。
把模型彻底脱离 Python 环境，编译成一个二进制文件（如 C++）。 追求极致的推理性能，常用于手机端、车载芯片等对延迟敏感的场景。

### Torch.dynamo (图捕获器)
PyTorch 2.0 的核心逻辑。它通过拦截 Python 的 Bytecode（字节码），尝试自动地将 Python 代码转化为“子图”。如果代码里有处理不了的复杂 Python 逻辑，它会把能优化的部分切出来，剩下的留给 Python 原生执行（Graph Break），不会像以前的 `torch.jit` 那样直接报错。
### Torch.fx (图表示/转换)
这是一个用于变换计算图的工具包，把模型变成一个可以编程修改的结构。
如：可以写一个脚本，自动搜索模型里所有的 `Conv2d` 和 `ReLU` 并把它们融合成一个操作。
### Inductor 后端编译器
接收 Dynamo 捕获并由 FX 优化后的图，然后为硬件（如 NVIDIA GPU）生成高性能的代码（通常是 Triton 或 C++）。
