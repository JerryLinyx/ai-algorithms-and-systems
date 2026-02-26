

## 大模型分布式训练

大模型时代，一家公司的大模型能力通常与其单位时间迭代次数正相关。为此，我们需要优化训练耗时。训练数据规模，单步计算量和计算速率基本决定了训练耗时，其中的可变因素是计算速率，也是我们优化的目标。

• 深度学习训练耗时：

$$
\text{训练耗时}
=

\underbrace{\text{训练数据规模} \times \text{单步计算量}}_{\text{模型相关，相对固定}}
/
\underbrace{\text{计算速率}}_{\text{可变因素}}

$$
计算速率又由三个因素决定：单设备计算速率，设备数，以及多设备并行效率。

• 计算速率：

$$
\text{计算速率}
=
\underbrace{\text{单设备计算速率}}_{
\begin{array}{c}
\text{混合精度} \\
\text{算子融合} \\
\text{梯度累加}
\end{array}}
\times
\underbrace{\text{设备数}}_{
\begin{array}{c}
\text{服务器架构} \\
\text{通信拓扑优化}
\end{array}}
\times
\underbrace{\text{多设备并行效率（加速比）}}_{
\begin{array}{c}
\text{数据并行} \\
\text{模型并行} \\
\text{流水并行}
\end{array}}
$$
为此，在现有的AI框架（如PyTorch、TensorFlow、Caffe、JAX、MindSpore ...）之上，我们还需要一层大模型的使能层，特别是大模型的训练和推理框架。常见的分布式训练框架有 DeepSpeed、Megatron-LM、Colossal-AI 以及 HuggingFace 的 Accelerate 库。

分布式训练的核心功能：
1. 部署和训练以Transformer为核心的大模型
2. 提供数据/模型/流水并行 - 多维并行
3. 以集合通信和参数服务器（Parameter Server）方式进行资源整合

首先来看 DeepSpeed，这是由微软开源的一个深度学习优化库，旨在提高大模型训练效率和扩展性。DeepSpeed 主要使用了数据并行（Data Parallelism, 通过 ZeRO）、模型并行（PP）、梯度累积、动态缩放、混合精度等训练加速手段。此外，该框架还支持分布式训练管理，内存优化和模型压缩等，协助管理和优化大模型训练任务。
再看 Megatron-LM ，这是 NVIDIA 开发的一个深度学习框架，专门用于大模型的分布式训练。它通过极致的算子融合，以及 3D 并行 —— 数据并行、张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）的组合，将单卡的计算效率（MFU/TFLOPs）推向极限。Megatron-LM 提供了强大的tokenizer以及数据预处理与管理工具。
DeepSpeed更多被用于中等规模集群（数十到数百卡）预训练以及下游任务微调环节，超大规模集群（千卡、万卡级）更多采用 NVIDIA 的 Megatron-LM，并随后取长补短，发展为 Megatron-DeepSpeed，既保证了超大规模下的并行效率，又利用了 ZeRO 来降低显存占用。
值得关注的是，随着分布式技术的进一步发展，格局也在发生细微变化。NVIDIA 通过推出的模块化版本的 Megatron Core，试图拿回被 DeepSpeed 占领的中小规模市场；而 Meta 推出的 PyTorch 原生的 FSDP 也正成为 DeepSpeed 的强力竞争对手，尤其是在追求与原生 PyTorch 兼容的科研场景中。
除此之外，潞晨科技（HPC-AI Tech，由新加坡国立大学校级教授尤洋创建）的 Colossal-AI 框架提供了丰富的张量并行策略与配置案例；由 **OpenBMB** 开源社区（由清华大学NLP实验室和面壁智能发起）开发的 BMTrain 框架深度优化了DeepSpeed的并行策略，支持北京智源研究院的 Aquila 系列模型；华为也推出了在大模型领域对标 NVIDIA 生态的昇腾（Ascend）原生大模型加速库 MindSpeed。

## DeepSpeed

https://github.com/deepspeedai/DeepSpeed

训练方面为大模型提供ZeRO、3D-Parallelism、DeepSpeed-MoE、ZeRO-Infinity等特性。在推理，压缩以及 AI4Science 也有相应功能，但不及其在训练的影响力，不作赘述。
### ZeRO
 


## 6D 并行是什么？


## 学习资源：
[大模型是怎么训起来的？分布式并行框架介绍 #大模型 #分布式并行 #训练 - ZOMI酱](https://www.bilibili.com/video/BV1op421C7wp/)

