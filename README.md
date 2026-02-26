# AI 学习笔记

本仓库整理了鄙人关于生成式人工智能（Generative AI）相关算法与系统工程方向的学习笔记。

---

## HPC 高性能计算

## Data 数据

## Algorithm 算法

## Training 训练

## Inference 推理



---




## 学习资源

- [CS自学指南](https://csdiy.wiki/)
  
- [【必看】历史技术文章导航 - 猛猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/654910335)
  
- [ShusenWang - B站](https://space.bilibili.com/1369507485)
  
- [楚国刮大风 - B站](https://space.bilibili.com/20942052)
  
- [ZOMI酱 - B站](https://space.bilibili.com/517221395)
- https://github.com/Infrasys-AI/AIInfra
  
- [Chayenne Zhao - 知乎](https://www.zhihu.com/people/alan-70-79-23)
- https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial
  
- https://github.com/zjhellofss/KuiperInfer
  
- https://github.com/jinbooooom/ai-infra-hpc
  
- [进击的Bruce - 知乎](https://www.zhihu.com/people/void-73-73)
  
- [叫我Alonzo就好了 - 知乎](https://www.zhihu.com/people/liu-chang-82-34-78)

- https://se7en.mintlify.app/

## 思维导图 Mind Map by Mermaid

https://mermaid.js.org/intro/syntax-reference.html

```mermaid
flowchart LR

%% ======================
%% 顶层
%% ======================

AI["AI Infra"]

AI --> HPC["高性能计算"]
AI --> Training["训练框架"]
AI --> Inference["推理框架"]
AI --> Compiler["AI编译器/其他"]



%% ======================
%% Profiling
%% ======================
HPC --> NVidia
NVidia --> CUDA["CUDA, Cutlass, Ciblas, Cudnn"]
NVidia --> Profiler
Profiler --> NCU["Nsight Compute(NCU) & Nsight Systems(NSYS)"]


%% ======================
%% NPU 技术栈
%% ======================

HPC --> NPU
NPU --> Others["寒武纪，昆仑芯，燧原，摩尔线程，地平线，平头哥..."] 
NPU --> Huawei["华为"] 
Huawei --> Ascend["昇腾: Ascend C, 910, profile, TBE"] 
Huawei --> Mindspore["昇思: MindSpore, mlir"] 


%% ======================
%% 推理层
%% ======================
Inference --> Netron

Inference --> InferenceCore["AI 推理"]

InferenceCore --> TNN["TNN (Tencent)"]
InferenceCore --> MNN["MNN (Alibaba)"]
InferenceCore --> TensorRT["TensorRT (NVidia)"]
InferenceCore --> CANN["CANN (Huawei)"]


Inference --> LLMInfer["大模型推理"]
LLMInfer --> TensorRT-LLM
LLMInfer --> Llama.cpp
LLMInfer --> vLLM
LLMInfer --> SGLang


SGLang --> DSLCache["DSL / 流式返回 / 优先级调度 / KV 缓存管理"]
vLLM --> PagedAttention["Paged Attention / ORCA continuous batch"]
TensorRT-LLM --> Quant["量化 (SmoothQuant / AWQ)"]


%% ======================
%% 编译器栈
%% ======================

Compiler --> TVM
Compiler --> MLIR
Compiler --> IREE
Compiler --> TPUMLIR
Compiler --> XLA
Compiler --> Triton
Compiler --> LLVM




%% ======================
%% 训练框架
%% ======================

Training --> Pytorch
Training --> Tensorflow
Training --> Paddle
Training --> MindSpore
Training --> Oneflow
Training --> Deepspeed
Training --> Megatron-LM
Training --> verl
Training --> unsloth

```

```mermaid
flowchart TD
%% ======================
%% CUDA 编译链
%% ======================

CuFile[".cu (CUDA C++)"] --> NVCC["NVCC"]
NVCC -- .c (Host C/C++ code) --> HostCompiler["Host C/C++ Compiler"]
NVCC -- .ptx (PTX(Virtual) ISA code) --> JIT["Device JIT Compiler"]

HostCompiler -- Host Assembly (e.g. x86, Power, ARM) --> CPU
JIT -- Device Assembly (e.g. SASS) --> GPU
```