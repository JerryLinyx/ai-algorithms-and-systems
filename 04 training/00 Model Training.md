训练过程：前向、反向传播、算子写法

动态图、静态图
torch.fx、torch.dynamo、Inductor
JIT、AOT

## 优化器（Optimizer）




## 混合精度

FP32
 
FP16 + FP32
优化器使用FP32，权重前向/反向时使用FP16 


BF16 + FP32
FP8

## 显存占用分析
Model States 模型本身相关且必须存储的参数：
- Parameters：模型参数
- Gradients：模型梯度
- Optimizer States：Adam中 momentum和variance 

各种并行主要优化的是 model states

Residual States 非模型必须，训练过程中产生的参数：
- Activation：激活值
- Temporary Buffers：临时存储，如算子的中间变量
- Unusable Fragmented Memory：碎片化存储空间



## 梯度检查

## 梯度累积