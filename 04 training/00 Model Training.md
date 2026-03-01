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




## Quantization: Summary

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

训练过程：前向、反向传播、算子写法

动态图、静态图
torch.fx、torch.dynamo、Inductor
JIT、AOT

## 优化器（Optimizer）


苏剑林. (Nov. 14, 2024). 《当Batch Size增大时，学习率该如何随之变化？ 》[Blog post]. Retrieved from [https://kexue.fm/archives/10542](https://kexue.fm/archives/10542)

苏剑林. (Sep. 01, 2025). 《重新思考学习率与Batch Size（一）：现状 》[Blog post]. Retrieved from [https://kexue.fm/archives/11260](https://kexue.fm/archives/11260)

## 混合精度

FP32
 
FP16 + FP32
优化器使用FP32，权重前向/反向时使用FP16 


BF16 + FP32
FP8





## 梯度检查

## 梯度累积