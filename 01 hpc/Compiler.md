
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
