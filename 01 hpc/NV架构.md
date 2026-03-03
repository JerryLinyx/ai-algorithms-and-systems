芯片架构发展
1. 消费级 GeForce 系列
   RTX 50XX 系列（Blackwell）
   RTX 40XX 系列（Ada Lovelace）
   RTX 30XX 系列（Ampere）
   RTX 20XX 系列（Turing）
   GTX 16/10 系列（Turing / Pascal）
2. 专业工作站级
   RTX A50XX 系列（Ada / Ampere）
   RTX A60XX 系列（Ada / Ampere）
3. 数据中心级
   Rubin / Vera Rubin 平台（Rubin，2026 H2 起）
   Tesla H100 系列（Hopper）
   Tesla A100 系列（Ampere）
   Tesla V100 系列（Volta）
   Tesla L40 系列（Ada Lovelace）

为什么 Tensor Core 发展到现在这样？
Volta → Turing → Ampere → Hopper → Blackwell → Rubin

The number of cores in a GPU is still somewhat reliably scaling over time (although more slowly than Moore's law would dictate). 
2015ﾠMaxwellﾠGTX 980 Ti (2,816)
2017ﾠPascalﾠGTX 1080 Ti (3,584)
2018ﾠTuringﾠRTX 2080 Ti (4,352)
2020ﾠAmpereﾠRTX 3090 (10,496)
2022ﾠAda LovelaceﾠRTX 4090 (16,384)
2025ﾠBlackwellﾠRTX 5090 (21,760)


Rubin（Vera Rubin 平台，面向数据中心，2026）
- 定位：继 Blackwell 之后的下一代数据中心 AI 平台，强调“极限协同设计（extreme codesign）”，不仅是 GPU，而是整套机架级系统。
- 6 个核心组件（官方平台叙述）：Vera CPU + Rubin GPU + NVLink 6 Switch + ConnectX-9 SuperNIC + BlueField-4 DPU + Spectrum-6 Ethernet。
- 关键指标（平台级口径）：
  - 推理 token 成本：相对 Blackwell 平台最高可降低到 1/10。
  - MoE 训练：相对 Blackwell 平台可用 4x 更少 GPU 完成训练（同等目标下）。
- 关键指标（单芯片/互联口径）：
  - Rubin GPU：50 PFLOPS（NVFP4 inference）。
  - NVLink 6：每 GPU 3.6 TB/s；Vera Rubin NVL72 机架级 260 TB/s。
  - Vera CPU：88 个自研 Olympus ARM 核心（Armv9.2），NVLink-C2C 互联。
- 常见形态（官方命名）：Vera Rubin NVL72（机架级）与 HGX Rubin NVL8（板级，8 GPU）。

Rubin CPX（长上下文推理取向，2025-09 公布）
- Rubin CPX GPU：面向百万 token 级“massive-context”推理；最高 30 PFLOPS（NVFP4），配 128GB GDDR7。
- Vera Rubin NVL144 CPX（机架级）：8 EFLOPS AI compute、100TB fast memory、1.7 PB/s memory bandwidth（官方口径）；Rubin CPX 预计 2026 年底可用。
