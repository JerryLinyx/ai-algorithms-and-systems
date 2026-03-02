

## Conditional generation 
Classifier based guidance 训练前分，麻烦
Fixed Guidance  训练引入 prompt，diversity 差，遵循指令
Classifier Free 调节 guidance 强度




### Classifier-based Guidance

以Diffusion为例：$P(x_{t-1}|x_t)$ -> Guided: $P(x_{t-1}|x_t,y)$
  
 $$
 \begin{aligned}
 P(x_{t-1}\mid x_t,y) 
 &= \frac{P(x_{t-1}\mid x_t) P(y\mid x_{t-1}, x_t)}{P(y\mid x_{t})} \\
 &= \frac{P(x_{t-1}\mid x_t) P(y\mid x_{t-1})}{P(y\mid x_{t})} \\
 &= P(x_{t-1}\mid x_t) e^{ \log{P(y\mid x_{t-1})} - \log{P(y\mid x_{t})} }\\
 &\approx P(x_{t-1}\mid x_t) e^{ (x_{t-1} - x_t)\cdot\nabla_{x_t} \log{P(y \mid x_t)}} \\
 \end{aligned}
 $$
 $$ P(x_{t-1}\mid x_t) = N(x_{t-1}; \mu(x_t), \sigma_t^2 I ) \propto e^{{- \Vert {x_{t-1} - \mu(x_t)} \Vert^2} / {2\sigma_t^2}}$$
 $$\Rightarrow  P(x_{t-1}\mid x_t,y) \propto e^{\frac{- \Vert {x_{t-1} - \mu(x_t)} \Vert^2}{2\sigma_t^2} + (x_{t-1} - x_t)\cdot\nabla_{x_t} \log{P(y \mid x_t)} } 
 \propto e^{{- \Vert {x_{t-1} - \mu(x_t) - \sigma_t^2\cdot\nabla_{x_t} \log{P(y \mid x_t)}} \Vert^2}/{2\sigma_t^2}}$$
 $$\Rightarrow x_{t-1} = \mu(x_t) + \sigma_t^2 \nabla_{x_t}\log{P(y \mid x_t)} + \sigma_t\varepsilon$$
### Fixed Guidance

以flow-matching为例：
$$\mathcal{L}_{CFM}^{guided}(\theta ; y) = \mathbb{E}_{(x)}\Vert {\mu_t^\theta(x \mid y) - \mu_t(x | x_1)} \Vert^2$$

### Classifier-free Guidance (SOTA)


Hierarchical/Cascaded Diffusion
https://arxiv.org/abs/2106.15282

LDM
https://arxiv.org/abs/2112.10752 

Stable diffusion
Unet 

ViT -> DiT

ViT patch -> token

Diffusion Transformer

