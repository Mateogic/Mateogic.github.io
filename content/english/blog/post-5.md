---
title: "基于GDM的网络优化教程"
meta_title: ""
description: "this is meta description"
date: 2022-04-04T05:00:00Z
image: "/images/image-placeholder.png"
categories: ["Architecture"]
author: "Mateogic"
tags: ["silicon", "technology"]
draft: false
---
> [代码仓库](https://github.com/hongyangdu/gdmopt),[Diffusers-HuggingFace](https://huggingface.co/docs/diffusers/index)
- 文章脉络
![](/images/gdm/Pasted%20image%2020240622123219.png)
# 引言
## 背景
- 顾名思义，生成式人工智能(Generative Artificial Intelligence, GAI)可以生成新的数据，包括但不限于图像、文字、音频...该变革性能力赋予GAI突破传统AI应用的潜力。因此，GAI被广泛应用于各行各业，产生深远影响
- GAI是所有相关模型、技术组成的集合，它们各有所长，共同构成当前的GAI格局
	- Transformers：一种基于自注意力的序列到序列模型，支持并行计算，擅长语言理解，尤其适合处理自然语言生成任务
	- 生成式对抗网络(Generative Adversarial Networks, GANs)：由生成器和对抗器组成，二者相互作用共同促进性能提升
	- 变分自动编码器(Variational Autoencoders, VAEs)：先将输入数据压缩为潜在空间中的一组参数，然后利用这些参数生成与原始数据分布近似的新数据
	- 基于流的生成式模型(Flow-based Generative Models)：与传统批处理生成式模型一次性处理所有输入数据不同，它采用反向传播进行梯度计算提高了学习效率，因此具备实时输出性能，适合部署于移动边缘设备
	- 基于能量的生成式模型(Energy-based Generative Models)：尝试将DL、ML统一在能量模型的框架中，定义能量函数$E(x,y)$衡量$x$和$y$的匹配程度(能量值越小，匹配程度越高)。大多数概率模型均可看作特殊的基于能量的模型，得益于无需操作归一化后的概率分布，它具备更好的直观性、灵活性
	- 生成式扩散模型(Generative Diffusion Models, GDMs)：灵感源自热力学扩散，是一种基于马尔科夫链的生成式模型。由正向加噪和反向去噪两部分组成，前者通过计算向原始数据中不断添加噪声得到去噪模块的训练样本，后者通过神经网络不断去除输入数据中的噪声(先预测其中的<mark style="background: #BBFABBA6;">噪声</mark>再取差值)进而恢复原始数据
- GDM因其独特的数据生成方法和对复杂数据分布建模的能力脱颖而出，被广泛应用于图像、文本、音频以及分子、信号合成等新兴领域，其关键特性如下
	- 高质量数据生成能力：能够准确捕捉复杂数据分布
	- 灵活性：因为依赖于随机微分方程，GDM能够适应各种类型的数据和应用
	- 易于实现：相比GDM和VAE结构简单
## 动机
- 广泛的研究例证了扩散模型在处理传统生成领域之外的复杂问题时可提供有效的解决方案，启发我们将其用于智能网络优化
- 未来智能网络中可预见的应用如通感融合(ISAC)、语义通信(SemCom)、车联网(IoV)等涉及高维结构特征、非线性关系和复杂的决策过程。GDM善于捕捉其中的高维复杂结构，有效处理大量决策和优化问题,理解网络运营和优化中复杂权衡的细微差别
## 贡献
- 提供GDM应用的全面教程，详细介绍应用GDM解决动态无线环境中的复杂优化问题
- 提供研究案例证明GDM在新兴网络技术中的实用性和有效性
- 讨论GDM优化网络的未来研究和应用方向
# 利用GDM优化网络
## GDM的原理
### 从GAI到GDM
- 从生成式模型的生成器出发，其输入包括原始输入$x$和从随机分布中采样的$z$，它们经过生成器后输出$y$
	- 为方便采样，要求该随机分布足够简单，常用正态分布、均匀分布
	- 因为从随机分布中的采样结果不固定，导致生成器的输出同样具有随机性，服从复杂分布
	![](/images/gdm/Pasted%20image%2020240623204540.png)
- 设计输出结果服从概率分布的原因：训练数据中可能包含多种情况的数据样本，如果它们均被用于训练且要求输出结果单一，神经网络将会学到“多面玲珑”，这会导致错误的输出结果，如重影、模糊...因此，设计生成器可能输出一切可能的结果，这些结果服从复杂分布，<mark style="background: #BBFABBA6;">具体输出结果受到输入中随机分布的采样影响</mark>
- 考虑利用GDM生成维度为$256\times 256$的图像
![](/images/gdm/Pasted%20image%2020240623213316.png)
  1. 从正态分布中采样维度与目标图像同为$256\times 256$的噪声向量$x_{1000}$，将该向量与剩余去噪步数$1000$(表示当前噪声的严重程度)一同输入去噪模块
    - GDM从正态分布中采样对应GAI从随机分布中采样 ^7c60a6
  2. 去噪模块接收上述输入后，其内部结构中的关键组件<mark style="background: #BBFABBA6;">噪声预测器</mark>预测该输入图像中的噪声，再取差值间接地实现去噪效果得到$x_{999}$。这是因为预测噪声比直接预测去噪后的结果<mark style="background: #BBFABBA6;">更容易实现</mark>
  3. 将$x_{999}$与剩余去噪步数$999$一同输入下一级去噪模块，如此迭代最终得到目标图像$x_{0}$
![](/images/gdm/Pasted%20image%2020240623213151.png) ^da9208
- 噪声预测器训练
	- 训练目标：输入图像和剩余去噪步数，输出本次应该去除的噪声向量
	- 训练样本生成：原始图像+正态分布中随机采样的噪声=加噪图像，直至整张图像完全被噪声淹没，这就是所谓的正向加噪过程。记录该过程中的加噪次数、加噪向量本身以及加噪之后的图像，即得到噪声预测器的训练样本
	- 经过训练，噪声预测器根据第2次加噪后的图像和输入的剩余去噪次数2可以输出标准答案(第2次添加的噪声)
	![](/images/gdm/Pasted%20image%2020240623222106.png) ^3b8e67
- 类似地，考虑文本生成图像，将文本描述额外作为噪声预测器的输入指导它预测输入图像中应该去除的噪声。与之对应，训练过程中也要添加文本描述用于指导噪声生成
![](/images/gdm/Pasted%20image%2020240623223538.png)
![](/images/gdm/Pasted%20image%2020240623223546.png)
.
### 从GDM到GAI
- <mark style="background: #BBFABBA6;">重参数化</mark>是VAE中解决可微性问题的一种技术，因从分布中采样得到的隐含变量$z$具有随机性，故无法对其直接求导。重参数化通过将随机采样过程转换为确定性操作来解决此问题，具体来说分为以下两步
	1. 从固定分布(通常是标准正态分布)中采样一个辅助噪声$\epsilon$
	2. 通过可微变换将$\epsilon$映射到隐含变量$z$​
	- 由此，原本依赖于随机采样的模型输出变成依赖于确定性函数的输出，使得整个模型关于其参数可微，从而可利用标准的反向传播算法优化训练过程
	- 正态分布的重参数化采样：定义$y$服从正态分布$y\sim N(\mu,\sigma^2)$，$\epsilon$服从标准正态分布$\epsilon\sim N(0,1)$，变换$y=\sigma\epsilon+\mu\sim N(\mu,\sigma^2)$。由此推知，<mark style="background: #BBFABBA6;">标准正态分布经过线性变换后仍然服从正态分布</mark>。后续涉及$x_t=\sqrt{1-\beta_t}\times x_{t-1}+\sqrt{\beta_t}\times\epsilon_{t-1}$同样服从正态分布$x_t\sim N(\sqrt{1-\beta_t}\times x_{t-1},\beta_t)$
	![](/images/gdm/Pasted%20image%2020240624160740.png)
- 考虑简单的生成式模型，要求其生成数据与训练数据分布一致，因训练数据分布的复杂性，难以直接实现对训练数据的建模、采样。好在，我们可以将简单分布作为过渡，借助两种分布间的拟合特性和简单分布易于采样的优势实现该要求。事实上，GDM正是在原本复杂的训练数据上添加正态噪声得到简单分布，再利用简单分布采样训练噪声预测器，进而通过不断去噪实现该目标
![](/images/gdm/Pasted%20image%2020240624150907.png)
- 生成数据与训练数据分布一致的合理性猜测
	- 在训练数据上添加噪声得到的简单分布中包含原始数据，符合GAI从随机分布中采样且输出涵盖一切可能的结果的要求
	- 与傅里叶级数拟合复杂函数思想类似，简单的正态分布可以加权拟合复杂的观测数据分布
	- 根据中心极限定理，大量独立同分布的随机变量之和近似服从正态分布，也即正态分布中包含了所有可能的分布
	![](/images/gdm/v2-491dfd8467f994d8fe29d58076364863_1440w.png)
- 扩散去噪概率模型(Diffusion Denoising Probability Models, DDPM)是一种典型的GDM模型，以下从正向加噪和反向去噪两部分展开讨论其工作原理
![](/images/gdm/Pasted%20image%2020240624162331.png)
- 将其建模为包含$T$步的马尔科夫链，定义$x_0$为原始数据，所添加噪声均服从标准正态分布$\epsilon\sim N(0,\textbf{I})$，持续加噪最终得到$x_T$
	- 其中$\textbf{I}$为单位矩阵，表示各个维度具有相同的标准偏差
### 正向加噪阶段
- 相当于编码器：将观测数据分布映射到简单分布
- 定义加噪过程：不失一般性地将高斯噪声$\epsilon_{t-1}$加权添加到$x_{t-1}$得$x_t$，该过程可直观表示为：$$x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_{t-1}\tag{1}$$其中超参数$\beta_t\in(0,1)$为预定义的权重，通常它满足$\beta_1<...<\beta_t<...<\beta_T$，由此$x_t$随着$t$增大逐渐趋向于标准正态分布。将$\sqrt{1-\beta_t}\,x_{t-1}$作为均值，$\beta_t\textbf{I}$作为方差，通过条件概率分布描述该过程为：$$q(x_t|x_{t-1})=N(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t\textbf{I})\tag{2}$$进而，推导出正向加噪过程的联合概率密度：$$q(x_0,...,x_t,...,x_T)=q(x_{0:T})=q(x_0)\prod_{t=1}^{T}q(x_t|x_{t-1})\tag{3}$$联立条件概率公式$q(x_{0:T})=q(x_0)q(x_{1:T}|x_0)$得到后验概率密度：$$q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1})\tag{4}$$
- 为避免迭代采样的计算复杂度，定义$\alpha_t=1-\beta_t$改写式(1)得到：$$x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_{t-1}\tag{5}$$将$x_{t-1}$拆开代入有：$$\begin{align}x_t=&\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\,x_{t-2}+\sqrt{1-\alpha_{t-1}}\,\epsilon_{t-2})+\sqrt{1-\alpha_t}\epsilon_{t-1}\\=&\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\epsilon_{t-2}+\sqrt{1-\alpha_t}\epsilon_{t-1}\end{align}\tag{6}$$此处$\epsilon_{t-1}$和$\epsilon_{t-2}$是两个独立同标准正态分布的随机变量，由正态分布的再生性有：$$\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\epsilon_{t-2}+\sqrt{1-\alpha_t}\epsilon_{t-1}\sim N(0,((\sqrt{\alpha_t-\alpha_t\alpha_{t-1}})^2+(\sqrt{1-\alpha_t})^2)\textbf{I})\tag{7}$$据此合并式(6)的后两项整理得：$$x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\overline{\epsilon_{t-2}}\tag{8}$$可见该形式与(5)保持一致，可以理解为一次性加权添加$\epsilon_{t-2},\epsilon_{t-1}$两个噪声，同理合并从$x_0$到$x_T$添加的所有高斯噪声，利用数学归纳法易证：$$x_t=\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon,\,\epsilon\sim N(0,\textbf{I})\tag{9}$$其中$\overline{\alpha_t}=\prod_{j=1}^{t}\alpha_j$，给定超参数$\beta_t$(固定或服从一定规律)可预先计算$\alpha_t,\overline{\alpha_t}$，从而直接确定$x_t,t\in[1,T]$服从正态分布的参数：$$x_t\sim q(x_t|x_0)=N(x_t;\sqrt{\overline{\alpha_t}}\,x_0,(1-\overline{\alpha_t})\textbf{I})\tag{10}$$已知$1>\alpha_1>...>\alpha_t>...>\alpha_T>0$，当$T\rightarrow ∞$时$\overline{\alpha_T}=\prod_{j=1}^T\alpha_j\rightarrow 0$，此时公式(10)收敛到标准正态分布，也即$x_T\sim N(0,\textbf{I})$。至此，正向加噪目标实现
### 反向去噪阶段
- 相当于解码器：将简单分布映射到观测数据分布
- 去噪阶段的目标是学习上述加噪过程的逆过程。假如去噪过程可以像加噪过程一样写出条件概率密度表达式$q(x_{t-1}|x_t)$，根据$x_T$服从标准正态分布，我们可以从$N(0,\textbf{I})$中采样$x_T$，进而按照$q(x_{t-1}|x_t)$执行反向过程即可得到原始数据分布$q(x_0)$的一个样本$x_0$。然而，$q(x_{x-1}|x_t)$难以显式给出，因此需要使用参数模型$p_{\theta}$来估计$q(x_{x-1}|x_t)$，以上讨论建立的前提是原始数据分布$q(x_0)$未知。参数模型定义如下：$$p_{\theta}(x_{t-1}|x_t)=N(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t))\tag{11}$$联合概率密度：$$p_{\theta}(x_{0:T})=p(x_T)\prod_{t=1}^T{p_{\theta}(x_{t-1}|x_t)}\tag{12}$$
- 为训练噪声预测器，引入原始数据分布$q(x_0)$，根据贝叶斯公式推导：$$\begin{align}q(x_{t-1}|x_t,x_0)=&\frac{q(x_{t-1},x_t,x_0)}{q(x_t,x_0)}\\=&\frac{q(x_t|x_{t-1},x_0)q(x_{t-1},x_0)}{q(x_t|x_0)q(x_0)}\\=&\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)q(x_0)}{q(x_t|x_0)q(x_0)}\\\triangleq&\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\end{align}\tag{13}$$利用正向加噪过程中的结论整理这三项因式：$$q(x_t|x_{t-1},x_0)=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_{t-1}\sim N(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t))\tag{14}$$$$q(x_t|x_0)=\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon\sim N(x_t;\sqrt{\overline{\alpha_t}}\,x_0,(1-\overline{\alpha_t}))\tag{15}$$$$q(x_{t-1}|x_0)=\sqrt{\overline{\alpha_{t-1}}}\,x_0+\sqrt{1-\overline{\alpha_{t-1}}}\epsilon\sim N(x_{t-1};\sqrt{\overline{\alpha_{t-1}}}\,x_0,(1-\overline{\alpha_{t-1}}))\tag{16}$$根据高斯分布的概率密度函数$f(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$以及指数运算性质：$$\begin{align}q(x_{t-1}|x_t,x_0)\propto& exp(-\frac{1}{2}(\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{1-\alpha_t}\\&+\frac{(x_{t-1}-\sqrt{\overline{\alpha_{t-1}}}\,x_0)^2}{1-\overline{\alpha_{t-1}}}\\&-\frac{(x_t-\sqrt{\overline{\alpha_t}}\,x_0)^2}{1-\overline{\alpha_t}}))\\=&exp(-\frac{1}{2}((\frac{x_t^2-2\sqrt{\alpha_t}x_tx_{t-1}+\alpha_tx_{t-1}^2}{\beta_t})\\&+(\frac{x_{t-1}^2-2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_0+\overline{\alpha_{t-1}}x_0^2}{1-\overline{\alpha_{t-1}}})\\&-(\frac{(x_t-\sqrt{\overline{\alpha_t}}\,x_0)^2}{1-\overline{\alpha_t}})))\\=&exp(-\frac{1}{2}((\frac{\alpha_t}{\beta_t}+\frac{1}{1-\overline{\alpha_{t-1}}})x_{t-1}^2-\\&(\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\overline{\alpha_{t-1}}}}{1-\overline{\alpha_{t-1}}}x_0)x_{t-1}+C(x_t,x_0)))\end{align}\tag{17}$$上式巧妙地将反向过程转化为正向过程，且最终得到的概率密度函数与正态分布的指数部分：$$exp(-\frac{(x-\mu)^2}{2\sigma^2})=exp(-\frac{1}{2}(\frac{1}{\sigma^2}x^2-\frac{2\mu}{\sigma^2}x+\frac{\mu^2}{\sigma^2}))\tag{18}$$相对应，其中$C(x_t,x_0)$是与$x_{t-1}$无关的常数项。令：$$q(x_{t-1}|x_t,x_0)=N(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta_t}\textbf{I})\tag{19}$$观察对比式(17)和(18)可得式(19)的参数可得：$$\tilde{\beta_t}=\frac{1}{\frac{\alpha_t}{\beta_t}+\frac{1}{1-\overline{\alpha_{t-1}}}}=\beta_t\cdot\frac{1-\overline{\alpha_{t-1}}}{1-\overline{\alpha_t}}\tag{20}$$$$\begin{align}\tilde{\mu}_t(x_t,x_0)=&\frac{1}{2}\frac{\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\overline{\alpha_{t-1}}}}{1-\overline{\alpha_{t-1}}}x_0}{\frac{\alpha_t}{\beta_t}+\frac{1}{1-\overline{\alpha_{t-1}}}}\\=&\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_t}}x_t+\frac{\beta_t\sqrt{\overline{\alpha_{t-1}}}}{1-\overline{\alpha_t}}x_0\end{align}\tag{21}$$根据式(9)利用$x_t$表示$x_0$有$x_0=\frac{1}{\sqrt{\overline{\alpha_t}}}(x_t-\sqrt{1-\overline{\alpha_t}}\epsilon)$代入式(21)化简得：$$\tilde{\mu}_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon)\tag{22}$$.
### 训练与推理
- 综上，我们推导出了$q(x_{t-1}|x_t,x_0)=N(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta_t}\textbf{I})$中的各项参数，接下来还需要训练参数模型$p_{\theta}(x_{t-1}|x_t)=N(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t))$来估计拟合$q(x_{t-1}|x_t,x_0)$。注意：
	- $\tilde{\beta_t}$为给定$\beta_t$取值规律下的定值，理论上我们还需要预测方差，但实际上DDPM并没有使用神经网络预测方差，而是直接假定$q(x_{t-1}|x_t,x_0)$和$p_{\theta}(x_{t-1}|x_t)$具有相同的方差，即$\Sigma_{\theta}(x_t,t)=\tilde{\beta_t}\textbf{I}$
	- 我们仅规定加噪过程中的噪声服从正态分布，而实际上去噪过程中每次去除的噪声不一定服从正态分布。为此，设计[噪声预测器](#^3b8e67)，将$\tilde{\mu}_t(x_t,x_0)$表达式中本次应该去除的噪声$\epsilon$定义为输入图像$x_t$和剩余去噪次数$t$的参数函数$\epsilon_{\theta}(x_t,t)$得到参数模型$p_{\theta}(x_{t-1}|x_t)$中均值的函数：$$\mu_{\theta}(x_t,t)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_{\theta}(x_t,t))\tag{23}$$事实上，该函数含义符合上述噪声预测器的[内部结构](#^da9208)，即先预测输入图像中的噪声，再取差值间接地实现去噪效果。进而，整理参数模型的表达式：$$p_{\theta}(x_{t-1}|x_t)=N(x_{t-1};\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_{\theta}(x_t,t)),\tilde{\beta_t}\textbf{I})\tag{24}$$
- 至此，便可以根据[多元正态分布的KL散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)推导损失函数：$$\begin{align}L_t=&\mathbb{E}_{x_0,\epsilon}[\frac{1}{2||\Sigma_{\theta}(x_t,t)||^2}||\tilde{\mu}_t(x_t,x_0)-\mu_{\theta}(x_t,t)||^2]\\=&\mathbb{E}_{x_0,\epsilon}[\frac{1}{2||\Sigma_{\theta}(x_t,t)||^2}\\&||\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon)-\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}\epsilon_{\theta}(x_t,t))||^2]\\=&\mathbb{E}_{x_0,\epsilon}[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\overline{\alpha_t})||\Sigma_{\theta}(x_t,t)||^2}||\epsilon-\epsilon_{\theta}(x_t,t)||^2]\\=&\mathbb{E}_{x_0,\epsilon}[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\overline{\alpha_t})||\Sigma_{\theta}(x_t,t)||^2}||\epsilon-\epsilon_{\theta}(\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon,t)||^2]\end{align}\tag{25}$$[Ho et al. 2020](https://arxiv.org/pdf/2006.11239)通过实验发现忽略权重部分效果更好，简化损失函数：$$\begin{align}L_t^{simple}=&\mathbb{E}_{x_0,\epsilon}[||\epsilon-\epsilon_{\theta}(x_t,t)||^2]\\=&\mathbb{E}_{x_0,\epsilon}[||\epsilon-\epsilon_{\theta}(\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon,t)||^2]\end{align}\tag{26}$$
- 解释DDPM提出的算法
	- 训练算法：以公式（26）为损失函数执行梯度下降策略
	- 采样算法：首先，从标准正态分布中采样$x_T$，然后进入反向去噪过程。每次去噪先从标准正态分布中采样$z$用作重参数化，根据参数模型的表达式(24)计算$x_{t-1}$的标准差$\sigma_t=\sqrt{\tilde{\beta_t}}$，再结合标准正态分布的样本$z$重采样实现$x_t$去噪变成$x_{t-1}$。重复以上过程直至生成$x_0$
	![](/images/gdm/Pasted%20image%2020240626142017.png)
## GDM用于网络优化的动机
- 广泛的研究例证了扩散模型在处理传统生成领域之外的复杂问题时可提供有效的解决方案，启发我们将其用于智能网络优化。具体来说，应用GDM优化网络的动机源自其独特的特性和能力
	- 稳健的生成能力：GDM具有处理复杂数据分布和生成高质量样本的能力。尤其是应对动态网络优化普遍缺少专家数据集的问题时，它可以凭借反向去噪网络最大/小化给定的优化目标函数(速率、延迟或者能效等关键性能指标)生成解决方案，然后基于接收到的反馈调整网络参数解决该问题
	- 条件信息整合能力：智能网络的优化方案通常随动态无线环境(诸如路径损耗和小尺度衰落信道参数)改变。将它们作为去噪过程中的条件信息，经过充分训练去噪网络可在任何动态无线环境条件下生成最优解决方案
	- GDM与DRL互补增强
		- 一方面，GDM训练去噪网络过程中受到外部环境的反馈指导体现出强化学习范式。可采用Q网络技术促进去噪网络有效训练
		- 另一方面，可以利用GDM强大的生成能力增强各种DRL算法性能或用GDM替代DRL算法中的行动网络，将DRL中的行动看作去噪过程的输出
## 案例教程
### 问题表述
- 考虑无线通信网络中一个总发射功率为$P_T$的基站用于通过多个正交信道为一组用户提供服务
	- 目标是通过在信道之间最佳地分配功率来最大化所有信道的总和速率
	- $g_n$表示第$n$个信道的信道增益
	- $p_n$表示分配给第$n$个信道的功率
	- 所有$M$个正交信道的总和速率为它们各速率之和：$$\sum_{m=1}^M{log_2\,(1+g_mp_m/N_0)}\tag{27}$$通常设置噪声级别$N_0=1$，$\{p_1,...,p_M\}$为一组功率分配策略，在非负性约束和功率预算约束下最大化总传输速率$C$，问题表述为：$$\begin{align}\underset{\{p_1,...,p_M\}}{max}&{C=\sum_{m=1}^Mlog_2\,(1+g_mp_m)}\\&s.t.,\,\,\,\begin{cases}p_m\geq0,\forall m\\\sum_{m=1}^Mp_m\leq P_T\\\end{cases}\end{align}\tag{28}$$
- 无线环境的动态性质导致信道增益在一定范围内波动，带来以下问题
	- 传统数学方案依赖准确的信道估计，然而导频信号和执行算法消耗资源较多，并且会引入延迟
	- 启发式算法可获得近似最优解，但是求解过程会涉及多轮迭代，同样面临资源消耗和延迟问题
	- water-filling算法可较好地解决该问题，但此过程可能是计算密集型的，因此不适合处理大规模信道功率分配问题
### GDM方案
- 使用GDM解决该问题的步骤如下
  1. 解空间定义：解向量维数=信道数$M$![](/images/gdm/Pasted%20image%2020240626210411.png)
  2. 目标函数定义：最大化由GDM产生的功率分配实现的速率总和，其上界可利用water-filling算法求得
  3. 动态环境定义：考虑一种普遍情况，各信道增益在一定范围内随机波动且服从均匀分布，实际情况中均匀分布可替代为瑞利分布、莱斯分布...![](/images/gdm/Pasted%20image%2020240626210436.png)
  4. 训练与推理：GDM通过在给定环境$g$下对初始分布降噪来生成最优功率分配方案$p$。解生成网络$\epsilon_{\theta}(p|g)$按照上述的目标函数将环境状态映射到最优功率分配方案，其中$\theta$为神经网络参数
  ![](/images/gdm/Pasted%20image%2020240626204005.png)
- 根据是否有可用专家数据集，有两种方式训练$\epsilon_{\theta}$
	- 无可用专家数据集：引入解决方案评估网络$Q_v$将代表期望目标函数的$Q$值分配给环境$g$-功率分配方案$p$对，在$Q_v$网络的指导下训练解决方案生成网络。最优的解决方案生成网络应该能生成给定环境$g$下具有最高$Q$值的功率分配策略$p_0$，故最优解决方案生成网络可通过下式计算：$$\underset{\epsilon_{\theta}}{arg\,min}\,\,\mathcal{L}_{\epsilon}(\theta)=-\mathbb{E}_{p_0\sim\epsilon_{\theta}}[Q_v(g,p_0)]\tag{29}$$其中$Q_v$网络的训练目标是最小化当前网络预测$Q$值与实际$Q$值间的差异，因此$Q_v$网络的优化目标为：$$\underset{Q_v}{arg\,min}\,\,\mathcal{L}_{Q}(v)=\mathbb{E}_{p_0\sim\pi_{\theta}}[||r(g,p_0)-Q_v(g,p_0)||^2]\tag{30}$$其中$r$表示在环境$g$中执行生成的功率分配方案$p_0$时的目标函数值
		- 注：此处可采用双$Q-learning$技术避免过度估计
	- 有可用专家数据集：可按照如下方式设计损失函数以最小化生成功率分配策略和专家方案之间的差距：$$\underset{\pi_{\theta}}{arg\,min}\,\,\mathcal{L}(\theta)=\mathbb{E}_{p_0\sim\pi_{\theta}}[||r(g,p_0)-r_{exp}(g)||^2]\tag{31}$$其中$r_{exp}(g)$是给定环境$g$下的目标函数值。为实现高效训练，可采用式(26)类似的损失函数通过正向扩散和反向去噪训练GDM，此时损失函数可写作：$$\underset{\pi_{\theta}}{arg\,min}\,\,\mathcal{L}(\theta)=\mathbb{E}[||\epsilon-\epsilon_{\theta}(\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon,t,g)||^2]\tag{32}$$其中$x_0$表示专家方案，$\epsilon$为添加的高斯噪声，$\sqrt{\overline{\alpha_t}}\,x_0+\sqrt{1-\overline{\alpha_t}}\epsilon$表示经正向加噪后被破坏的专家方案，$t,g$分别为步次信息和环境条件。$\epsilon_{\theta}$用于精确地预测该输入条件下应该去除的噪声
	![](/images/gdm/Pasted%20image%2020240626220941.png)
- 值得注意的是，算法3是为特定环境条件下获得最优解的场景而设计的。但在智能网络中，执行解决方案后可能无法立即获得目标函数值(如服务提供商选择问题需经过较长时间才能得到用户的总效用)，该情况可通过调整算法3的7-13行将决策过程建模为马尔科夫链处理
.
### 实验效果
- 将通过执行GDM在训练过程中产生的功率分配方案获得的总和速率表示为测试总和速率，将使用water-filling算法得到的总和速率称为最大总和速率
- 条件：信道数$M=3$，信道增益随机选择在0.5到2.5之间，去噪步数设置为9
	- 两种学习速率下的GDM方法均优于基于DRL的PRO方法，表现为测试总和速率和最大总和速率之间的差距波动小、收敛快。其中更高的学习速率带来更快的收敛速度，但是波动相比学习速率低者更大，说明较低的学习速率可实现更彻底的学习![](/images/gdm/Pasted%20image%2020240626213057.png)
	- 测试GDM在不同随机种子下的性能表现，实验表明在50个时间步左右三种随机种子下的GDM性能均趋于稳定，说明GDM方法具有较强的鲁棒性![](/images/gdm/Pasted%20image%2020240626213651.png)
- 条件：信道数$M=5$，信道增益随机选择在0.5到5之间
	- 实现表明GDM方法在收敛程度和收敛速度方面均优于SAC和PRO![](/images/gdm/Pasted%20image%2020240626214218.png)
	- 研究去噪步数对GDM性能的影响，去噪步数过高可能会因为过度去噪导致训练数据过拟合，失去有效探索环境的能力；去噪步数过少可能会因为去噪不足导致生成的分配方案具有较强的不确定性。因此需要平衡充分去噪和探索环境的能力，仔细选择GDM的去噪步数![](/images/gdm/Pasted%20image%2020240626214758.png)
- 条件：信道数$M=71$，信道增益随机选择在2到25之间
	- 研究有无可用专家数据集时GDM的性能，训练过程中使用专家数据集可以显著加快收敛速度，而在五专家数据集时GDM方法依然可以有效缩小测试总和速率和最大总和速率间的差距。无论有无专家数据集，其性能均强于随机分配和平均分配策略![](/images/gdm/Pasted%20image%2020240626215258.png)
	- GDM从高斯噪声生成功率分配方案的可视化过程，子图(a)到(e)展现了通过去噪对功率分配方案逐步改进且最终接近最优解的过程。实验表明GDM方法实现的测试总和速率与最大总和速率间的差距仅为$0.11bit/s/Hz$，强调了GDM学习专家方案的有效性以及对复杂模式识别和模仿的能力![](/images/gdm/Pasted%20image%2020240626215523.png)