---
categories: AI
publish: true
excerpt: ''
autoExcerpt: true
tags:
  - AI
  - Algorithm
status: 已发布
pinned: true
public: true
urlname: decoder-transformer
title: Decoder Only Transformer
date: '2026-04-13 01:00:00'
updated: '2026-04-13 01:29:00'
---

## Decoder-Only Transformers: The Workhorse of Generative LLMs

# Decoder-Only Transformer 详细解读


## 1. 总体框架


这篇文章要解释的是：为什么现代大语言模型大多仍然建立在 decoder-only transformer 之上，以及这个架构内部到底由哪些部件组成。


最核心的主线可以概括为：


text -> tokenizer -> token ids -> token embeddings + position information -> transformer blocks -> logits over vocabulary -> next-token prediction


也就是说，文本先被切成 token，再被映射成向量；这些向量经过多层 transformer block；最后每个位置输出一个对整个词表的概率分布，用来预测下一个 token。


## 2. 基本记号


$$
B = batch size \\
T = sequence length \\
d = model dimension \\
H = number of heads \\
d_h = d / H \\
V = vocabulary size
$$


输入张量记作 X，形状是 [B, T, d]。


## 3. Self-Attention 的核心思想


self-attention 的本质不是“比较 token 相似度”，而是：


对序列中的每个位置，决定它应该从哪些位置取信息，以及分别取多少。


self-attention 的第一步，是对输入序列里的每个 token 向量，各做三次不同的线性变换。


这三次变换分别生成：

- query 向量序列
- key 向量序列
- value 向量序列

所以，原来只有一份输入 token 序列，经过三套不同参数的映射后，变成了三份新序列


为此，输入 X 会先被投影成三组向量：(输入乘上参数矩阵，本质就是算参数矩阵)


$$
Q = XW_Q\
K = XW_K\
V = XW_V
$$


这里可以这样理解：


Q 表示“我在寻找什么信息”
K 表示“我这里有什么信息可供匹配”
V 表示“如果你关注我，你最终真正读取什么内容”


所以 attention 的流程其实就是：


“用 query 和每个 key 比一比，看它和谁更匹配；匹配越高，对应的 value 权重越大；最后把所有 value 按权重加权求和，得到输出。”


训练会自动推动参数朝着“更能降低任务损失”的方向更新。


于是模型最终学会了一种内部表示方式，使得：

- query 和 key 的点积能有效反映相关性
- value 能携带对当前任务有帮助的信息

## 4. 注意力分数


每个位置 i 会用自己的 query 去和所有位置的 key 做点积，从而得到匹配分数。


矩阵形式写成：SCORE


$S(q,k) = QK^T$


为了让训练更稳定，实际使用的是缩放点积注意力：


$S(q.k) = \frac{QK^T}{\sqrt{d_h}}$


然后对每一行做 softmax：


$A = \mathrm{softmax}(S(q.k))$


这里 A 的每一行都表示：


当前位置在更新自己时，对所有位置分别分配了多少注意力权重。


## 5. 为什么输出是 AV


注意力的输出不是直接对输入 X 做加权，而是对 V 做加权汇总：


$Y=AV$


这很关键，因为 attention 的逻辑是：


先用 Q 和 K 算出“该看谁”
再用这些权重去加权 V
最后得到新的表示 Y


## 6. Causal Self-Attention


decoder-only transformer 不能看未来 token，因为它的训练目标是根据前文预测后文。


所以在 softmax 之前，需要对注意力分数矩阵加 mask：


对角线以上的所有元素都设为 $-\infty$


然后再做：


$A = \mathrm{softmax}(S_{\mathrm{mask}})$


这样一来，第 i 个 token 只能看见 1 到 i 的位置，不能看见未来位置。


这就是 causal mask，也是 autoregressive language modeling 成立的结构条件。


## 7. Multi-Head Attention


单头 attention 往往不够，因为模型可能需要同时捕捉多种不同关系。


所以实际做法是使用多头注意力。第 h 个头写成：


$\mathrm{head}_h = \mathrm{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_h}}\right)V_h$


所有头计算完成后，再拼接起来：


$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_H)W_O$


它的直觉是：


不同的 head 可以在不同的子空间中学习不同类型的依赖关系。


## 8. LayerNorm


transformer 中还需要归一化来稳定训练。最经典的是 LayerNorm：


$\mathrm{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$


这里的均值和方差是在最后一个维度上计算的，也就是 embedding 维度。


它的作用不是增加表达能力，而是让深层网络更容易训练。


## 9. Feed-Forward Network


attention 不是 block 的全部。每个 transformer block 里还有一个前馈网络 FFN。


标准写法是：


$\mathrm{FFN}(x) = W_2 \sigma(W_1x + b_1) + b_2$


其中最常见的维度变化是：


$W_1 : d \to 4d$


$W_2 : 4d \to d$


这里的含义是：


attention 负责跨位置的信息交互
FFN 负责对每个位置的表示做更复杂的非线性变换


## 10. Residual Connection


为了让深层网络能稳定训练，每个子层外面都加残差连接：


$y = x + f(x)$


它的含义是：


每一层不是推翻上一层，而是在上一层表示的基础上做增量修正。


## 11. 一个标准 Transformer Block


把前面的部件组合起来，一个标准的 decoder-only transformer block 可以写成：


$$x' = x + \mathrm{MHA}(\mathrm{LN}_1(x))$$


$$y = x' + \mathrm{FFN}(\mathrm{LN}_2(x'))$$


这两条式子非常重要，因为它们几乎概括了整个 block 的核心：


先做跨位置的信息聚合
再做单位置的非线性变换


## 12. 输入构造


输入文本首先被 tokenizer 切成 token，然后映射成 token embedding。


记第 t 个 token 的 embedding 为：


$$e_t = E_{\mathrm{tok}}(x_t)$$


位置 embedding 记为：


$$p_t = E_{\mathrm{pos}}(t)$$


那么输入层的表示就是：


$$h_t^{(0)} = e_t + p_t$$


这一步表示：


模型不仅要知道“这是什么 token”
还要知道“它在第几个位置”


## 13. 输出层


经过多层 transformer block 后，得到每个位置的 hidden state。


对第 t 个位置，输出层写成：


$$\mathrm{logits}_t = h_t W_{\mathrm{vocab}} + b$$


再把它变成概率分布：


$$p(x_{t+1} \mid x_{\le t}) = \mathrm{softmax}(\mathrm{logits}_t)$$


这表示：


当前位置的 hidden state 用来预测下一个 token。


## 14. 训练目标


decoder-only language model 的训练目标是 next-token prediction。


最基本的损失函数写成：


$$L = -\sum_t \log p(x_{t+1} \mid x_{\le t})$$


这条式子可以理解为：


让模型尽可能给真实的下一个 token 更高的概率。


## 15. 整体前向传播


整个模型的逻辑可以压缩成下面这条链：


text
-> tokenizer
-> token ids
-> token embeddings + positional embeddings
-> repeated transformer blocks
-> final normalization
-> linear vocabulary head
-> next-token distribution


## 16. RMSNorm


现代 LLM 常常把 LayerNorm 换成 RMSNorm。


它的写法是：


$$\mathrm{RMSNorm}(x) = g \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$


和 LayerNorm 相比，它更轻量，也常常更高效。


## 17. MQA 和 GQA


现代模型还会对注意力结构做效率优化。


最基础的多头注意力是：


每个 head 都有自己的 K 和 V


MQA 的做法是：


所有 head 共享同一套 K 和 V


GQA 的做法是：


把多个 head 分组，每组共享一套 K 和 V


所以可以简单记成：


Vanilla MHA: each head has its own K and V
MQA: all heads share one K and one V
GQA: heads are grouped and share K/V within each group


## 18. RoPE


RoPE 是一种现代常用的位置编码方法。


在理解上，不必强行记复杂旋转矩阵，只要抓住一点：


RoPE 不是只在输入层加一次位置向量，
而是把位置信息直接注入每一层 self-attention。


它的优势在于：


同时携带绝对位置和相对位置信息，
并且通常更适合更长上下文。


## 19. ALiBi


ALiBi 也是一种位置建模方案。


它不显式使用 position embeddings，而是直接给 attention 分数加偏置：


$$S' = S + B$$


其中 B 是和相对距离有关的线性偏置矩阵。


可以把它理解成：


越远的位置会被施加更大的惩罚。


## 20. 最重要的结论


这篇文章最重要的结论可以压缩成一句话：


现代生成式大语言模型的核心仍然是 decoder-only transformer。
它通过 causal self-attention 从左侧上下文取信息，
通过 FFN 做局部非线性变换，
通过 LayerNorm 或 RMSNorm 与残差连接稳定训练，
再通过词表输出层完成 next-token prediction。


而 FlashAttention、MQA、GQA、RoPE、ALiBi 等现代改进，
本质上都是围绕这个基础骨架做效率、稳定性和长上下文能力优化。

