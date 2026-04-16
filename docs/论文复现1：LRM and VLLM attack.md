---
categories: AI
publish: true
excerpt: ''
autoExcerpt: true
tags:
  - AI
  - Algorithm
  - LLM Safety
status: 已发布
pinned: true
public: true
urlname: lrm-vllm-attack
title: 论文复现1：LRM and VLLM attack
date: '2026-04-16 19:24:00'
updated: '2026-04-17 00:04:00'
---

## 论文复现1：LRM and VLLM attack

Appendix里一般都会有非常详细的实验设置，结合论文结果所展示的参数以及实验和方法部分的实现，基本可以完成复现。


截至2026.4.16，LRM复现主要是一个play性质，没有严格按照appendix70轮setting和eval，evaluation没有严格按照evaluation prompt原文，否则时间和api调用过多，缺乏实验所需api。VLLM复现是尝试性质，没有使用原数据集（原数据集过大），也只跑了50items而不是1000，eval model是反代copilot api调用不是官方API，vqa和text只有较为初步的可行性demo，不具有结果参考价值。


# _Large Reasoning Models Are Autonomous Jailbreak Agents_.


Hagendorff, Thilo, et al. “Large Reasoning Models Are Autonomous Jailbreak Agents.” Nature Communications, vol. 17, no. 1, Feb. 2026. Crossref, [https://doi.org/10.1038/s41467-026-69010-1](https://doi.org/10.1038/s41467-026-69010-1).


[link_preview](https://github.com/fycorex/attack-lrm)


## 复现重点


论文是想表达越狱攻击可以简单到不需要”complex prompt instructions, finetuning, or steering conversation behavior.”


事实上直接拿文章里的prompt和模型对话就行，但如果要做实验需要写pipeline。

- Prompt
    - appendix里面给出的system prompt和evaluation prompt，Scoring scale很重要，最重要的依然是system prompt，可能直接关系攻击效果，eval影响结果可信度和可解释性
    - 后面很多页的dialogue没那么有用
- 攻击的多轮对话规划
    - attack strategies are not predefined
    - target model have conversation history
- LRM
    - 虽然现在大部分商业模型都是reasoning first了

这个pipeline写的相当简陋，主要是我觉得写这个意义不大


# _Transferable Adversarial Attacks on Black-Box Vision-Language Models_


Hu, K., Yu, W., Zhang, L., Robey, A., Zou, A., Xu, C., Hu, H., & Fredrikson, M. (2025). _Transferable Adversarial Attacks on Black-Box Vision-Language Models_ (arXiv:2505.01050). arXiv. [https://doi.org/10.48550/arXiv.2505.01050](https://doi.org/10.48550/arXiv.2505.01050)


[link_preview](https://github.com/fycorex/attack-vllm)


这个论文关注图像攻击，不用直接攻击目标 VLM 本身，而是先用一组代理视觉模型（surrogates）来优化扰动，让对抗图像在语义嵌入空间里更接近攻击者指定的“目标类别/目标答案”，再把这个扰动迁移到 GPT-4o、VQA 模型、OCR 模型等黑盒受害模型上测试


不过不得不吐槽的一点是后面text攻击为了成功率不要打的太低都有32/255的ε，这个图像其实已经模糊得有点没法看了，实际上较为隐蔽得ε大概在8/255，但可能结果打出来都是零蛋或者太小。


## 复现重点


复现主要是按照论文里的最好实践复现的，真实实验过程中不可能提前知道最好实践，就需要像论文一样控制变量分析，跑多组不同config和方法组合的实验

- 代理模型
    - 复现用了8 个 surrogate：4 个 ViT-H、3 个 SigLIP、1 个 ConvNeXt XXL（大致需要20G显存，复现时使用的模型显存总共21G，A6000可以并行跑两组ε的实验，50item*300step*299image_size预计10-12小时）
- 攻击策略
    - 先为每个 surrogate 预计算正例、负例和干净图像的嵌入，然后对扰动δ做迭代优化；每一步都把扰动限制在L无穷的 ε 球内，最后再生成 adversarial image
- Loss function
    - top_k=10（正样本）
    - 视觉对比损失（攻击图像更靠近 target 的正样本图像嵌入，同时远离 source 的负样本嵌入）
    - relative_proxy_loss，这个我没调
- 增强策略
    - 高斯噪声、随机裁剪、pad + resize、JPEG 压缩
- caption VQA 和 text的不同数据集

Loss:


```shell
def visual_contrastive_loss(
    image_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    temperature: float,
    top_k: int,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_logits = image_embeddings @ positive_embeddings.t()
    negative_logits = image_embeddings @ negative_embeddings.t()

    all_logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    log_probs = torch.log_softmax(all_logits, dim=1)

    pos_log_probs = log_probs[:, : positive_embeddings.shape[0]]
    neg_log_probs = log_probs[:, positive_embeddings.shape[0] :]

    k = max(1, min(top_k, positive_embeddings.shape[0]))
    topk_positive = torch.topk(pos_log_probs, k=k, dim=1).values

    loss = -topk_positive.mean() + neg_log_probs.mean()
    if not collect_metrics:
        return loss, {}
    metrics = {
        "positive_logprob_mean": float(pos_log_probs.mean().detach().cpu()),
        "negative_logprob_mean": float(neg_log_probs.mean().detach().cpu()),
        "topk_positive_logprob_mean": float(topk_positive.mean().detach().cpu()),
    }
    return loss, metrics


def relative_proxy_loss(
    clean_image_embeddings: torch.Tensor,
    adversarial_image_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    top_k: int,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_positive_logits = clean_image_embeddings @ positive_embeddings.t()
    clean_negative_logits = clean_image_embeddings @ negative_embeddings.t()
    adversarial_positive_logits = adversarial_image_embeddings @ positive_embeddings.t()
    adversarial_negative_logits = adversarial_image_embeddings @ negative_embeddings.t()

    k = max(1, min(top_k, positive_embeddings.shape[0]))
    clean_positive_topk = torch.topk(clean_positive_logits, k=k, dim=1).values
    adversarial_positive_topk = torch.topk(adversarial_positive_logits, k=k, dim=1).values

    positive_gain = adversarial_positive_topk.mean() - clean_positive_topk.mean()
    negative_shift = adversarial_negative_logits.mean() - clean_negative_logits.mean()
    loss = -positive_gain + negative_shift
    if not collect_metrics:
        return loss, {}
    metrics = {
        "relative_positive_gain": float(positive_gain.detach().cpu()),
        "relative_negative_shift": float(negative_shift.detach().cpu()),
    }
    return loss, metrics
```

