from typing import Optional, Tuple
import torch

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    # [b, dim]
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: Optional[torch.Tensor] = None,
    # [b, window_size]
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[:, 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


# 采样
def sample(
    # 要采样的 tensor [b, dim]
    logits,
    # 采样 temperature
    temperature: torch.Tensor,
    # 对前 top 的数据进行采样
    # 如 top = 0.5， logits 对应元素的概率为 [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]，则只会从第一和第二个元素中进行采样
    top_p: torch.Tensor,
    # 重复惩罚
    # 会减少当前惩罚窗口内的 token 的采样概率
    repetition_penalty: Optional[torch.Tensor] = None,
    # 重复惩罚窗口
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


if __name__ == '__main__':
    device = 'cuda'
    # 采样参数
    temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    top_p = torch.tensor(0.5, device=device, dtype=torch.float)
    # 重复惩罚
    repetition_penalty = torch.tensor(
        1.1, device=device, dtype=torch.float
    )

    logits = torch.tensor([
        [0.3, 0.2, 0.15, 0.15, 0.1, 0.1],
        [0.15, 0.15, 0.1, 0.1, 0.3, 0.2]
    ], dtype=torch.float, device=device)

    previous_tokens = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    ], dtype=torch.long, device=device)

    for i in range(10):
        result = sample(logits, temperature, top_p, repetition_penalty, previous_tokens)
        print(result[0])
    pass