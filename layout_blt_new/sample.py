
import torch
from torch.nn import functional as F

def choose_from_probs(probs, sample=False, top_k=None):
    """按照sample,top_k的要求从probs中取样"""
    if sample:
        if top_k:
            topk_probs, topk_choose = probs.topk(top_k, dim=-1)

            flat_probs = topk_probs.reshape(-1, topk_probs.shape[-1])
            flat_indices = torch.multinomial(flat_probs, 1)
            indices = flat_indices.reshape(-1, topk_probs.shape[-2])

            choose = topk_choose.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            confidence = topk_probs.gather(-1,
                                           indices.unsqueeze(-1)).squeeze(-1)
        else:
            flat_probs = probs.reshape(-1, probs.shape[-1])
            flat_indices = torch.multinomial(flat_probs, 1)
            indices = flat_indices.reshape(-1, probs.shape[-2])

            choose = indices
            confidence = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, choose = probs.max(dim=-1)

    return confidence, choose

def sample(model, x, masks, steps=10, temperature=1.0, sample=False, top_k=None, y=None):
    """从模型的结果中采样的函数"""
    model.eval()
    weights = masks.clone()
    total_dim = 5
    layout_dim = 2
    position_ids = torch.arange(x.shape[-1])[None, :].to(x.device)
    is_asset = (position_ids % total_dim == 0)
    is_size = ((position_ids % total_dim >= 1) & (position_ids %
               total_dim < layout_dim + 1)).to(torch.bool)
    is_position = ((position_ids % total_dim >= layout_dim + 1)
                   & (position_ids % total_dim < total_dim))

    with torch.no_grad():
        if steps == 0:  # 一次采样
            logits, _ = model(x)

            probs = F.softmax(logits / temperature, dim=-1)

            confidence, choose = choose_from_probs(probs, sample, top_k)

            x = torch.where(weights, choose, x)
        else:  # 按照论文要求多次采样
            for mask in [is_asset, is_size, is_position]:
                for j in range(steps):
                    remaining_mask = mask & weights
                    if remaining_mask.sum() == 0:
                        break

                    r = 1-(steps - j - 1) / steps
                    n = remaining_mask.sum(dim=-1)
                    n = torch.ceil(r*n).to(torch.int64)

                    logits, _ = model(x, y)

                    probs = F.softmax(logits / temperature, dim=-1)

                    confidence, choose = choose_from_probs(
                        probs, sample, top_k)

                    confidence[~remaining_mask] = 0

                    sorted_confidence, _ = confidence.sort(
                        dim=-1, descending=True)

                    min_value = torch.inf * \
                        torch.ones(
                            sorted_confidence.shape[:-1]).unsqueeze(-1).to(x.device)
                    sorted_confidence = torch.cat(#这一步是为了防止n出界
                        (min_value,sorted_confidence), dim=-1)
                    

                    selected_confidence = sorted_confidence.gather(
                        -1, n.unsqueeze(-1))

                    selected_indices = (confidence >= selected_confidence)

                    x = torch.where(selected_indices, choose, x)
                    weights &= (~selected_indices)
    model.train()
    return x

def sample1(model, x, masks, steps=10, temperature=1.0, sample=False, top_k=None, y=None):
    """从模型的结果中采样的函数"""
    model.eval()
    weights = masks.clone()
    total_dim = 5
    layout_dim = 2
    position_ids = torch.arange(x.shape[-1])[None, :].to(x.device)
    is_asset = (position_ids % total_dim == 0)
    is_size = ((position_ids % total_dim >= 1) & (position_ids %
               total_dim < layout_dim + 1)).to(torch.bool)
    is_position = ((position_ids % total_dim >= layout_dim + 1)
                   & (position_ids % total_dim < total_dim))

    with torch.no_grad():
        if steps == 0:  # 一次采样
            logits, _ = model(x)

            probs = F.softmax(logits / temperature, dim=-1)

            confidence, choose = choose_from_probs(probs, sample, top_k)

            x = torch.where(weights, choose, x)
        else:  # 按照论文要求多次采样
            while weights.any():
                remaining_mask = weights
                if remaining_mask.sum() == 0:
                    break

                r = 0.001
                n = remaining_mask.sum(dim=-1)
                n = torch.ceil(r*n).to(torch.int64)

                logits, _ = model(x, y)

                probs = F.softmax(logits / temperature, dim=-1)

                confidence, choose = choose_from_probs(
                    probs, sample, top_k)

                confidence[~remaining_mask] = 0

                sorted_confidence, _ = confidence.sort(
                    dim=-1, descending=True)

                min_value = torch.inf * \
                    torch.ones(
                        sorted_confidence.shape[:-1]).unsqueeze(-1).to(x.device)
                sorted_confidence = torch.cat(#这一步是为了防止n出界
                    (min_value,sorted_confidence), dim=-1)
                

                selected_confidence = sorted_confidence.gather(
                    -1, n.unsqueeze(-1))

                selected_indices = (confidence >= selected_confidence)

                x = torch.where(selected_indices, choose, x)
                weights &= (~selected_indices)
    model.train()
    return x

def sample2(model, x, masks, steps=10, temperature=1.0, sample=False, top_k=None, y=None):
    """从模型的结果中采样的函数"""
    model.eval()
    weights = masks.clone()
    total_dim = 5
    layout_dim = 2
    position_ids = torch.arange(x.shape[-1])[None, :].to(x.device)
    asset_ids= position_ids//total_dim

    with torch.no_grad():
        if steps == 0:  # 一次采样
            logits, _ = model(x)

            probs = F.softmax(logits / temperature, dim=-1)

            confidence, choose = choose_from_probs(probs, sample, top_k)

            x = torch.where(weights, choose, x)
        else:  # 从两侧多次采样
            while weights.any():
                last,_ = torch.max(torch.where(weights,asset_ids,-1),dim=-1)
                first,_ = torch.min(torch.where(weights,asset_ids,torch.inf),dim=-1)
                n = weights.sum(dim=-1)
                n = torch.ceil(0.001*n).to(torch.int64)

                mask_to_fill=((asset_ids==last.unsqueeze(1))|(asset_ids==first.unsqueeze(1)))&weights

                logits, _ = model(x, y)

                probs = F.softmax(logits / temperature, dim=-1)

                confidence, choose = choose_from_probs(
                    probs, sample, top_k)

                confidence[~mask_to_fill] = 0

                sorted_confidence, _ = confidence.sort(
                    dim=-1, descending=True)

                min_value = torch.inf * \
                    torch.ones(
                        sorted_confidence.shape[:-1]).unsqueeze(-1).to(x.device)

                sorted_confidence = torch.cat(#这一步是为了防止n出界
                    (min_value,sorted_confidence), dim=-1)
                

                selected_confidence = sorted_confidence.gather(
                    -1, n.unsqueeze(-1))

                selected_indices = (confidence >= selected_confidence)

                x = torch.where(selected_indices, choose, x)
                weights &= (~selected_indices)
    model.train()
    return x