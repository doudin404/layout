
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


def sample(model, x, masks, once=False,max_steps=0,temperature=1.0, sample=False, top_k=None, y=None):
    """从模型的结果中采样的函数"""
    model.eval()
    weights = masks.clone()
    total_dim = 7
    layout_dim = 2
    position_ids = torch.arange(x.shape[-1])[None, :].to(x.device)
    asset_ids= position_ids//total_dim

    max_value = torch.inf * \
    torch.ones(
        x.shape[:-1]).unsqueeze(-1).to(x.device)
    
    min_value = torch.inf * \
    torch.ones(
        x.shape[:-1]).unsqueeze(-1).to(x.device)

    with torch.no_grad():
        if once:  # 一次采样
            logits, _ = model(x)

            probs = F.softmax(logits / temperature, dim=-1)

            confidence, choose = choose_from_probs(probs, sample, top_k)

            x = torch.where(weights, choose, x)
        else:  # 从两侧多次采样
            step=0
            while weights.any():
                step+=1
                n = weights.sum(dim=-1)
                if max_steps==0:
                    step_length=torch.where(n>0,1,0)
                else:
                    step_length=torch.ceil(n/(max_steps-step+1)).type(torch.int64)
                sorted_weights,_ = torch.sort(torch.where(weights,asset_ids,torch.inf)) # 对weights进行排序,并用inf替换0元素
                sorted_weights = torch.cat(#这一步是为了防止step_length出界
                    (min_value,sorted_weights), dim=-1)
                kth_smallest = torch.gather(sorted_weights, 1, step_length.unsqueeze(-1)) # 根据step_length中的值获取sorted_weights中对应位置的元素
                sorted_weights,_ = torch.sort(torch.where(weights,asset_ids,-torch.inf), descending=True) # 对weights进行降序排序，并用-inf替换0元素
                sorted_weights = torch.cat(#这一步是为了防止step_length出界
                    (max_value,sorted_weights), dim=-1)
                kth_largest = torch.gather(sorted_weights, 1, step_length.unsqueeze(-1)) # 根据step_length中的值获取sorted_weights中对应位置的元素

                mask_to_fill=((asset_ids>=kth_largest)|(asset_ids<=kth_smallest))&weights

                logits, _ = model(x, y)

                probs = F.softmax(logits / temperature, dim=-1)

                confidence, choose = choose_from_probs(
                    probs, sample, top_k)
                confidence[~mask_to_fill] = 0

                sorted_confidence, _ = confidence.sort(
                    dim=-1, descending=True)

                sorted_confidence = torch.cat(#这一步是为了防止step_length出界
                    (max_value,sorted_confidence), dim=-1)
                

                selected_confidence = sorted_confidence.gather(
                    -1, step_length.unsqueeze(-1))

                selected_indices = (confidence >= selected_confidence)

                x = torch.where(selected_indices, choose, x)
                weights &= (~selected_indices)
    model.train()
    return x