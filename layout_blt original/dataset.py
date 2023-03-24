import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps, ImageFont
import seaborn as sns
import json
import os
from torch.nn import functional as F


def main():
    dataset = JSONLayout("PubLayNet/val.json")
    for i in range(3):
        random_index = np.random.choice(np.arange(len(dataset.data)))
        random_row = dataset.data[random_index]
        dataset.render(random_row).show()


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)]
                   for x in palette]
    return rgb_triples


class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, percentage=1, precision=8):
        # 取出文件名
        file_name = os.path.basename(json_path)
        # 去掉文件名后缀
        file_name = os.path.splitext(file_name)[0]
        # 将文件名插入模板
        save_path = f"{file_name}.npy"

        if os.path.exists(save_path) and os.path.exists("categories_name.npy"):
            self.data = np.load(save_path)
            self.categories_name = np.load("categories_name.npy")
            self.size = pow(2, precision)
            self.max_length = self.data.shape[-1]
            self.vocab_size = self.size + \
                self.categories_name.shape[-1]+3  # mask, eos, pad tokens

            self.mask_token = self.vocab_size - 3
            self.eos_token = self.vocab_size - 2
            self.pad_token = self.vocab_size - 1

            self.colors = gen_colors(self.categories_name.shape[-1])
        else:
            with open(json_path, "r") as f:
                data = json.loads(f.read())

            self.categories_name = {}
            for category in data['categories']:
                self.categories_name[category["id"]] = category["name"]
            self.categories_name = np.array(
                list(self.categories_name.values()))

            dict_images = {}
            dict_bboxs = {}
            dict_categories = {}
            for image in data['images']:
                dict_images[image["id"]] = [image["width"], image["height"]]
                dict_bboxs[image["id"]] = []
                dict_categories[image["id"]] = []

            for annotation in data['annotations']:
                dict_bboxs[annotation["image_id"]].append(annotation["bbox"])
                dict_categories[annotation["image_id"]].append(
                    annotation["category_id"])

            del data

            self.size = pow(2, precision)
            self.vocab_size = self.size + \
                self.categories_name.shape[-1]+3  # mask, eos, pad tokens

            self.mask_token = self.vocab_size - 3
            self.eos_token = self.vocab_size - 2
            self.pad_token = self.vocab_size - 1

            self.max_length = max_length
            self.percentage = percentage
            if self.max_length is None:
                all_lengths = [len(categories)
                               for categories in dict_categories.values()]
                sorted_lengths = sorted(all_lengths)
                num_to_keep = int(len(sorted_lengths) * self.percentage)
                self.max_length = sorted_lengths[num_to_keep - 1] * 5 + 1

            self.data = []
            for id in dict_bboxs:  # 335703
                bboxs = np.array(dict_bboxs[id])
                images = np.array(dict_images[id])
                categories = np.array(dict_categories[id])
                widths = images[0][np.newaxis, np.newaxis]
                heights = images[1][np.newaxis, np.newaxis]

                bboxs = bboxs / \
                    np.concatenate((widths, heights, widths, heights), axis=-1)
                bboxs = np.round(bboxs * self.size).astype(int)

                if not (np.all((bboxs >= 0) & (bboxs <= self.size - 1))):
                    continue  # 原文中使用了裁剪方案,我选择直接跳过出界的数据
                if not (np.all((bboxs[..., :2] + bboxs[..., 2:] <= self.size - 1))):
                    continue

                indices = np.lexsort((bboxs[:, 0], bboxs[:, 1]))
                bboxs = bboxs[indices]
                categories = categories[indices]+self.size-1

                combined = np.column_stack((categories, bboxs))
                combined = np.append(combined.flatten(), self.eos_token)
                if combined.shape[-1] > self.max_length:
                    continue

                data = np.pad(combined.flatten(), (0, self.max_length - len(
                    combined.flatten())), 'constant', constant_values=(self.pad_token,))
                self.data.append(data)
            self.colors = gen_colors(self.categories_name.shape[-1])

            np.save(save_path, self.data)  # 335685
            np.save("categories_name.npy", self.categories_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

    def render(self, layout):
        img = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = layout[:np.argwhere(layout == self.eos_token)[-1][0]]
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box = box / self.size
        box = box * 1024
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat - \
                self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            font = ImageFont.truetype("arial.ttf", int(20 * 2))
            draw.text((x1, y1), str(i + 1), font=font,
                      fill=(0, 0, 0, 255), align="left")

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img


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
    weights=masks.clone()
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

                    min_value = -1 * \
                        torch.ones(
                            sorted_confidence.shape[:-1]).unsqueeze(-1).to(x.device)
                    sorted_confidence = torch.cat(
                        (sorted_confidence, min_value), dim=-1)

                    selected_confidence = sorted_confidence.gather(
                        -1, n.unsqueeze(-1))

                    selected_indices = torch.where(
                        (confidence >= selected_confidence), True, False) & remaining_mask

                    x = torch.where(selected_indices, choose, x)
                    weights &= (~selected_indices)
    model.train()
    return x


if __name__ == '__main__':
    main()
